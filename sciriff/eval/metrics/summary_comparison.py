import json
import re
from openai import OpenAI, AzureOpenAI
import openai
import random
import time, os

# Call judge LLM to compare model answer to reference.
API_KEY = os.getenv('OPENAI_API_KEY')
CLIENT = OpenAI(api_key=API_KEY)


def create_batch_file(instances, eval_type):
    tasks = []
    for idx, instance in enumerate(instances):
        prompt = make_prompt(instance, eval_type)
        task = {
            "custom_id": f"task-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                # "model": "gpt-4o-mini",
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            }
        }
        tasks.append(task)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("./summary_comparison-logs", exist_ok=True)
    file_name = f"./summary_comparison-logs/{timestr}-batch_tasks.jsonl"
    with open(file_name, 'w') as file:
        for task in tasks:
            file.write(json.dumps(task) + '\n')
    
    return file_name

def submit_batch_job(file_name):
    batch_file = CLIENT.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )
    
    batch_job = CLIENT.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Submitted job ID: {batch_job.id}")
    
    return batch_job.id

def call_lm_judge(prompt, lm_judge_model):
    chat_completion = CLIENT.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=lm_judge_model,
    )
    return chat_completion.choices[0].message.content


def make_prompt(instance, eval_type):
    if eval_type == "model_comparison":
        pred_first = instance["pred_first_in_prompt"]
        prompt = f"""Below you will be shown the title and body text of a computer
        science research paper. Then, you will be given two model-generated summaries of
        the paper, named "Summary1" and "Summary2". Please compare between Summary1 and
        Summary2 on their usefulness for a peer reviewer assessing the paper's quality,
        and pick the better one for that purpose (there can be a tie). The target length
        for the summary should be roughly {instance['target_sents']} sentences; please
        penalize summaries that are much longer or shorter.

        In your response, give an explanation for your rating, followed by your rating.

        Response format:
        Explanation: Your explanation here
        Decision: 1 or 2 or tie
        _Do not_ include any additional text after the rating.

        Here's the paper title and body text:
        Title: {instance['title']}

        Body text:
        {instance['body_text']}

        Here are the summaries:
        Summary1:
        {instance['prediction'] if pred_first else instance['baseline']}

        Summary2:
        {instance['baseline'] if pred_first else instance['prediction']}
        """
    elif eval_type == "reference_free":
        if 'title' in instance:
            prompt = f"""Below you will be shown the title and body text of a computer
            science research paper. Then, you will be given a model-generated summary.
            Please rate the model summary on an overall 1-5 scale. 

            - 5: The model answer includes all important information.
            - 3: The model answer includes important information, but not completely accurate.
            - 1: The model answer is totally inaccurate or is unrelated.
            In your response, give an explanation for your rating, followed by your rating.

            Response format:
            Explanation: "Your explanation here"
            Decision: A single integer between 1 and 5.
            _Do not_ include any additional text after the rating.

            Here's the paper title and body text:
            Title: {instance['title']}

            Body text:
            {instance['body_text']}

            Model summary:
            {instance['prediction']}
            """
        else:
            prompt = f"""Below you will be shown the body text of a computer
            science research paper. Then, you will be given a model-generated summary.
            Please rate the model summary on an overall 1-5 scale. 

            - 5: The model answer includes all important information.
            - 3: The model answer includes important information, but not completely accurate.
            - 1: The model answer is totally inaccurate or is unrelated.
            In your response, give an explanation for your rating, followed by your rating.

            Response format:
            Explanation: "Your explanation here"
            Decision: A single integer between 1 and 5.
            _Do not_ include any additional text after the rating.

            Here's the body text:
            Body text:
            {instance['body_text']}

            Model summary:
            {instance['prediction']}
            """
    else:
        prompt = f"""Below you will be shown the title and body text of a computer
        science research paper. Then, you will be given a reference summary of that
        paper that was written by an expert, followed by a model-generated summary.
        Please rate the similarity of the model summary to the reference summary on an
        overall 1-5 scale. Do not penalize the model for including additional
        information that is not in the reference.

        - 5: The model answer includes all important information found in the reference.
        - 3: The model answer is somewhat similar to the reference, but not completely accurate.
        - 1: The model answer is totally inaccurate or is unrelated to the reference.
        In your response, give an explanation for your rating, followed by your rating.

        Response format:
        Explanation: "Your explanation here"
        Decision: A single integer between 1 and 5.
        _Do not_ include any additional text after the rating.

        Here's the paper title and body text:
        Title: {instance['title']}

        Body text:
        {instance['body_text']}

        Here are the summaries:
        Reference summary:
        {instance['reference']}

        Model summary:
        {instance['prediction']}
        """

    prompt = prompt.replace("        ", "")

    return prompt


def decode_summary_order(explanation, pred_first):
    "Replaces 'summary1 \ summary2' occurances within the lm-judge's explanation with the summary types they represent (prediction / baseline)"

    summary1 = "prediction summary" if pred_first else "baseline summary"
    summary2 = "baseline summary" if pred_first else "prediction summary"
    explanation = explanation.replace("Summary1", summary1).replace(
        "Summary 1", summary1
    )
    explanation = explanation.replace("Summary2", summary2).replace(
        "Summary 2", summary1
    )
    return explanation


def parse_model_response(response, instance, eval_type):
    if "Decision:" in response:
        decision = response[response.find("Decision:") + 9 :].strip()
        explanation = (
            response[: response.find("Decision")].replace("Explanation:", "").strip()
        )
        if eval_type == "model_comparison" and decision in ["1", "2", "tie"]:
            pred_first = instance["pred_first_in_prompt"]
            explanation = decode_summary_order(explanation, pred_first)
            instance["decision_explanation"] = explanation
            if decision == "tie":
                instance["decision"] = "tie"
            else:
                instance["decision"] = (
                    "win"
                    if (
                        (decision == "1" and pred_first)
                        or (decision == "2" and not pred_first)
                    )
                    else "lose"
                )
        elif (eval_type == "reference_comparison" or eval_type == "reference_free") and decision in [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]:
            # Convert to an int between 0 and 1 so we can compare.
            instance["decision"] = (int(decision) - 1) / 4
            instance["decision_explanation"] = explanation
        else:
            instance["decision"] = "error"
    
    elif "Rating:" in response:
        decision = response[response.find("Rating:") + 7 :].strip()
        explanation = (
            response[: response.find("Rating")].replace("Explanation:", "").strip()
        )
        if eval_type == "model_comparison" and decision in ["1", "2", "tie"]:
            pred_first = instance["pred_first_in_prompt"]
            explanation = decode_summary_order(explanation, pred_first)
            instance["decision_explanation"] = explanation
            if decision == "tie":
                instance["decision"] = "tie"
            else:
                instance["decision"] = (
                    "win"
                    if (
                        (decision == "1" and pred_first)
                        or (decision == "2" and not pred_first)
                    )
                    else "lose"
                )
        elif (eval_type == "reference_comparison" or eval_type == "reference_free") and decision in [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]:
            # Convert to an int between 0 and 1 so we can compare.
            instance["decision"] = (int(decision) - 1) / 4
            instance["decision_explanation"] = explanation
        else:
            instance["decision"] = "error"
    else:
        instance["decision"] = "error"


def batch_lm_judge(instances, eval_type, lm_judge_raw=None, lm_judge_mapping=None):
    if not lm_judge_raw.exists():
        file_name = create_batch_file(instances, eval_type)
        batch_job_id = submit_batch_job(file_name)
        with open(lm_judge_mapping, 'w') as f:
            json.dump({'batch_id': batch_job_id}, f)
        # results = get_batch_results(batch_job_id)
        return None
    else:
        results = json.load(open(lm_judge_raw))
    
    ratings = []
    for result, instance in zip(results, instances):
        # response = result['response']['choices'][0]['message']['content']
        # response = result['response']['body']['choices'][0]['message']['content']
        # rating = extract_rating(response)
        # ratings.append(rating)
        response = result['response']['body']['choices'][0]['message']['content']
        parse_model_response(response, instance, eval_type)
        result = instance['decision']
        if result == 'error':
            continue
        if eval_type == "model_comparison":
            results_encoding = {"win": (1, 1), "tie": (0, 1), "lose": (0, 0)}
            ratings.append(results_encoding[result])
        elif eval_type == "reference_comparison" or eval_type == "reference_free":
            ratings.append(result)
    return ratings

def lm_judge(instance, eval_type):
    # lm_judge_model = (
    #     "gpt-4-turbo-preview" if eval_type == "model_comparison" else "gpt-3.5-turbo"
    # )
    lm_judge_model = "gpt-4o-2024-08-06"
    instance["prompt"] = make_prompt(instance, eval_type)
    try:
        response = call_lm_judge(instance["prompt"], lm_judge_model)
        parse_model_response(response, instance, eval_type)
    except openai.BadRequestError:
        instance["decision"] = "error"
    result = instance["decision"]
    if result != "error":
        if eval_type == "model_comparison":
            results_encoding = {"win": (1, 1), "tie": (0, 1), "lose": (0, 0)}
            return results_encoding[result]
        elif eval_type == "reference_comparison":
            return result


# ------------------------- Evaluator -------------------------
class SummaryComparison:

    def _extract_data_from_prompt(self, prompt, data_type):
        if data_type == "title":
            pattern = r"Paper title:\n(.*?)(?=\n)"
            match = re.search(pattern, prompt)
            if match:
                data = match.group(1)
                return data
        elif data_type == "body_text":
            if "Paper body:" in prompt:
                data = prompt[prompt.find("Paper body:") + 11 :]
                if "sentence paper summary" in data:
                    data = data[: data.find("sentence paper summary") - 2]
                data = data.strip()
                return data
        print(f"Could not extract {data_type}.")

    def _evaluate_model_comparison(self, instances):
        "Have a judge LM choose between the model's summary and the baseline model's summary"
        self.scores = {
            "lm_judge_n_samples": 0,
            "lm_judge_wins": 0,
            "lm_judge_wins_and_ties": 0,
        }

        for index, instance in enumerate(instances.values()):
            del instance["reference"]
            instance["pred_first_in_prompt"] = True if index % 2 == 0 else False
            instance["title"] = self._extract_data_from_prompt(
                instance["prompt"], "title"
            )
            instance["body_text"] = self._extract_data_from_prompt(
                instance["prompt"], "body_text"
            )
            # Get the target number of sentences in the summary.
            instance["target_sents"] = int(
                re.search(
                    r"Your summary should be (\d+) sentences long", instance["prompt"]
                ).group(1)
            )
            # results = lm_judge(instance, eval_type="model_comparison")
            # if results:
            #     self.scores["lm_judge_n_samples"] += 1
            #     self.scores["lm_judge_wins"] += results[0]
            #     self.scores["lm_judge_wins_and_ties"] += results[1]
        results = batch_lm_judge(instances.values(), eval_type="model_comparison")
        for result in results:
            self.scores["lm_judge_n_samples"] += 1
            self.scores["lm_judge_wins"] += result[0]
            self.scores["lm_judge_wins_and_ties"] += result[1]

        # calculate win ratio & wins_and_ties ratio
        self.scores["lm_judge_wins"] = (
            self.scores["lm_judge_wins"] / self.scores["lm_judge_n_samples"]
        )
        self.scores["lm_judge_wins_and_ties"] = (
            self.scores["lm_judge_wins_and_ties"] / self.scores["lm_judge_n_samples"]
        )

        return instances

    def _batch_evaluate_reference_comparison(self, instances, lm_judge_raw, lm_judge_mapping, eval_type="reference_comparison"):
        "Have a judge LM grade how well the model's summary matches reference summary, on a scale of 1 to 5."
        self.scores = {"ratings": []}

        for index, instance in instances.items():
            instance["title"] = self._extract_data_from_prompt(
                instance["prompt"], "title"
            )
            instance["body_text"] = self._extract_data_from_prompt(
                instance["prompt"], "body_text"
            )
            # results = lm_judge(instance, eval_type="reference_comparison")
            # if results:
            #     self.scores["ratings"].append(results)
        results = batch_lm_judge(
            instances.values(), 
            eval_type=eval_type, 
            lm_judge_raw=lm_judge_raw, 
            lm_judge_mapping=lm_judge_mapping
        )
        if results:
            # self.scores["ratings"].append(results)
            self.scores["ratings"] = results
        else:
            print("Job submitted. Come back later! ")
            return None

        # calculate average lm-judge rating
        self.scores["lm_judge_n_samples"] = len(self.scores["ratings"])
        self.scores["avg_rating"] = (
            sum(self.scores["ratings"]) / self.scores["lm_judge_n_samples"]
        )
        del self.scores["ratings"]
        # instances.update(self.scores)

        return instances
    
    def _evaluate_reference_comparison(self, instances):
        "Have a judge LM grade how well the model's summary matches reference summary, on a scale of 1 to 5."
        self.scores = {"ratings": []}

        for index, instance in instances.items():
            instance["title"] = self._extract_data_from_prompt(
                instance["prompt"], "title"
            )
            instance["body_text"] = self._extract_data_from_prompt(
                instance["prompt"], "body_text"
            )
            results = lm_judge(instance, eval_type="reference_comparison")
            if results:
                self.scores["ratings"].append(results)

        # calculate average lm-judge rating
        self.scores["lm_judge_n_samples"] = len(self.scores["ratings"])
        self.scores["avg_rating"] = (
            sum(self.scores["ratings"]) / self.scores["lm_judge_n_samples"]
        )
        del self.scores["ratings"]

        return instances

    def evaluate(
        self,
        instances,
        lm_judge_raw_results_file,
        lm_judge_agg_results_file,
        eval_type,
        n_samples,
        lm_judge_raw,
        lm_judge_mapping,
        use_batch_api=False
    ):

        # sample n_samples examples for evaluation
        random.seed(42)
        n_samples = min(n_samples, len(instances))
        chosen_samples_for_llm = sorted(random.sample(range(len(instances)), n_samples))
        filtered_instances = {
            key: value
            for key, value in instances.items()
            if key in chosen_samples_for_llm
        }

        # run evaluation based on eval_type
        if eval_type == "model_comparison":
            filtered_instances = self._evaluate_model_comparison(filtered_instances)
        elif eval_type == "reference_comparison" or eval_type == "reference_free":
            if use_batch_api:
                filtered_instances = self._batch_evaluate_reference_comparison(filtered_instances, lm_judge_raw, lm_judge_mapping, eval_type)
            else:
                filtered_instances = self._evaluate_reference_comparison(filtered_instances)
        
        if not filtered_instances:
            return None

        # Dump results so we don't need to recompute.
        with open(lm_judge_raw_results_file, ("w")) as f:
            json.dump(filtered_instances, f, indent=2)

        with open(lm_judge_agg_results_file, ("w")) as f:
            json.dump(self.scores, f, indent=2)

        return self.scores
