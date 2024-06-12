import json
import re
from openai import OpenAI
import openai
import random

# Call judge LLM to compare model answer to reference.
CLIENT = OpenAI()


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
        elif eval_type == "reference_comparison" and decision in [
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


def lm_judge(instance, eval_type):
    lm_judge_model = (
        "gpt-4-turbo-preview" if eval_type == "model_comparison" else "gpt-3.5-turbo"
    )
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
            results = lm_judge(instance, eval_type="model_comparison")
            if results:
                self.scores["lm_judge_n_samples"] += 1
                self.scores["lm_judge_wins"] += results[0]
                self.scores["lm_judge_wins_and_ties"] += results[1]

        # calculate win ratio & wins_and_ties ratio
        self.scores["lm_judge_wins"] = (
            self.scores["lm_judge_wins"] / self.scores["lm_judge_n_samples"]
        )
        self.scores["lm_judge_wins_and_ties"] = (
            self.scores["lm_judge_wins_and_ties"] / self.scores["lm_judge_n_samples"]
        )

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
        elif eval_type == "reference_comparison":
            filtered_instances = self._evaluate_reference_comparison(filtered_instances)

        # Dump results so we don't need to recompute.
        with open(lm_judge_raw_results_file, ("w")) as f:
            json.dump(filtered_instances, f, indent=2)

        with open(lm_judge_agg_results_file, ("w")) as f:
            json.dump(self.scores, f, indent=2)

        return self.scores
