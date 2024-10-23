import numpy as np
import evaluate
import json
import re
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
import time, os

from sciriff.eval.metrics import util


########################################

# Call judge LLM to compare model answer to reference.

CLIENT = OpenAI()
# CLIENT = AzureOpenAI(
#   azure_endpoint = "https://default-yalenlp.openai.azure.com/", 
#   api_key="0928d74bbed94cd18922f7bc6bf45b4c",  
#   api_version="2024-07-01-preview"
# )


def create_batch_file(instances):
    tasks = []
    for idx, instance in enumerate(instances):
        prompt = make_prompt(instance)
        task = {
            "custom_id": f"task-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                # "model": "gpt-4o-mini-2024-07-18",
                "model": "gpt-4o-2024-08-06",
                # "model": "gpt-3.5-turbo-0125",
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
    os.makedirs("./attributed_qa_eval-logs", exist_ok=True)
    file_name = f"./attributed_qa_eval-logs/{timestr}-batch_tasks.jsonl"
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

def get_batch_results(batch_job_id):
    while True:
        batch_job = CLIENT.batches.retrieve(batch_job_id)
        if batch_job.status == "completed":
            break
        time.sleep(60)  # Wait for 1 minute before checking again

    result_file_id = batch_job.output_file_id
    result = CLIENT.files.content(result_file_id).content
    
    results = []
    for line in result.decode().split('\n'):
        if line:
            results.append(json.loads(line))
    
    return results

def call_lm_judge(prompt):
    chat_completion = CLIENT.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="gpt-4o-2024-08-06",
    )

    return chat_completion.choices[0].message.content


def make_prompt(instance):
    lines = instance["prompt"].split("\n")
    title = lines[5]
    question = [line for line in lines if re.match("^Question:", line)]
    if len(question) != 1:
        raise ValueError("Couldn't find the question.")
    question = question[0].replace("Question: ", "")
    excerpts = "\n".join(instance["ref"]["evidence"])

    prompt = f"""\
    Below you will be shown a paper title, an excerpt from the paper, and a question
    about the excerpt. Then, you will be given a reference answer written by an expert,
    followed by a model-generated answer. Please rate the similarity of the model answer
    to the reference on a 1-5 scale. Do not penalize the model for including additional
    information that is not in the reference.

    - 5: The model answer includes all important information found in the reference.
    - 3: The model answer is somewhat similar to the reference, but not completely accurate.
    - 1: The model answer is totally inaccurate or is unrelated to the reference.
    In your response, give an explanation for your rating, followed by your rating.

    Response format:
    Explanation: Your explanation here.
    Rating: A single integer between 1 and 5.

    _Do not_ include any additional text after the rating.

    Here's the article, question, and answers:

    Title: {title}

    Excerpts:
    {excerpts}

    Question: {question}

    Reference answer: {instance['ref']['answer']}

    Model answer: {instance['pred']['answer'] if 'answer' in instance['pred'] else None}
    """
    prompt = prompt.replace("    ", "")

    return prompt


def extract_rating(response):
    pattern = r"Rating:\s*(\d+)"
    match = re.search(pattern, response)
    if match:
        score = int(match.group(1))
        if score in [1, 2, 3, 4, 5]:
            return score
        else:
            print("Invalid score from LM judge.")
            return 3
    else:
        print("Failed to extract LM judge rating")
        return 3


def batch_lm_judge(instances, lm_judge_raw=None):
    if not lm_judge_raw.exists():
        file_name = create_batch_file(instances)
        batch_job_id = submit_batch_job(file_name)
        # results = get_batch_results(batch_job_id)
        return None
    else:
        results = json.load(open(lm_judge_raw))
    
    ratings = []
    for result in results:
        # response = result['response']['choices'][0]['message']['content']
        response = result['response']['body']['choices'][0]['message']['content']
        rating = extract_rating(response)
        ratings.append(rating)
    
    return ratings

def lm_judge(instance):
    "Have a judge LM grade how well the answer matches reference, on a scale of 1 to 5."
    # Get rid of leading space.
    prompt = make_prompt(instance)
    response = call_lm_judge(prompt)
    rating = extract_rating(response)

    return rating


########################################

# Evaluator.

class AttributedQAEval:
    "Attributed question answer. Answer the question and provide evidence."

    def _normalize_answer(self, answer):
        "If the model gave a list as the answer instead of a string, convert it."
        if isinstance(answer, str):
            self.failure_counts["answer_entry_is_str"]["yes"] += 1
            return answer
        elif isinstance(answer, list):
            self.failure_counts["answer_entry_is_str"]["no"] += 1
            return " ".join([str(x) for x in util.flatten(answer)])
        elif isinstance(answer, dict):
            self.failure_counts["answer_entry_is_str"]["no"] += 1
            return str(answer)
        else:
            raise Exception("Unexpected answer type")

    def _evaluate_one(self, instance, use_batch_api=False):
        # Compute answer token F1. Record cases where the answer failed and give no
        # credit.

        # In some cases the model doesn't return a dict; just mark these as None.
        if isinstance(instance["pred"], dict):
            answer_pred = instance["pred"].get("answer", None)
            evs_pred = instance["pred"].get("evidence", None)
        else:
            answer_pred = None
            evs_pred = None

        # Answer accuracy.
        answer_ref = instance["ref"]["answer"]
        self.answers["refs_all"].append(answer_ref)
        if answer_pred is None:
            self.failure_counts["answer"]["no"] += 1
            self.scores["f1_answer_all"].append(0)
            self.answers["preds_all"].append("")
            if self.do_lm_judge:
                self.scores["lm_judge"].append(1)
        else:
            self.failure_counts["answer"]["yes"] += 1
            answer_pred = self._normalize_answer(answer_pred)
            f1_answer = util.compute_token_f1(answer_pred, answer_ref)
            self.scores["f1_answer_all"].append(f1_answer)
            self.scores["f1_answer_parsed"].append(f1_answer)

            if self.do_lm_judge and not use_batch_api:
                lm_judge_score = lm_judge(instance)
                self.scores["lm_judge"].append(lm_judge_score)

            self.answers["refs_parsed"].append(answer_ref)
            for the_key in ["preds_all", "preds_parsed"]:
                self.answers[the_key].append(answer_pred)

        # Same for the evidence
        if evs_pred is None:
            self.failure_counts["evidence"]["no"] += 1
            self.scores["f1_evidence_all"].append(0)
        else:
            # Flatten in case the model returned the wrong list structure.
            self.failure_counts["evidence"]["yes"] += 1
            ev_pred = " ".join(util.flatten(evs_pred))
            ev_ref = " ".join(instance["ref"]["evidence"])
            f1_evidence = util.compute_token_f1(ev_pred, ev_ref)
            self.scores["f1_evidence_all"].append(f1_evidence)
            self.scores["f1_evidence_parsed"].append(f1_evidence)

    def _get_rouge_scores(self):
        scorer = evaluate.load("rouge")
        res = {}
        for version in ["parsed", "all"]:
            scores_loop = scorer.compute(
                predictions=self.answers[f"preds_{version}"],
                references=self.answers[f"refs_{version}"],
            )
            for k, v in scores_loop.items():
                res[f"{k}_{version}"] = v

        return res

    def evaluate(self, instances, lm_judge_file=None, lm_judge_raw_file=None, use_batch_api=False):
        self.lm_judge_file = lm_judge_file
        self.do_lm_judge = False
        if self.lm_judge_file is not None:
            if self.lm_judge_file.exists():
                lm_judge_scores = json.load(open(self.lm_judge_file))
            else:
                self.do_lm_judge = True
                lm_judge_scores = []
        else:
            lm_judge_scores = None

        self.scores = {
            "f1_answer_parsed": [],
            "f1_answer_all": [],
            "f1_evidence_parsed": [],
            "f1_evidence_all": [],
            "lm_judge": lm_judge_scores,
        }
        self.failure_counts = {
            "answer": util.count_dict(["yes", "no"]),
            "evidence": util.count_dict(["yes", "no"]),
            "answer_entry_is_str": util.count_dict(["yes", "no"]),
        }
        self.answers = {
            "preds_parsed": [],
            "refs_parsed": [],
            "preds_all": [],
            "refs_all": [],
        }

        # Get token F1 scores for answer and evidence.
        for instance in instances:
            self._evaluate_one(instance, use_batch_api)

        # Dump LM judge scores so we don't need to recompute.
        if self.do_lm_judge:
            if use_batch_api:
                self.scores["lm_judge"] = batch_lm_judge(instances, lm_judge_raw=lm_judge_raw_file)
                if not self.scores["lm_judge"]:
                    return
            with open(self.lm_judge_file, ("w")) as f:
                json.dump(self.scores["lm_judge"], f, indent=2)

        scores = {k: np.mean(v) for k, v in self.scores.items() if v is not None}
        # Rescale LM judge scores from 0 to 1
        if "lm_judge" in scores:
            scores["lm_judge"] = (scores["lm_judge"] - 1) / 4

        # Get rouge scores.
        rouge_scores = self._get_rouge_scores()
        scores.update(rouge_scores)

        for k in self.failure_counts:
            self.failure_counts[k]["frac_success"] = util.safe_div(
                self.failure_counts[k]["yes"], util.sum_dict(self.failure_counts[k])
            )
        res = {"scores": scores, "answer_parse": self.failure_counts}

        return res
