# Evaluation tasks

This doc has a list of all evaluation tasks, including input / output examples and evaluation metrics.

## Table of contents

- [BioASQ](#bioasq): question answering
- [BioRED](#biored): named entity recognition
- [Discomat](#discomat): table extraction
- [Evidence inference](#evidence-inference): evidence tuple extraction
- [Multicite](#multicite): citation intent classification
- [MUP](#mup): summarization
- [Qasper](#qasper): paper question answering
- [SciERC](#scierc): named entity recognition
- [SciFact](#scifact): claim verification

## BioASQ

- Task input: A collection of biomedical research excerpts and a question answerable from the excerpts.
- Task output: A list of answers to the question.
- Metrics: Compare predicted vs. reference answers using exact-match F1.

Input

```text
Below are a collection of excerpts from biomedical research articles. Excerpts are separated by
newlines. Your task is to answer a question based these excerpts. Your response should be formatted
as a `json` array.

For instance, given excerpts from articles studying breast cancer, and the question "what are some
common genes associated with breast cancer?", an answer might be formatted like: ["BRCA1", "BRCA2",
"TP53", ...]. Only include answers that are mentioned in the provided exerpts. The array should
always have at least one answer; do not respond with an empty array []. Do not include any text in
your response other than the answer array.

Context: sensitization, behavioral changes, and low body mass index (BMI). One possible cellular
target that may mediate some of these findings is the hypocretin/orexin neurons. This neuronal
system plays a role in regulating wakefulness/sleep cycles, pain perception, and appetite. Food
intake, in contrast, receives circadian modulation through hormones such as leptin, ghrelin, insulin
and orexin. A low level of hypocretin-1/orexin-A in the cerebrospinal fluid is sufficient to
diagnose narcolepsy type 1, being a highly specific and sensitive biomarker, and the irreversible
loss of hypocretin neurons is responsible for the main symptoms of the disease: Orexins, or
hypocretins, are excitatory neuropeptides involved in the regulation of feeding behavior and the
sleep and wakefulness states.

[Lines omitted for space]

Orexin A (OXA) and orexin B (OXB) are recently discovered neuropeptides that appear to play a role
in various distinct functions such as arousal and the sleep-wake cycle as well as on appetite and
regulation of feeding and energy homeostasis. Orexins were first described as neuropeptides
expressed by a sp Orexin/hypocretin neurons located in the lateral hypothalamus play a critical role
in the maintenance of arousal and contribute to the regulation of multiple homeostatic and
behavioral processes.

Question: What processes do orexin/hypocretin neurons regulate?
```

Output

```json
[
  "sleep",
  "appetite",
  "wakefullness",
  "pain",
  "reward",
  "energy homeostasis",
  "goal-directed behaviors",
  "Arousal",
  "addiction"
]
```

## BioRed

- Task input: Abstract of a biomedical research article.
- Task output: All entities in the article of the following types:
  - cell line
  - chemical
  - disease
  - gene
  - gene variant
  - species
- Metrics: Compare predicted vs. reference entities using exact-match F1.

Input

```text
You will be shown an abstract from a biomedical research paper. Given this abstract, your task is to
extract all unique entities of the following types: ["Chemical", "Variant", "Gene", "CellLine",
"Disease", "Species"].

Please return the output as a JSON object of the format: {"CellLine": ["hRPTEC", ...], "Chemical":
["Glucose", ...], "Disease": ["Diabetes", ...], "Gene": ["HNF-6", ...], "Species": ["Patients",
...], "Variant": ["Pro75Ala", ...]}. The keys should be entity types and values should be lists of
extracted entities belonging to the corresponding type. If you cannot find entities belonging to a
specific type, the value should be [].

Only output the JSON object and do not include any additional text.

Abstract:

Fatal carbamazepine induced fulminant eosinophilic (hypersensitivity) myocarditis: emphasis on
anatomical and histological characteristics, mechanisms and genetics of drug hypersensitivity and
differential diagnosis. The most severe adverse reactions to carbamazepine have been observed in the
haemopoietic system, the liver and the cardiovascular system. A frequently fatal, although
exceptionally rare side effect of carbamazepine is necrotizing eosinophilic (hypersensitivity)
myocarditis. We report a case of hypersensitivity myocarditis secondary to administration of
carbamazepine. Acute hypersensitivity myocarditis was not suspected clinically, and the diagnosis
was made post-mortem. Histology revealed diffuse infiltration of the myocardium by eosinophils and
lymphocytes with myocyte damage. Clinically, death was due to cardiogenic shock. To best of our
knowledge this is the second case of fatal carbamazepine induced myocarditis reported in English
literature.
```

Output

```json
{
  "CellLine": [],
  "Chemical": [
    "carbamazepine"
  ],
  "Disease": [
    "hypersensitivity",
    "death",
    "myocarditis",
    "cardiogenic shock",
    "drug hypersensitivity"
  ],
  "Gene": [],
  "Species": [],
  "Variant": []
}
```

## Discomat

- Task input: A passage from a research paper including a table.
- Task output: The table, with each cell as a `json` line.
- Metrics: BLEU score between predicted and gold reference. Manual inspection showed that BLEU was pretty reliable for this task.

Input

```text
| Sample no. | Ph, volume percent of crystals | Activation energy (kJ/mol) of the scale factor for normalised frequency | ln(t 0, s) of the scale factor | G  unrelaxed shear modulus (GPa) |
| Glas 0     | 0                              | 137+-18                                                                 | -50.15                         | 24.3                             |
| Glas 1     | 17                             | 129+-13                                                                 | -47.68                         | 23.8                             |
| Glas 3     | 22                             | 126+-16                                                                 | -45.72                         | 24.7                             |
| Glas 5     | 27                             | 117+-10                                                                 | -42.25                         | 25.0                             |
Caption: Activation energies of shear stress relaxation and unrelaxed shear modulus of disilicate
lithium glasses

You are provided with a table from a material science paper. Here are JSON templates for two types
of numeric cells: "Other" and "Glass_Compound_Amount": {"value": "xx", "type": "Other"} {"value":
"xx", "type": "Glass_Compound_Amount", "constituent": "xx", "unit": "xx", "material": "xx"}

Please describe all numeric cells in the above table following the JSON templates (proceeding by row
in a left-right, top-down direction). For each cell, output one JSON description per line. For any
unanswerable attributes in the templates, set their value to the placeholder "xx".

Cell Description:
```

Output

```json
{"value": "0", "type": "Other"}
{"value": "137", "type": "Other"}
{"value": "24.3", "type": "Other"}
{"value": "17", "type": "Other"}
{"value": "129", "type": "Other"}
{"value": "23.8", "type": "Other"}
{"value": "22", "type": "Other"}
{"value": "126", "type": "Other"}
{"value": "24.7", "type": "Other"}
{"value": "27", "type": "Other"}
{"value": "117", "type": "Other"}
{"value": "25.0", "type": "Other"}
```

## Evidence Inference

- Task input: Abstract of a clinical trial report.
- Task output: List of all ICO `(intervention / comparator / outcome)` tuples, together with the effect of the intervention on the outcome and the textual evidence of this effect.
- Metrics: "Fuzzy" F1. Given a prediction and a reference tuple, compute the token overlap for each tuple item. If token overlaps for all fields exceed 0.3, the predicted tuple is judged as a match to the reference.

Input

```text
You will be shown the abstract of a medical clinical trial report. Your task is to extract all the
findings from this report into a JSON array. Each finding should contain the following five
elements:

- Intervention: The medical intervention being tested. This should be a text span copied from the
  input passage.
- Comparator: The baseline against which the intervention is being evaluated. This should be a text
  span copied from the input passage. If no comparator is reported, set to `null`.
- Outcome: The medical outcome whose effect is being measured. This should be a text span copied
  from the input passage.
- Effect: The effect of the intervention on the outcome, relative to the comparator. The effect
  should be one of the following three values: ("significantly increased", "significantly
  decreased", "no significant difference").
- Evidence: The evidence for the effect. This should be a text span copied from the input passage.

Please format your output as a JSON array. Each entry in the output should be an array containing
the 5 elements listed above, in the following order: [<intervention>, <comparator>, <outcome>,
<effect>, <evidence>].

For example, an output with two findings might read: [["aspirin", "placebo", "headache severity",
"significantly decreased", "Mean headache severity was significantly decreased in the aspirin group
compared to the placebo group (p < 0.05)."], ["aspirin", "placebo", "weight loss", "no significant
difference", "We did not observe any difference in weight loss between the group given aspirin
relative to the control group"]]

There are 3 finding(s) in the abstract below. Please extract them. Output only the JSON array with
these 3 findings. Do not include any additional text.

Abstract: ABSTRACT.OBJECTIVES: To compare the efficacy and safety of SB4 (an etanercept biosimilar)
with reference product etanercept (ETN) in patients with moderate to severe rheumatoid arthritis
(RA) despite methotrexate (MTX) therapy.

ABSTRACT.METHODS: This is a phase III, randomised, double-blind, parallel-group, multicentre study
with a 24-week primary endpoint. Patients with moderate to severe RA despite MTX treatment were
randomised to receive weekly dose of 50 mg of subcutaneous SB4 or ETN. The primary endpoint was the
American College of Rheumatology 20% (ACR20) response at week 24. Other efficacy endpoints as well
as safety, immunogenicity and pharmacokinetic parameters were also measured.

ABSTRACT.RESULTS: 596 patients were randomised to either SB4 (N=299) or ETN (N=297). The ACR20
response rate at week 24 in the per-protocol set was 78.1% for SB4 and 80.3% for ETN. The 95% CI of
the adjusted treatment difference was -9.41% to 4.98%, which is completely contained within the
predefined equivalence margin of -15% to 15%, indicating therapeutic equivalence between SB4 and
ETN. Other efficacy endpoints and pharmacokinetic endpoints were comparable. The incidence of
treatment-emergent adverse events was comparable (55.2% vs 58.2%), and the incidence of antidrug
antibody development up to week 24 was lower in SB4 compared with ETN (0.7% vs 13.1%).

ABSTRACT.CONCLUSIONS: SB4 was shown to be equivalent with ETN in terms of efficacy at week 24. SB4
was well tolerated with a lower immunogenicity profile. The safety profile of SB4 was comparable
with that of ETN.

ABSTRACT.TRIAL REGISTRATION NUMBERS: NCT01895309, EudraCT 2012-005026-30.

Findings:
```

Output

```json
[
  [
    "SB4 (an etanercept biosimilar)",
    "etanercept (ETN)",
    "therapeutic equivalence - Patients with moderate to severe rheumatoid arthritis (RA) despite methotrexate (MTX) treatment",
    "no significant difference",
    "The 95% CI of the adjusted treatment difference was \u22129.41% to 4.98%, which is completely contained within the predefined equivalence margin of \u221215% to 15%, indicating therapeutic equivalence between SB4 and ETN."
  ],
  [
    "SB4 (an etanercept biosimilar)",
    "etanercept (ETN)",
    "incidence of antidrug antibody development up to week 24 - Patients with moderate to severe rheumatoid arthritis (RA) despite methotrexate (MTX) treatment",
    "significantly decreased",
    "the incidence of antidrug antibody development up to week 24 was lower in SB4 compared with ETN (0.7% vs 13.1%)."
  ],
  [
    "SB4 (an etanercept biosimilar)",
    "etanercept (ETN)",
    "incidence of treatment-emergent adverse events - Patients with moderate to severe rheumatoid arthritis (RA) despite methotrexate (MTX) treatment",
    "no significant difference",
    "The incidence of treatment-emergent adverse events was comparable (55.2% vs 58.2%)"
  ]
]
```

## Multicite

- Task Input: A citation sentence from a research paper.
- Task output: A list of intents for the citation sentence.
- Metrics: Compare predicted vs. reference intents using exact-match F1.

Input

```text
Your task is to classify the citation intent within the following provided text from a computational
linguistics research paper. The cited work is demarcated by "<cite>" and "</cite>". Determine the
purpose of the cited work by selecting from the listed categories:

- Background: The cited paper underpins the subject matter.
- Motivation: The cited paper inspires or provides a rationale for the current research.
- Uses: The current work utilizes concepts or tools from the cited paper.
- Extends: The current work advances ideas or methods from the cited paper.
- Similarities: The current work identifies commonalities with the cited paper.
- Differences: The current work delineates its distinction from the cited paper.
- FutureWork: The cited paper is acknowledged as groundwork for prospective research.

Indicate the intents by listing them in a `json` array, e.g. ["Background", "Uses"]. More than one
intent may be applicable. Do not include any extraneous text in your response.

Context with Citation: In addition to that, we implemented semi-supervised classification by
training in the positive samples of the <cite>[9]</cite> dataset and training in only the lexicon as
negative samples.
```

Output

```json
[
  "Similarities",
  "Uses"
]
```

## MUP

- Task input: Full text of a machine learning paper.
- Task output: Short paper summary that a reviewer might write as part of a paper review.
- Metrics: Use GPT-3.5 to judge similarity of generated summary to human
  reference on 1-5 scale. Based on manual inspection, this was higher-quality
  than automated metrics like ROUGE.

Input

```text
You will be presented with the title and body text of a computer science research paper. Please
write a summary of the work that would be informative for a peer reviewer assessing its quality.
Your summary should be 3 sentences long. In your response, include only the summary and no
additional text.

Paper title: Reinforcement Learning with Efficient Active Feature Acquisition

Paper body: 1 INTRODUCTION . Recently , machine learning models for automated sequential decision
making have shown remarkable success across many application areas , such as visual recognition (
Mathe et al. , 2016 ; Das et al. , 2017 ) , robotics control ( Finn et al. , 2016 ; Zhang et al. ,
2018 ) , medical diagnosis ( Ling et al. , 2017 ; Peng et al. , 2018 ) and computer games ( Mnih et
al. , 2015 ; Silver et al. , 2016 ) . One fundamental reason that drives the success of such models
and enables them to outperform classical algorithms is the availability of large amounts of training
data . Typically such training data is either fully observed or the features stem from an
action-independent observation model ( which clearly can depend on the state of the system ) .
However , the fundamental assumption that the same features are always readily available during
deployment could not hold in many real-world applications . For instance , consider a medical
support system for monitoring and treating patients during their stay at hospital which was trained
on rich historical medical data . To provide the best possible treatment , the system might need to
perform several measurements of the patient over time , while some of them could be costly or even
pose a health risk . Therefore , during deployment , it is more ideal that the system could function
with minimal features while during training more features might have been available . In such cases
, we are interested in decision making models that actively take the measurement process , i.e. ,
feature acquisition , into account and only acquire the information relevant for making a decision .
In this paper , we consider the challenging problem of learning effective policies when the cost of
information acquisition can not be neglected . To be successful , we need to learn policies which
acquires the information required for solving a task in the cheapest way possible . [Truncated for
space].

3-sentence paper summary:
```

Output

```text
In this paper the authors propose an approach for simultaneously learning how to explore more efficiently in POMDPs via targeted feature acquisition, and learning a reward-maximizing control policy, balancing the cost of feature acquisition with the expected reward. Learning is done via a VAE framework which combines a belief inference model and an observation decoder, with a key innovation being that inference is done as a sequential process. Results comparing this approach to other variational inference approaches show the proposed framework reaches better performance with lower cost (particularly, number of acquired features).
```

### Qasper

- Task input: The full text of an NLP research paper, and a question answerable from the paper body (but not the abstract).
- Task output: An answer to the question, accompanied by the extracts from the paper body supplying the answer.
- Metrics: We compute metrics for both the answer and the evidence.
  - Answer: GPT-3.5 judge of similarity of model answer to human reference (1-5 scale).
  - Evidence: Token F1 overlap with gold evidence.

Input

```text
You will be shown sections from a scientific research paper, together with a question about the
paper. Paragraphs in the paper are separated by newlines. Your task is to answer the question based
on the contents of the paper.

Paper:
----------------------------------------
Named Entity Disambiguation for Noisy Text

We address the task of Named Entity Disambiguation (NED) for noisy text. We present WikilinksNED, a
large-scale NED dataset of text fragments from the web, which is significantly noisier and more
challenging than existing news-based datasets. To capture the limited and noisy local context
surrounding each mention, we design a neural model and train it with a novel method for sampling
informative negative examples. We also describe a new way of initializing word and entity embeddings
that significantly improves performance. Our model significantly outperforms existing
state-of-the-art methods on WikilinksNED while achieving comparable performance on a smaller
newswire dataset.

The WikilinksNED Dataset:             Entity Mentions in the Web We introduce WikilinksNED, a
large-scale NED dataset based on text fragments from the web. Our dataset is derived from the
Wikilinks corpus BIBREF14 , which was constructed by crawling the web and collecting hyperlinks
(mentions) linking to Wikipedia concepts (entities) and their surrounding text (context). Wikilinks
contains 40 million mentions covering 3 million entities, collected from over 10 million web pages.
Wikilinks can be seen as a large-scale, naturally-occurring, crowd-sourced dataset where thousands
of human annotators provide ground truths for mentions of interest. This means that the dataset
contains various kinds of noise, especially due to incoherent contexts. The contextual noise
presents an interesting test-case that supplements existing datasets that are sourced from mostly
coherent and well-formed text.

[Truncated for space]
----------------------------------------

Question: How was a quality control performed so that the text is noisy but the annotations are
accurate?

To answer the question, format your response as a `json` object with two fields:

"answer": A string providing a succinct answer to the question, in your own words. "evidence": An
array of strings. Each entry should be a full paragraph from the paper. Together, the evidence
should serve as a justification for the answer.

For instance, for the question "What baselines did the authors compare against?", a sample response
might be:

{ "answer": "BERT and RoBERTa." "evidence": ["We compare our approach against two baselines. In
  Table 1, we compare against BERT. In Table 2, we compare against RoBERTa. Our findings indicate
  that our approach improves over both baeslines..."] }

The "answer" field should be roughly 190 characters in length.

Do not include any text in your response other than the json. If the question is unanswerable given
the provided excerpts, respond with the single word "null".

To repeat, the question is: How was a quality control performed so that the text is noisy but the
annotations are accurate?

Answer JSON object:
```

Output

```json
{
  "answer": "Profile pictures from the Twitter users' profiles.",
  "evidence": [
    "The recent advancements in deep neural networks, specifically for image analysis task, can lead to determining demographic features such as age and gender BIBREF13 . We show that by determining and integrating heterogeneous set of features from different modalities \u2013 aesthetic features from posted images (colorfulness, hue variance, sharpness, brightness, blurriness, naturalness), choice of profile picture (for gender, age, and facial expression), the screen name, the language features from both textual content and profile's description (n-gram, emotion, sentiment), and finally sociability from ego-network, and user engagement \u2013 we can reliably detect likely depressed individuals in a data set of 8,770 human-annotated Twitter users."
  ]
}
```

## SciERC

- Task input: An abstract of an NLP paper.
- Task output: A list of all entities mentioned in the paper of the following types:
  - Material
  - Method
  - Metric
  - Task
  - Generic
  - Other scientific term
- Metrics: Exact-match F1.

Input

```text
You will be shown an abstract from a computer science research paper. Given this abstract, your task
is to extract all unique entities with the following types:

- "Task": Applications, problems to solve, systems to construct. Examples include "information
  extraction", "machine reading system", "image segmentation".
- "Method": : Methods, models, systems to use, or tools, components of a system, frameworks.
  Examples include "language model", "CORENLP", "POS parser".
- "Metric": Metrics, measures, or entities that can express quality of a system / method. Examples
  include "F1", "BLEU", "Precision", "time complexity".
- "Material": Data, datasets, resources, Corpus, Knowledge base. Examples include "image data",
  "speech data", "stereo images", "CoNLL", "Wikipedia".
- "OtherScientificTerm": Phrases that are a scientific terms but do not fall into any of the above
  classes. Examples include "physical or geometric constraints", "qualitative prior knowledge",
  "tree kernel", "noise".
- "Generic": General terms or pronouns that may refer to a entity but are not themselves
  informative, often used as connection words. Examples include "model", "approach", "them".

Please return the output as a JSON object of the format: {"type1" : ["example_entity", ...], "type2"
: ["example_entity", ...]}. The keys should be entity types and values should be lists of extracted
entities belonging to the corresponding type. Entity types with no matching entities should be
assigned an empty array [].

For instance, the output might look like: {"Task": ["speech recognition", ...], "Method":
["Conditional random field"], "Material": [], ...}.

Only output the JSON object and do not include any additional text.

Abstract:

We present a syntax-based constraint for word alignment, known as the cohesion constraint. It
requires disjoint English phrases to be mapped to non-overlapping intervals in the French sentence.
We evaluate the utility of this constraint in two different algorithms. The results show that it can
provide a significant improvement in alignment quality.
```

Output

```json
{
  "Generic": [
    "algorithms"
  ],
  "Material": [
    "English phrases",
    "French sentence"
  ],
  "Method": [],
  "Metric": [
    "alignment quality"
  ],
  "OtherScientificTerm": [
    "cohesion constraint",
    "syntax-based constraint"
  ],
  "Task": [
    "word alignment"
  ]
}
```

## SciFact

- Task input: An abstract from a biomedical research article, and a scientific claim.
- Task output:
  - A fact-checking verdict indicating whether the abstract supports or refutes the claim, or has no relevant information.
  - The evidence -- i.e. sentences from the abstract justifying the verdict.
- Metrics: We compute metrics for both the answer and the evidence.
  - Verdict: Label F1.
  - Evidence: Token F1 overlap with gold evidence.

Input

```text
You will be shown a scientific claim, and the abstract of a biomedical research paper. Each sentence
from the abstract will be on a separate line. Your task is to return a JSON object with two fields:

- "verdict": The fact-checking verdict. If the information in the abstract supports the claim, write
  "SUPPORT". If the abstract contradicts the claim, write "CONTRADICT". If the abstract does not
  provide enough information to arrive at a verdict, write "NEI" (for "not enough information").
- "evidence": An array of sentences providing evidence for the verdict. Please copy all relevant
  sentences verbatim from the abstract. If the verdict was "NEI", then return an empty array.

For instance, if the model were given the claim "smoking causes cancer", the output might be {
"verdict": "SUPPORT", "evidence": ["The results of our meta-analysis provide overwhelming support
  that cigarette smoking is a risk cause for lung cancer."] }

Your response should not include any text other than the json.

Claim: Therapeutics receiving accelerated approval encounter a lower frequency of post-marketing
safety events

Abstract: Importance Postmarket safety events of novel pharmaceuticals and biologics occur when new
safety risks are identified after initial regulatory approval of these therapeutics. These safety
events can change how novel therapeutics are used in clinical practice and inform patient and
clinician decision making. Objectives To characterize the frequency of postmarket safety events
among novel therapeutics approved by the US Food and Drug Administration (FDA), and to examine
whether any novel therapeutic characteristics known at the time of FDA approval were associated with
increased risk. [Truncated for space] Biologics, psychiatric therapeutics, and accelerated and
near–regulatory deadline approval were statistically significantly associated with higher rates of
events, highlighting the need for continuous monitoring of the safety of novel therapeutics
throughout their life cycle.
```

Output

```json
{
  "verdict": "CONTRADICT",
  "evidence": [
    "In multivariable analysis, postmarket safety events were statistically significantly more frequent among biologics (incidence rate ratio [IRR] = 1.93; 95% CI, 1.06-3.52; P = .03), therapeutics indicated for the treatment of psychiatric disease (IRR = 3.78; 95% CI, 1.77-8.06; P < .001), those receiving accelerated approval (IRR = 2.20; 95% CI, 1.15-4.21; P = .02), and those with near\u2013regulatory deadline approval (IRR = 1.90; 95% CI, 1.19-3.05; P = .008); events were statistically significantly less frequent among those with regulatory review times less than 200 days (IRR = 0.46; 95% CI, 0.24-0.87; P = .02)."
  ]
}
```
