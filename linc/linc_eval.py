# This file runs LINC's evaluation method on FOLIO. Run this using:
# python linc_eval.py --model_name "llama" --tasks "folio" --output_dir "output" --precision "fp16"

import argparse
import json
import openai
import hashlib
import time
import random
import os
import torch
import math
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod, ABC
from warnings import warn
from diskcache import Cache
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset
from transformers import StoppingCriteria, StoppingCriteriaList
from accelerate.utils import set_seed
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
from functools import lru_cache
from collections import Counter
from abc import abstractmethod, ABC
from datasets import load_dataset
from warnings import warn


import folio
import proofwriter


INFILL_MODE = False

class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except:
            warn(
                "This task will use a locally downloaded dataset, not from the HF hub."
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass


class OWAFOLTask(Task):
    """An OWA (Open World Assumption) FOL (First Order Logic) Task is a Task in which the goal
    is to generate True/False/Uncertain answers to First Order Logic questions.
    """

    TRAIN_DATASET_PATH = "minimario/FOLIO"
    ERROR_TOKEN = "Error"
    MAX_SHOTS = 16

    def __init__(self, mode, n):
        assert n <= self.MAX_SHOTS, f"supports up to {self.MAX_SHOTS}-shot"
        super().__init__(
            stop_words=["</EVALUATE>"], requires_execution=True,
        )
        self._mode = mode
        self._nshot = n
        self.train_dataset = load_dataset(self.TRAIN_DATASET_PATH)["train"]
        self._train_dataset = self.reformat_fol_samples_train(self.train_dataset)
        self._train_dataset = self.add_conclusion_fols_train(self._train_dataset)
        self._train_dataset = self.add_cot_train(self._train_dataset)
        self._train_dataset = self._train_dataset.map(
            lambda x: {"label": "Uncertain" if x["label"] == "Unknown" else x["label"]},
            remove_columns=["label"],
        )
        self._train_fewshot_indices_all = [
            125,
            23,
            60,
            275,
            148,
            261,
            263,
            683,
            299,
            684,
            850,
            853,
            886,
            892,
            930,
            980,
        ]
        # Labels:
        # 23 (True), 60 (False), 125 (Uncertain), 148 (False), 261 (True), 263 (True), 275 (Uncertain), 683 (Uncertain)
        # 299 (True), 684 (False), 850 (False), 853 (Uncertain), 886 (True), 892 (Uncertain), 930 (False), 980 (False)

        self._train_fewshot_indices = self._train_fewshot_indices_all[:n]
        self._train = self._train_dataset.select(self._train_fewshot_indices)

    def reformat_fol_samples_train(self, train_dataset):
        def reformat_fol_sample(sample):
            sample["premises-FOL"] = [
                convert_to_nltk_rep(premise) for premise in sample["premises-FOL"]
            ]
            return sample

        return train_dataset.map(reformat_fol_sample)

    def add_conclusion_fols_train(self, train_dataset):
        train_conclusion_fols = {
            23: "HigherRank(RealMadrid, Barcelona)",
            60: "-OlympicGoldMedalWinner(Amy) -> NobelLaureate(Amy)",
            125: "-Dispensable(Worksheet)",
            148: "FolkSong(Inception)",
            261: "MakeGoodBreakfast(Luke)",
            263: "exists x. (Develops(Ets, x) & For(x, k-OneTwoandhighereducation)) & exists x. (Develops(Ets, x) & AssociatedWith(x, Entrytouseducationinstitutions))",
            275: "ContributeToCountry(James)",
            299: "GetRhythmRight(John)",
            683: "exists x. (BRICS(x) & Speaks(x, Hindi))",
            684: "Film(Hamilton)",
            850: "-Liked(Leo, Charlie) & -Cares(Charlie, Leo)",
            853: "Won(Threebodyproblem, Hugoaward)",
            886: "Dagfinn(DagfinnAarskog)",
            892: "PartOf(Minsk, Scottishpremiership)",
            930: "-Locate(Boves, Europe)",
            980: "(InvitedTakePhoto(James) & -HappyCommunicate(James)) | (-InvitedTakePhoto(James) & HappyCommunicate(James))",
        }
        conclusions = [None for _ in range(len(train_dataset))]
        for index, conclusion_fol in train_conclusion_fols.items():
            conclusions[index] = conclusion_fol
        train_dataset = train_dataset.add_column("conclusion-FOL", conclusions)
        return train_dataset

    def add_cot_train(self, train_dataset):
        train_cots = {
            23: "Let's think step by step. We want to evaluate if in La Liga 2021-2022, Real Madrid ranks higher than Barcelona. From premise 1, we know that a La Liga soccer team ranks higher than another if it receives more points. From premise 4, we know that in La Liga 2021-2022, Real Madrid received more points than Barcelona. Therefore, in La Liga 2021-2022, Real Madrid received more points than Barcelona, so Real Madrid ranks higher than Barcelona, so the statement is true.\nANSWER:\tTrue",
            60: "Let's think step by step. We want to evaluate the statement \"if Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate\". Let's assume that Amy is not an Olympic gold medal winner. This doesn't tell us anything about whether Amy is a Nobel laureate, so the statement isn't true, meaning it is either False or Uncertain. To distinguish between the two, notice that we could have a scenario where Amy is neither an Olympic gold medal winner nor a Nobel laureate. None of the premises are violated in this case. This means the statement must be false.\nANSWER:\tFalse",
            125: "Let's think step by step. We want to evaluate if a worksheet is not dispensable. From premise 6, we know that a worksheet is either paper or is environment-friendly. If it is paper, then from premise 3, a worksheet is woodware, and from premise 2, a worksheet is dispensable. If it is environment-friendly, we know it is good from premise 5, but we know nothing about whether it is dispensable. Therefore, we don't know if a worksheet is dispensible or not, so the statement is uncertain.\nANSWER:\tUncertain",
            148: "Let's think step by step. We want to evaluate if Inception is a folk song. We know that Inception is a sci-fi movie. Since all movies are videos and Inception is a movie, it is a video, which means it is visual. On the other hand, we know that all folk songs are songs, and no songs are visual, so no folk songs are visual. Therefore, since Inception is visual but no folk songs are visual, we know that Inception cannot be a folk song, so the statement is false.\nANSWER:\tFalse",
            261: "Let's think step by step. We want to evaluate if Luke can make a good breakfast. From the last premise, we know that Luke can make cookies, scrambled eggs, and muffins. Since Luke can make cookies and muffins, they are a baker. Now, combining the information we have, since Luke is a baker and can make scrambled eggs, this means that they can make a good breakfast. Therefore, Luke can make a good breakfast, so the statement is true.\nANSWER:\tTrue",
            263: "Let's think step by step. We want to evaluate if ETS develops assessments for K-12 statewide as well as entry to US tertiary and quaternary educatiand doon institutions. We know that ETS develops assessments for K-12 statewide. We also know that ETS develops assessments associated with entry to the US tertiary and quaternary education institutes. Therefore, both parts of the conclusion are true, and the statement is true.\nANSWER:\tTrue",
            275: "Let's think step by step. We want to evaluate if James contributes to the country. Let's think about what we know about James. First, we know that James was either sentenced for thief or stayed in prison. However, this doesn't tell us anything about whether James contributed to the country. Second, we know that James either had a bad record in the local state or that he was respected by others. However, the premises don't tell us anything about the relationship between having a bad record and contributing to the country. Therefore, it is uncertain whether James contributes to the country.\nANSWER:\tUncertain",
            299: "Let's think step by step. We want to evaluate if John can get the rhythms right. We know that John is a student learning piano. Since all students learning piano can strike the right notes, John can strike the right notes. Since all students who can strike the right notes can get the rhythms right and John can strike the right notes, John can get the rhythms right, so the conclusion is true.\nANSWER:\tTrue",
            683: "Let's think step by step. We want to evaluate if there is a person from BRICS speaking Hindi. We know that there is an Indian, and since India is one of BRICS, we know that there is an Indian in BRICS. Furthermore, we know that they speak either Hindi or English, however, we don't know which one. Therefore, there could be a person in BRICS speaking Hindi, or there could not. Therefore, it is uncertain whether there is a person from BRICS speaking Hindi.\nANSWER:\tUncertain",
            684: "Let's think step by step. We want to evaluate if Hamilton is a film. Since Daveed Diggs played two roles in the musical Hamilton, Hamilton is a musical. Since musicals are not films and Hamilton is a musical, Hamilton is not a film, and the conclusion is false.\nANSWER:\tFalse",
            850: "Let's think step by step. We want to evaluate if Charlie does not like Leo and does not care for Leo. Let's first evaluate if Charlie does not like Leo. We know Charlie has a naughty pet named Leo. Since pets who are naughty are not liked as much, Charlie does not like Leo. Now, let's evaluate if Charlie cares for Leo. We know that if a person has a pet, they care for that pet. Since Leo is Charlie's pet, Charlie cares for Leo. Therefore, Charlie does not like Leo but cares for Leo, so the second part of the conclusion is false, which means the entire conclusion is false.\nANSWER:\tFalse",
            853: "Let's think step by step. We want to evaluate if the Three Body Problem won the Hugo Award. The only thing we know about the Hugo Award is that some books that have won the Hugo Award were written by Cixin Liu. However, we know nothing about whether The Three Body Problem was written by Cixin Liu, so the conclusion is uncertain.\nANSWER:\tUncertain",
            886: "Let's think step by step. We want to evaluate if Dagfinn is Dagfinn Aarskog's given name. We know that Dagfinn is a given name, and that notable people with the given name Dagfinn includes Dagfinn Aarskog, which means that Dagfinn is Dagfinn Aarskog's given name, so the conclusion is true.\nANSWER:\tTrue",
            892: "Let's think step by step. We want to evaluate if Minsk joined the Scottish Premiership. We know that Minsk and St Johnstone are different teams and that St Johnstone is part of the Scottish Premiership, but we don't know anything about whether or not Minsk joined the Scottish Premiership from the premises. Therefore, the conclusion is uncertain.\nANSWER:\tUncertain",
            930: "Let's think step by step. We want to evaluate if Boves is not in Europe. We know that Boves is a railway station located in France. We also know that since France is a European country, France is located in Europe. Furthermore, we know that if A is located in B and B is located in C, then A is located in C. Therefore, we know that because Boves is located in France and France is located in Europe, that means Boves is located in Europe. Therefore, the conclusion is false.\nANSWER:\tFalse",
            980: "Let's think step by step. We want to evaluate if James is either invited to take a photo with the audience or happy to communicate with each other during the dinner. We know that James does not attend the conference in person and is not provided with souvenirs. There are no premises that apply to people who do not attend the conference. Since James is not provided with souvenirs, since all who attended the conference in person are provided with souvenirs, we know that James did not attend the conference in person. However, we don't know anything else, so it is possible that James was neither invited to take a photo with the audience nor happy to communicate during the dinner. Therefore, the conclusion is false.\nANSWER:\tFalse",
        }
        cots = [None for _ in range(len(train_dataset))]
        for index, cot in train_cots.items():
            cots[index] = cot
        train_dataset = train_dataset.add_column("cot", cots)
        return train_dataset

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self._test

    def get_instructions(self):
        instructions = ""
        instructions += "The following is a first-order logic (FOL) problem.\n"
        instructions += "The problem is to determine whether the conclusion follows from the premises.\n"
        instructions += "The premises are given in the form of a set of first-order logic sentences.\n"
        instructions += "The conclusion is given in the form of a single first-order logic sentence.\n"
        if self._mode == "baseline":
            instructions += f"The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
        else:
            instructions += "The task is to translate each of the premises and conclusions into FOL expressions, "
            if self._mode == "scratchpad":
                instructions += f"and then to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
            elif self._mode == "neurosymbolic":
                instructions += "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
                instructions += "Expressions should be adhere to the format of the Python NLTK package logic module."
        return instructions + "\n\n"

    def format_train_example(self, doc):
        example = self.format_test_example(doc)
        if self._mode == "baseline":
            example += f"{doc['label'].strip()}\n"
        elif self._mode == "cot":
            example += f"{doc['cot']}\n"
        else:
            for premise, fol in zip(doc["premises"], doc["premises-FOL"]):
                example += f"TEXT:\t{premise.strip()}\nFOL:\t{fol.strip()}\n"
            example += f"TEXT:\t{doc['conclusion'].strip()}\nFOL:\t{doc['conclusion-FOL'].strip()}\n"
            if self._mode == "scratchpad":
                example += f"ANSWER:\t{doc['label'].strip()}\n"
        return example + "</EVALUATE>\n"

    def format_test_example(self, doc):
        example = "<PREMISES>\n"
        for premise in doc["premises"]:
            example += f"{premise.strip()}\n"
        example += "</PREMISES>\n"
        example += f"<CONCLUSION>\n{doc['conclusion'].strip()}\n</CONCLUSION>\n"
        example += "<EVALUATE>\n"
        return example

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instructions = self.get_instructions()
        train = self.fewshot_examples()
        test = self.format_test_example(doc)
        prompt = "\n".join([instructions, train, test])
        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["label"]

    def postprocess_generation(self, generation, idx, completion_only=False):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        try:
            if completion_only:
                gen = generation.strip()
            else:
                prefix = self.get_prompt(self.get_dataset()[idx])
                assert generation.startswith(
                    prefix
                ), "Increase `--max_length_generation` to avoid truncation"
                gen = generation[len(prefix) :].strip()
                for stop_word in self.stop_words:
                    gen = gen.split(stop_word)[0].strip()
            if self._mode == "baseline":
                resp = gen.strip()
            elif self._mode == "scratchpad":
                flag = "ANSWER:"
                resp = gen.split(flag)[-1].strip()
            elif self._mode == "neurosymbolic":
                flag = "FOL:"
                parses = [
                    line.replace(flag, "").strip()
                    for line in gen.split("\n")
                    if flag in line
                ]
                premises, conclusion = parses[:-1], parses[-1]
                resp = evaluate(premises, conclusion)
            elif self._mode == "cot":
                flag = "ANSWER:"
                resp = gen.split(flag)[-1].strip()
            else:
                raise ValueError(f"Invalid mode: {self._mode}")
            assert resp in ["True", "False", "Uncertain"], f"Invalid generation: {resp}"
            return resp
        except Exception as e:
            # TODO: explore failure cases and improve postprocessing
            print(f"Error in parsing and/or evaluating LLM output: {e}")
            return self.ERROR_TOKEN

    @staticmethod
    def metric(generations, references, error_token):
        correct = 0
        for gens, ref in zip(generations, references):
            gens = [gen for gen in gens if gen != error_token]
            if len(gens) > 0:
                majority = Counter(gens).most_common(1)[0][0]
                if majority == ref:
                    correct += 1
        return {f"accuracy (pass@1 majority)": correct / len(references)}

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        return self.metric(generations, references, self.ERROR_TOKEN)

    @lru_cache(maxsize=None)
    def fewshot_examples(self):
        """
        Returns a few-shot example for the task.
        :param n: int
            number of examples
        :param seed: int
            seed for random number generator
        :return: str
        """
        examples = []
        for doc in self._train.select(range(self._nshot)):
            examples.append(self.format_train_example(doc))
        return "\n".join(examples)

_CITATION = """
@inproceedings{Tafjord2020ProofWriterGI,
  title={ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language},
  author={Oyvind Tafjord and Bhavana Dalvi and Peter Clark},
  booktitle={Findings},
  year={2020}
}
"""
TASK_REGISTRY = {
    **folio.create_all_tasks(),
    **proofwriter.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))

def create_all_tasks():
    def create_task(mode, n):
        class ProofWriter(ProofWriterBase):
            def __init__(self):
                super().__init__(mode, n)

        return ProofWriter

    return {
        f"proofwriter-{mode}-{n}shot": create_task(mode, n)
        for mode in ["baseline", "scratchpad", "neurosymbolic", "cot"]
        for n in [1, 2, 4, 8, 16]
    }


class ProofWriterBase(OWAFOLTask):
    DATASET_PATH = "theoxo/proofwriter-deduction-balanced"
    DATASET_NAME = None

    def __init__(self, mode, n, seed=7):
        super().__init__(mode, n)
        self._test = self.reformat(self.dataset["test"]).shuffle(seed)


    def reformat(self, dataset):

        def punctuate(s):
            if s[-1] not in [".", "?", "!"]:
                s += "."
            return s

        def reformat_sample(sample):
            sample["premises"] = [punctuate(p) for p in sample.pop("theory").split(". ")]
            sample["conclusion"] = punctuate(sample.pop("question"))
            sample["label"] = sample.pop("answer")
            return sample

        return dataset.map(reformat_sample)

def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        n_tasks=None,
        n_copies=1,
        prefix="",
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix

    def __iter__(self):
        prompts = []
        infill = []
        for sample in range(self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                infill.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                assert set(prompt_contents.keys()) == {"prefix", "suffix"}
                infill.append(True)
                prompt = self.prefix + self._make_infill_prompt(**prompt_contents)
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)

        if not len(set(infill)) == 1:
            raise ValueError("Mixed infill and completion prompts are not supported.")
        global INFILL_MODE
        INFILL_MODE = infill[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )

        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[sample],
                    "task_id": sample,
                    "input_len": outputs.attention_mask[sample].sum(),
                }

    def _make_infill_prompt(self, prefix, suffix):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    batch_size=20,
    prefix="",
    postprocess=True,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    gen_token_dict = defaultdict(list)
    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        with torch.no_grad():
            if task.stop_words:
                gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["ids"][:, : batch["input_len"]],
                num_return_sequences=batch_size,
                **gen_kwargs,
            )
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_tokens)

    def parse_infill(code, tokenizer):
        """Reorder infill code and remove remaining special tokens."""
        model_id = tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            prefix, suffix, infill = code.split("<|mask:0|>", 2)
            infill = infill.split("<|endofmask|>")[0]
        elif model_id in ["bigcode/santacoder"]:
            prefix, rest = code.split("<fim-suffix>", 1)
            suffix, infill = rest.split("<fim-middle>", 1)
            infill = infill.split("<|endoftext|>")[0]
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")
        code = "".join([prefix, infill, suffix])
        for k, v in tokenizer.special_tokens_map.items():
            if k == "additional_special_tokens":
                for t in v:
                    code = code.replace(t, "")
            else:
                code = code.replace(v, "")
        return code

    code_gens_raw = [[] for _ in range(n_tasks)]
    code_gens_prc = [[] for _ in range(n_tasks)]
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            if INFILL_MODE:
                gen_code = parse_infill(
                    tokenizer.decode(
                        s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    tokenizer,
                )
            else:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            code_gens_raw[sample].append(gen_code[len(prefix) :])
            if postprocess:
                code_gens_prc[sample].append(
                    task.postprocess_generation(gen_code[len(prefix) :], int(sample))
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens_prc[sample].append(gen_code[len(prefix) :])

    return code_gens_prc, code_gens_raw


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


def parallel_generations(task, dataset, accelerator, model, tokenizer, n_tasks, args):
    if args.generations_path:
        with open(args.generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length_generation,
    }
    if task.stop_words:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )

    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = args.n_samples // args.batch_size

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    model, ds_loader = accelerator.prepare(model, ds_loader)
    generations_prc, generations_raw = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        prefix=args.prefix,
        postprocess=args.postprocess,
        **gen_kwargs,
    )
    return generations_prc, generations_raw

# Evaluators are from LINC code (https://github.com/benlipkin/linc/blob/718dfe8fae342028a837a96096e5e1e412ad4f2e/eval/evaluator.py):
_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The task you are about to use executes untrusted model-generated code.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator(ABC):
    def __init__(self, args):
        self.args = args
        self.allow_code_execution = args.allow_code_execution

    @abstractmethod
    def generate_text(self, task_name):
        pass

    def evaluate(self, task_name):
        task = tasks.get_task(task_name)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations_prc, generations_raw, references = self.generate_text(task_name)
        if len(generations_prc[0]) != self.args.n_samples:
            generations_prc = [l[: self.args.n_samples] for l in generations_prc]
            warnings.warn(
                "Number of tasks wasn't proportional to number of devices, we removed extra predictions"
            )

        if not hasattr(self, "accelerator") or self.accelerator.is_main_process:
            if not self.args.generations_path:
                if self.args.save_generations_raw:
                    with open(self.args.save_generations_raw_path, "w") as fp:
                        json.dump(generations_raw, fp)
                        print("raw generations were saved")
                if self.args.save_generations_prc:
                    with open(self.args.save_generations_prc_path, "w") as fp:
                        json.dump(generations_prc, fp)
                        print("processed generations were saved")
                if self.args.save_references:
                    with open(self.args.save_references_path, "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            results = task.process_results(generations_prc, references)
            return results


class HFEvaluator(Evaluator):
    def __init__(self, accelerator, model, tokenizer, args):
        super().__init__(args)
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        generations_prc, generations_raw = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        return generations_prc, generations_raw, references


class OAIEvaluator(Evaluator):
    def __init__(self, args, chat=False):
        super().__init__(args)
        self.chat = chat
        self.model = args.model
        self.api_keys = [os.environ[key] for key in args.openai_api_env_keys]
        self.cache = Cache(args.cache_dir)
        assert (
            len(self.api_keys) >= 1
        ), "You must provide at least one OpenAI API key to use OAIEvaluator"

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        prompts = [task.get_prompt(dataset[i]) for i in range(n_tasks)]
        stops = [task.stop_words for _ in range(n_tasks)]

        with ThreadPoolExecutor() as executor:
            res = executor.map(self.get_completion, prompts, stops)
        generations_raw = list(res)
        if self.args.postprocess:
            generations_prc = [
                [
                    task.postprocess_generation(
                        generations_raw[i][j], i, completion_only=True
                    )
                    for j in range(self.args.n_samples)
                ]
                for i in range(n_tasks)
            ]
        else:
            generations_prc = generations_raw
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        return generations_prc, generations_raw, references

    def make_request(self, prompt, stop):
        if self.chat:
            response = openai.ChatCompletion.create(
                model=self.model,
                n=self.args.n_samples,
                messages=[
                    {"role": "system", "content": self.args.chat_system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.args.temperature,
                max_tokens=self.args.max_length_generation,
                top_p=self.args.top_p,
                stop=stop,
                stream=False,
            )
        else:
            response = openai.Completion.create(
                engine=self.model,
                n=self.args.n_samples,
                prompt=prompt,
                temperature=self.args.temperature,
                max_tokens=self.args.max_length_generation,
                top_p=self.args.top_p,
                stop=stop,
                stream=False,
            )
        return response

    def get_completion(self, prompt, stop, api_key=None, exhausted={}, retry_after=60):
        if self.args.temperature == 0:
            request_id = "_".join(
                str(x)
                for x in [
                    self.model,
                    self.args.n_samples,
                    prompt,
                    self.args.max_length_generation,
                    stop,
                ]
            )
            request_key = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            if request_key in self.cache:
                print(
                    "Identical OpenAI API request previously executed. Loading response from cache."
                )
                return self.cache[request_key]
        if api_key is None:
            api_key = random.choice(self.api_keys)
        openai.api_key = api_key
        try:
            response = self.make_request(prompt, stop)
        except openai.error.RateLimitError:
            if len(self.api_keys) == 1:
                warn(
                    f"Only one API key was provided, and it has been rate limited. sleeping for {retry_after}s. Please provide more API keys to avoid sleeping."
                )
                time.sleep(retry_after)
                return self.get_completion(prompt, stop, api_key)
            else:
                print(f"Rate limit error; trying again with a different API key.")
                exhausted[api_key] = time.time()
                exhausted = {
                    k: v
                    for k, v in exhausted.items()
                    if (time.time() - v) < retry_after
                }
                if len(exhausted) == len(self.api_keys):
                    print(
                        f"All API keys have been exhausted. sleeping for {retry_after}s then trying again with all keys."
                    )
                    time.sleep(retry_after)
                    exhausted = {}
                try_next = random.choice(
                    [k for k in self.api_keys if k != api_key and k not in exhausted]
                )
                return self.get_completion(prompt, stop, try_next, exhausted)
        except (openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
            print(f"API Error; sleeping for {retry_after}s then trying again.")
            time.sleep(retry_after)
            return self.get_completion(prompt, stop, api_key, exhausted)
        if self.chat:
            response = [c["message"]["content"] for c in response["choices"]]
        else:
            response = [c["text"] for c in response["choices"]]
        if self.args.temperature == 0:
            print("Temperature is 0, caching OpenAI API response for future use.")
            self.cache[request_key] = response
        return response


def main(model_name, tasks="folio", output_dir="output", precision="fp16"):
    from accelerate import Accelerator
    import transformers

    # Initialize Accelerator for distributed setup
    accelerator = Accelerator()
    evaluator = None
    token = ''

    # Determine if using Hugging Face or OpenAI models
    if "gpt" in model_name:  # OpenAI model names typically start with "gpt"
        # OpenAI API based evaluation
        evaluator = OAIEvaluator(model_name=model_name)
    else:
        # Hugging Face model setup
        precision_mapping = {"fp16": torch.float16, "fp32": torch.float32}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=token, torch_dtype=precision_mapping.get(precision, torch.float32)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        evaluator = HFEvaluator(accelerator, model, tokenizer, args)

    # Load and evaluate tasks
    results = {}
    for task in [task for task in ALL_TASKS if task.startswith(tasks)]:
        results[task] = evaluator.evaluate(task)

    # Save results
    with open(f"{output_dir}/linc_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LINC evaluation on FOLIO")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for evaluation")
    parser.add_argument("--tasks", type=str, default="folio", help="Tasks to run")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--precision", type=str, default="fp16", help="Precision for model")
    args = parser.parse_args()
    main(args.model_name, args.tasks, args.output_dir, args.precision)
