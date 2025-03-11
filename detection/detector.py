import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from detection.utils.detectgpt import perturb_texts, DetectGPTArgs, load_mask_filling_model


class CodeDetectorABC(ABC):
    @abstractmethod
    def compute_score(self, chat: List[dict]) -> float:
        # Compute score using the conditional probability distribution (conditioned by a task)
        pass

    @abstractmethod
    def compute_score_without_task(self, code: str) -> float:
        # Compute score using the conditional probability distribution
        pass

    @abstractmethod
    def compute_score_infer_task(self, code: str, cached_task: Optional[str] = None) -> float:
        # Compute score using the conditional probability distribution (conditioned by an approximated task)
        pass

    @staticmethod
    def _validate_chat(chat: List[dict]):
        for message in chat:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Unsupported role in chat: {message['role']}")

        assert chat[-1]["role"] == "assistant", "The last message in the chat must be the assistant's answer."


class LogitsBasedDetector(CodeDetectorABC):
    def __init__(self, model_name: str, pattern_weight_mapping: Optional[dict] = None,
                 infer_task_cfg: Optional['InferTaskConfig'] = None, language: str = 'python'):
        if pattern_weight_mapping is None:
            pattern_weight_mapping = {}
        self.model_name = model_name
        self.pattern_weight_mapping = pattern_weight_mapping
        self.infer_task_cfg = infer_task_cfg
        self.language = language
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="cuda")

    def _get_chat_tokens(self, chat: List[dict]):
        # This function returns the input tokens for a given chat, alongside the number of assistant tokens
        # The number of assistant tokens is needed to ignore the task prompt when computing the score
        CodeDetectorABC._validate_chat(chat)

        raw_chat_without_assistant_answer = self.tokenizer.apply_chat_template(chat[:-1], tokenize=False)
        raw_chat = self.tokenizer.apply_chat_template(chat, tokenize=False)

        raw_input_without_assistant_answer = self.tokenizer(raw_chat_without_assistant_answer, return_tensors="pt")
        raw_input = self.tokenizer(raw_chat, return_tensors="pt", return_offsets_mapping=True)

        num_assistant_tokens = raw_input["input_ids"].size(1) - raw_input_without_assistant_answer["input_ids"].size(1)
        if num_assistant_tokens <= 0:
            raise ValueError(
                "num_assistant_tokens must be greater than 0. Ensure the input includes the assistant's answer.")

        return raw_input, raw_chat, num_assistant_tokens

    def _find_interesting_tokens(self, raw_input: dict, code: str, code_start_index: int):
        # This function returns a mapping of token indices to assigned weights for score computation
        # Currently it supports assigning different weights for comments and docstring
        # We use it to ignore the scores of comment and docstrings in score computation
        comment_patterns = {
            "python": '^\s*(#.*)$',
            "java": r"^\s*(//.*)$",
            "cpp": r"^\s*(//.*)$"
        }

        docstring_patterns = {
            "python": r'\s*(""".*?""")',
            "java": r"(/\*.*?\*/)",
            "cpp": r"(/\*.*?\*/)"
        }

        ranges_to_weights = []
        for pattern, weight in self.pattern_weight_mapping.items():
            if pattern == 'comments':
                ranges = [(m.start() + code_start_index, m.end() + code_start_index) for m in
                          re.finditer(comment_patterns[self.language], code[code_start_index:], flags=re.MULTILINE)]
                if len(ranges) > 0:
                    ranges_to_weights.append((ranges, weight))
            elif pattern == 'docstrings':
                ranges = [(m.start(1) + code_start_index, m.end(1) + code_start_index) for m in
                          re.finditer(docstring_patterns[self.language], code[code_start_index:], flags=re.DOTALL)]
                if len(ranges) > 0:
                    ranges_to_weights.append((ranges, weight))
            else:
                raise ValueError(f"Unsupported pattern: {pattern}")

        offsets = raw_input.pop("offset_mapping")
        tokens_to_weights = []
        for ranges, weight in ranges_to_weights:
            for s, e in ranges:
                matching_token_indices = [
                    i for i, (start, end) in enumerate(offsets.squeeze()) if start >= s and end <= e
                ]
                tokens_to_weights.append((matching_token_indices, weight))

        return tokens_to_weights

    def compute_score(self, chat: List[dict]) -> float:
        raw_input, raw_chat, num_assistant_tokens = self._get_chat_tokens(chat)
        code_start_index = raw_chat.index(f'```{self.language}')
        tokens_to_weights = self._find_interesting_tokens(raw_input, raw_chat, code_start_index)

        # Each detector that inherits from LogitsBasedDetector should implement _compute_score_impl
        return self._compute_score_impl(raw_input, num_assistant_tokens, tokens_to_weights)

    def _compute_score_impl(self, raw_input: dict, num_assistant_tokens: int,
                            tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        raise NotImplementedError()

    def compute_score_without_task(self, code: str) -> float:
        raw_input = self.tokenizer(code, return_tensors="pt", return_offsets_mapping=True)
        tokens_to_weights = self._find_interesting_tokens(raw_input, code, code_start_index=0)

        # Each detector that inherits from LogitsBasedDetector should implement _compute_score_without_task_impl
        return self._compute_score_without_task_impl(raw_input, tokens_to_weights)

    def _compute_score_without_task_impl(self, raw_input: dict,
                                         tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        raise NotImplementedError()

    def _infer_task(self, code: str):
        # The function used for task approximation
        if self.infer_task_cfg.prompt_style == 'regular':
            chat = [
                {"role": "system", "content": f"You are a {self.language.title()} developer."},
                {"role": "user",
                 "content": f"Based on the provided code snippet, create a simple one line task description that, when given to an LLM, would likely result in the generation of a similar piece of code. The output should be wrapped in <task> and </task>. \n\n ```{self.language}\n" + code + "\n```"}
            ]
        elif self.infer_task_cfg.prompt_style == 'short':
            chat = [
                {"role": "system", "content": f"You are a {self.language.title()} developer."},
                {"role": "user",
                 "content": f"Based on the provided code snippet, create a very short and simple task that, when given to an LLM, would likely result in the generation of a similar piece of code. The output should be wrapped in <task> and </task>. \n\n ```{self.language}\n" + code + "\n```"}
            ]
        elif self.infer_task_cfg.prompt_style == 'long':
            chat = [
                {"role": "system", "content": f"You are a {self.language.title()} developer."},
                {"role": "user",
                 "content": f"Based on the provided code snippet, create a long and detailed task description that, when given to an LLM, would likely result in the generation of a similar piece of code. The output should be wrapped in <task> and </task>. \n\n ```{self.language}\n" + code + "\n```"}
            ]
        elif self.infer_task_cfg.prompt_style == 'storytelling':
            chat = [
                {"role": "system",
                 "content": f"You are writing a programming questions textbook. Each question is based on a short fictional story, and the reader is required to write a piece of code that solves the question in the story."},
                {"role": "user",
                 "content": f"Here are a few examples of stories from the textbook: \n\n"
                            "<task> Noah was a software engineer building a chatbot for customer support. During testing, he noticed that users sometimes included extra punctuation at the start and end of their messages, confusing the bot. To make responses more accurate, he needed a function that could strip these unnecessary characters. Help him write the relevant piece of code. </task>\n\n"
                            "<task> Lisa loved solving puzzles. She was designing a word game where players submitted answers, but some people accidentally added spaces or special characters at the edges. To ensure fairness, Lisa needed a way to clean up the inputs before checking them. Help her write the relevant piece of code. </task>"
                            f"\n\n Based on the provided code snippet, create the required story description that would likely result in the generation of a similar piece of code. The entire output, should be wrapped in <task> and </task>. \n\n ```{self.language}\n" + code + "\n```"}
            ]
        elif self.infer_task_cfg.prompt_style == 'pseudocode':
            chat = [
                {"role": "system",
                 "content": f"You are a {self.language.title()} developer experienced in writing structured pseudocode."},
                {"role": "user",
                 "content": f"Translate the given code snippet into a pseudocode-like task. The output should be wrapped in <task> and </task>. \n\n ```{self.language}\n" + code + "\n```"}
            ]
        elif self.infer_task_cfg.prompt_style == 'friendly':
            chat = [
                {"role": "system",
                 "content": f"You are a {self.language.title()} developer helping a friend understand coding tasks."},
                {"role": "user",
                 "content": f"Based on the provided code snippet, create a task description that, when given to an LLM, would likely result in the generation of a similar piece of code. Write the task description in a casual and friendly manner, as if explaining to a peer who is learning {self.language}. The tone should be engaging and approachable. The output should be wrapped in <task> and </task>. \n\n ```{self.language}\n" + code + "\n```"}
            ]
        elif self.infer_task_cfg.prompt_style == 'critical':
            chat = [
                {"role": "system",
                 "content": f"You are a no-nonsense {self.language.title()} developer who has no patience for inefficiency or poorly written code."},
                {"role": "user",
                 "content": f"Write a brutally honest task description that, when given to an LLM, would likely result in the generation of a similar piece of code. The tone should be direct, demanding, and critical. Do not sugarcoat anything. The output should be wrapped in <task> and </task>. \n\n ```{self.language}\n" + code + "\n```"}
            ]
        else:
            raise ValueError(f"Unsupported prompt style: {self.infer_task_cfg.prompt_style}")

        with torch.no_grad():
            raw_input = self.tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to("cuda")
            attention_mask = raw_input.ne(self.tokenizer.eos_token_id).to("cuda")

            output = self.model.generate(raw_input, attention_mask=attention_mask,
                                         max_length=raw_input.shape[-1] + 1000, temperature=0.7,
                                         top_p=0.95, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
            output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            output = output.split("<task>")[-1].split("</task>")[0].strip()
            print(f"Generated task: {output}")

        return output

    def compute_score_infer_task(self, code: str, cached_task: Optional[str] = None) -> \
            Tuple[float, str]:
        # We approximate the task and then run self.compute_score with the approximated task
        if self.infer_task_cfg.use_cache and cached_task:
            task = cached_task
        else:
            task = self._infer_task(code)

        chat = [
            {"role": "system", "content": f"You are a {self.language.title()} developer."},
            {"role": "user",
             "content": task + f'\n\n The {self.language} code should be organized in a single markdown block.'},
            {"role": "assistant", "content": f"```{self.language}\n" + code + "\n```"}
        ]

        return self.compute_score(chat), task


class MeanLogLikelihoodDetector(LogitsBasedDetector):
    @staticmethod
    def compute_ll(model, raw_input, tokens_for_loss, tokens_to_weights):
        labels = raw_input["input_ids"].clone()

        # -100 label is ignored in CrossEntropyLoss
        # https://discuss.huggingface.co/t/will-trainer-loss-functions-automatically-ignore-100/36134
        labels[:, :-tokens_for_loss] = -100
        # Generate mask for ignoring tokens
        if tokens_to_weights is None:
            tokens_to_weights = {}
        for tokens, weight in tokens_to_weights:
            if labels.shape[-1] in tokens:
                tokens.remove(labels.shape[-1])
            if weight != 0:
                raise ValueError("MeanLogLikelihoodDetector currently does not support non-zero weights.")
            labels[:, tokens] = -100

        with torch.no_grad():
            raw_input = {key: value.to("cuda") for key, value in raw_input.items()}
            output = model(**raw_input, labels=labels)

        return -output.loss.item()

    def _compute_score_impl(self, raw_input: dict, num_assistant_tokens: int,
                            tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        return MeanLogLikelihoodDetector.compute_ll(self.model, raw_input, num_assistant_tokens, tokens_to_weights)

    def _compute_score_without_task_impl(self, raw_input: dict,
                                         tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        return MeanLogLikelihoodDetector.compute_ll(self.model, raw_input, raw_input["input_ids"].size(1),
                                                    tokens_to_weights)


class EntropyDetector(LogitsBasedDetector):
    def _compute_neg_entropy(self, raw_input: dict, num_assistant_tokens: int,
                             tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        with torch.no_grad():
            raw_input = {key: value.to("cuda") for key, value in raw_input.items()}
            output = self.model(**raw_input)

            # Extract logits (unnormalized probabilities)
            logits = output.logits[:, :-1]  # Shape: (batch_size=1, seq_length, vocab_size)

            # Compute token probabilities using softmax
            probs = F.softmax(logits, dim=-1)  # Shape: (1, seq_length, vocab_size)

            # Compute entropy for each token
            entropy = -(probs * probs.log()).sum(dim=-1)  # Shape: (1, seq_length)

            # Generate mask for ignoring tokens
            if tokens_to_weights is None:
                tokens_to_weights = {}
            mask = torch.ones_like(entropy)
            for tokens, weight in tokens_to_weights:
                if mask.shape[-1] in tokens:
                    tokens.remove(mask.shape[-1])
                mask[:, tokens] = weight

            # Apply mask to logits
            mask = mask[:, -num_assistant_tokens:]
            entropy = entropy[:, -num_assistant_tokens:] * mask

            # Compute mean entropy across all tokens
            avg_entropy = (entropy.sum() / mask.sum()).item()

        return -avg_entropy

    def _compute_score_impl(self, raw_input: dict, num_assistant_tokens: int,
                            tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        return self._compute_neg_entropy(raw_input, num_assistant_tokens, tokens_to_weights)

    def _compute_score_without_task_impl(self, raw_input: dict,
                                         tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        # Compute entropy on all tokens (since we do not have a task)
        return self._compute_neg_entropy(raw_input, raw_input["input_ids"].size(1), tokens_to_weights)


class RankDetector(LogitsBasedDetector):
    def __init__(self, model_name: str, pattern_weight_mapping: dict,
                 infer_task_cfg: Optional['InferTaskConfig'] = None, language: str = 'python'):
        super().__init__(model_name, pattern_weight_mapping, infer_task_cfg, language)
        self._log = False

    @staticmethod
    def compute_rank(model: AutoModelForCausalLM, raw_input: dict, num_assistant_tokens: int,
                     apply_log: bool, tokens_to_weights) -> float:
        # Adapted from https://github.com/eric-mitchell/detect-gpt/
        with torch.no_grad():
            raw_input = {key: value.to("cuda") for key, value in raw_input.items()}
            logits = model(**raw_input).logits[:, :-1]
            labels = raw_input['input_ids'][:, 1:]

            # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1  # convert to 1-indexed rank
            if apply_log:
                ranks = torch.log(ranks)

            # avg_rank = ranks[-num_assistant_tokens:].float().mean().item()

            # Generate mask for ignoring tokens
            if tokens_to_weights is None:
                tokens_to_weights = {}
            mask = torch.ones_like(ranks)
            for tokens, weight in tokens_to_weights:
                if mask.shape[-1] in tokens:
                    tokens.remove(mask.shape[-1])
                mask[tokens] = weight

            # Apply mask to logits
            mask = mask[-num_assistant_tokens:]
            ranks = ranks[-num_assistant_tokens:] * mask

            avg_rank = (ranks.float().sum() / mask.sum()).item()

        return -avg_rank

    def _compute_score_impl(self, raw_input: dict, num_assistant_tokens: int,
                            tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        return RankDetector.compute_rank(self.model, raw_input, num_assistant_tokens, self._log, tokens_to_weights)

    def _compute_score_without_task_impl(self, raw_input: dict,
                                         tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        # Compute rank on all tokens (since we do not have a task)
        return RankDetector.compute_rank(self.model, raw_input, raw_input["input_ids"].size(1), self._log,
                                         tokens_to_weights)


class LogRankDetector(RankDetector):
    def __init__(self, model_name: str, pattern_weight_mapping: dict,
                 infer_task_cfg: Optional['InferTaskConfig'] = None, language: str = 'python'):
        super().__init__(model_name, pattern_weight_mapping, infer_task_cfg, language)
        self._log = True


class LRRDetector(RankDetector):
    def _compute_score_impl(self, raw_input: dict, num_assistant_tokens: int,
                            tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        mll = MeanLogLikelihoodDetector.compute_ll(self.model, raw_input, num_assistant_tokens, tokens_to_weights)
        mlr = RankDetector.compute_rank(self.model, raw_input, num_assistant_tokens, apply_log=True,
                                        tokens_to_weights=tokens_to_weights)

        # Mean log rank is already returned negative, so we do not need to negate again
        return mll / mlr

    def _compute_score_without_task_impl(self, raw_input: dict,
                                         tokens_to_weights: List[Tuple[List[int], float]]) -> float:
        mll = MeanLogLikelihoodDetector.compute_ll(self.model, raw_input, raw_input["input_ids"].size(1),
                                                   tokens_to_weights)
        mlr = RankDetector.compute_rank(self.model, raw_input, raw_input["input_ids"].size(1), apply_log=True,
                                        tokens_to_weights=tokens_to_weights)

        # Mean log rank is already returned negative, so we do not need to negate again
        try:
            return mll / mlr
        except ZeroDivisionError:
            # Encountered one edge case in codeparrot, for CodeLlama13b
            return mll / 1e-6


class DetectGPT(LogitsBasedDetector):
    def __init__(self, model_name: str, args: DetectGPTArgs):
        super().__init__(model_name)
        self.args = args
        self.mask_tokenizer, self.mask_model = load_mask_filling_model(args.mask_filling_model_name)

    @staticmethod
    def _calc_final_score(original_score, score_per_perturbation):
        mean_perturbed_ll = np.mean([i for i in score_per_perturbation if not math.isnan(i)])
        std_perturbed_ll = np.std([i for i in score_per_perturbation if not math.isnan(i)]) if len(
            [i for i in score_per_perturbation if not math.isnan(i)]) > 1 else 1
        return (original_score - mean_perturbed_ll) / std_perturbed_ll

    def _get_perturbed_chats(self, chat: List[dict]) -> List[List[dict]]:
        self._validate_chat(chat)
        perturbed_texts = perturb_texts([chat[-1]["content"] for _ in range(self.args.n_perturbations)],
                                        self.mask_tokenizer, self.mask_model, args=self.args, ceil_pct=True)

        perturbed_chats = [chat] + [chat[:-1] + [{"role": "assistant", "content": perturbed_text}] for perturbed_text in
                                    perturbed_texts]

        return perturbed_chats

    def compute_score(self, chat: List[dict]) -> float:
        perturbed_chats = self._get_perturbed_chats(chat)

        score_per_perturbed_chat = []
        for perturbed_chat in tqdm.tqdm(perturbed_chats, desc='Computing MLL per perturbed chat'):
            raw_input, _, num_assistant_tokens = self._get_chat_tokens(perturbed_chat)

            labels = raw_input["input_ids"].clone()
            labels[:, :-num_assistant_tokens] = -100

            with torch.no_grad():
                raw_input = {key: value.to("cuda") for key, value in raw_input.items()}
                output = self.model(**raw_input, labels=labels)

            score_per_perturbed_chat.append(-output.loss.item())

        return DetectGPT._calc_final_score(score_per_perturbed_chat[0], score_per_perturbed_chat[1:])

    def compute_score_without_task(self, code: str) -> float:
        perturbed_texts = perturb_texts([code for _ in range(self.args.n_perturbations)], self.mask_tokenizer,
                                        self.mask_model, args=self.args, ceil_pct=True)

        score_per_perturbed_text = []
        for perturbed_text in tqdm.tqdm([code] + perturbed_texts, desc='Computing MLL per perturbed text'):
            raw_input = self.tokenizer(perturbed_text, return_tensors="pt")
            labels = raw_input["input_ids"].clone()

            with torch.no_grad():
                raw_input = {key: value.to("cuda") for key, value in raw_input.items()}
                output = self.model(**raw_input, labels=labels)

            score_per_perturbed_text.append(-output.loss.item())

        return DetectGPT._calc_final_score(score_per_perturbed_text[0], score_per_perturbed_text[1:])


class NPRDetector(DetectGPT):
    def __init__(self, model_name: str, args: DetectGPTArgs):
        super().__init__(model_name, args)

    @staticmethod
    def _calc_final_score(original_score, score_per_perturbation):
        mean_perturbed_log_rank = np.mean([i for i in score_per_perturbation if not math.isnan(i)])
        return mean_perturbed_log_rank / original_score

    def compute_score(self, chat: List[dict]) -> float:
        perturbed_chats = self._get_perturbed_chats(chat)

        score_per_perturbed_chat = []
        for perturbed_chat in tqdm.tqdm(perturbed_chats, desc='Computing LogRank per perturbed chat'):
            raw_input, _, num_assistant_tokens = self._get_chat_tokens(perturbed_chat)

            score_per_perturbed_chat.append(
                RankDetector.compute_rank(self.model, raw_input, num_assistant_tokens, apply_log=True, tokens_to_weights=None))

        return NPRDetector._calc_final_score(score_per_perturbed_chat[0], score_per_perturbed_chat[1:])

    def compute_score_without_task(self, code: str) -> float:
        perturbed_texts = perturb_texts([code for _ in range(self.args.n_perturbations)], self.mask_tokenizer,
                                        self.mask_model, args=self.args, ceil_pct=True)

        score_per_perturbed_text = []
        for perturbed_text in tqdm.tqdm([code] + perturbed_texts, desc='Computing LogRank per perturbed text'):
            raw_input = self.tokenizer(perturbed_text, return_tensors="pt")

            score_per_perturbed_text.append(
                RankDetector.compute_rank(self.model, raw_input, raw_input["input_ids"].size(1), apply_log=True, tokens_to_weights=None))

        return NPRDetector._calc_final_score(score_per_perturbed_text[0], score_per_perturbed_text[1:])


class OpenAIDetector(CodeDetectorABC):
    def __init__(self, model_name: str):
        assert model_name in ["roberta-base-openai-detector", "roberta-large-openai-detector"]
        self.model_name = model_name
        self.detector = AutoModelForSequenceClassification.from_pretrained(self.model_name, device_map="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def compute_score(self, chat: List[dict]) -> float:
        raise NotImplementedError()

    def compute_score_without_task(self, code: str) -> float:
        # No batching support yet
        # truncation=True without explicitly setting max_length truncates at the maximum model input size
        raw_input = self.tokenizer(code, return_tensors="pt", truncation=True)
        raw_input = {key: value.to("cuda") for key, value in raw_input.items()}
        pred = self.detector(**raw_input).logits.softmax(-1)[:, 0].item()
        return pred

    def compute_score_infer_task(self, code: str, cached_task: Optional[str] = None) -> float:
        raise NotImplementedError()
