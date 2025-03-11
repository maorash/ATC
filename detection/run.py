import os
import gc
import random
from typing import Optional, List, Dict, Tuple

import torch
import pyrallis
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict

from code_datasets.prompts import get_prompts, PromptVariant
from detection.code_augmentor import CodeAugmentor
from detection.utils.detectgpt import DetectGPTArgs
from detection.detector import MeanLogLikelihoodDetector, EntropyDetector, RankDetector, LogRankDetector, DetectGPT, \
    NPRDetector, LRRDetector, OpenAIDetector, CodeDetectorABC
from code_datasets.loader import RawDatasetSaver, CodeDataset
from code_datasets.response_generator import HFModel


@dataclass
class AugmentationConfig:
    apply_augmentors: bool = False
    apply_for_human_code: bool = True
    methods: List[str] = field(default_factory=lambda: [])


@dataclass
class DetectorConfig:
    method: str = 'entropy'
    base_model_name: str = 'codellama/CodeLlama-7b-Instruct-hf'
    detectgpt_args: DetectGPTArgs = field(default_factory=DetectGPTArgs)
    pattern_weight_mapping: Dict[str, float] = field(default_factory=lambda: {})
    pattern_weight_mapping_str: str = 'comments:0,docstrings:0'

    def __post_init__(self):
        # A builtin method of dataclasses, used for post-processing our configuration.
        if len(self.pattern_weight_mapping_str) > 0:
            if self.pattern_weight_mapping_str.lower() == 'none':
                self.pattern_weight_mapping_str = ''
            else:
                self.pattern_weight_mapping = {k.strip(): float(v.strip()) for k, v in
                                               [x.split(':') for x in self.pattern_weight_mapping_str.split(',')]}


@dataclass
class InferTaskConfig:
    debug: bool = False
    debug_file: str = 'debug_documentation.txt'
    use_cache: bool = True
    prompt_style: str = 'regular'


@dataclass
class BenchmarkConfig:
    detector_cfg: DetectorConfig = field(default_factory=DetectorConfig)
    use_score_cache: bool = True
    augmentation_cfg: AugmentationConfig = field(default_factory=AugmentationConfig)
    original_dataset_name: str = 'google-research-datasets/mbpp'
    tasks_path: str = 'results/google-research-datasets_mbpp'
    scores_output_path: str = 'results/scores'
    use_original_task: bool = True
    infer_task: bool = False
    infer_task_cfg: InferTaskConfig = field(default_factory=InferTaskConfig)
    calculate_human_scores: bool = False
    calculate_aigc_scores: bool = True
    memory_clean_interval: int = 100
    filter_results_file_by_str: str = ''
    language: str = 'python'
    seed: int = 142


def init_detector(cfg: BenchmarkConfig):
    args = {
        'model_name': cfg.detector_cfg.base_model_name,
        'pattern_weight_mapping': cfg.detector_cfg.pattern_weight_mapping,
        'infer_task_cfg': cfg.infer_task_cfg,
        'language': cfg.language
    }
    if cfg.detector_cfg.method == 'mean_log_likelihood':
        return MeanLogLikelihoodDetector(**args)
    elif cfg.detector_cfg.method == 'entropy':
        return EntropyDetector(**args)
    elif cfg.detector_cfg.method == 'log_rank':
        return LogRankDetector(**args)
    elif cfg.detector_cfg.method == 'lrr':
        return LRRDetector(**args)
    elif cfg.detector_cfg.method == 'detectgpt':
        return DetectGPT(cfg.detector_cfg.base_model_name, cfg.detector_cfg.detectgpt_args)
    elif cfg.detector_cfg.method == 'npr':
        return NPRDetector(cfg.detector_cfg.base_model_name, cfg.detector_cfg.detectgpt_args)
    elif cfg.detector_cfg.method == 'supervised':
        return OpenAIDetector(cfg.detector_cfg.base_model_name)
    else:
        raise ValueError("Unsupported detector method")


def run_augmentors(code: str, augmentors: AugmentationConfig, language: str):
    if language != 'python':
        raise ValueError("Unsupported language for augmentation")
    for method in augmentors.methods:
        if getattr(CodeAugmentor, method, None) is not None:
            code = getattr(CodeAugmentor, method)(code)
        else:
            raise ValueError(f"Unsupported augmentation method: {method}")

    return code


def set_seed(seed: int):
    random.seed(seed)  # Python’s built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU (if used)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Hash randomization


def main(cfg: BenchmarkConfig, detector: Optional[CodeDetectorABC] = None):
    set_seed(cfg.seed)
    dataset_canonical_name = cfg.original_dataset_name.replace('/', '_')

    # Choose the output folder name (based on the given path, the augmentations and the detector method)
    if cfg.augmentation_cfg.apply_augmentors:
        scores_output_folder_path = os.path.join(cfg.scores_output_path, '_'.join(cfg.augmentation_cfg.methods),
                                                 cfg.detector_cfg.method)
    else:
        scores_output_folder_path = os.path.join(cfg.scores_output_path, cfg.detector_cfg.method)

    # Create the output folder and save the run configuration
    os.makedirs(scores_output_folder_path, exist_ok=True)
    print(asdict(cfg))
    pyrallis.dump(cfg, open(os.path.join(scores_output_folder_path, 'run_config.yaml'), 'w'))

    # Go over all task files in the given path (the task files are generated using code_datasets/generation_pipeline.py)
    results = {}
    for tasks_file_name in os.listdir(cfg.tasks_path):
        print(f"Calculating scores for file: {tasks_file_name}")
        if tasks_file_name.endswith('_responses.jsonl') and cfg.filter_results_file_by_str in tasks_file_name:
            infer_task_cache_file_path, results_file_path = choose_output_file_paths(cfg, dataset_canonical_name,
                                                                                     scores_output_folder_path,
                                                                                     tasks_file_name)

            results[results_file_path] = {}

            # Load the tasks from the responses file
            tasks = RawDatasetSaver.load_from_jsonl_file(os.path.join(cfg.tasks_path, tasks_file_name))
            task_wrapper = CodeDataset(cfg.original_dataset_name, language=cfg.language)

            # Load the cached scores and approximated tasks
            cached_scores_ids = load_cached_score_ids(results_file_path)
            if cfg.infer_task and cfg.infer_task_cfg.use_cache:
                cached_inferred_tasks_dict = load_infer_task_cache(infer_task_cache_file_path)
            else:
                cached_inferred_tasks_dict = {}

            curr_task_count = 0
            for task in tqdm(tasks):
                if task_wrapper.get_task_id(task) in cached_scores_ids and cfg.use_score_cache:
                    continue

                if detector is None:
                    detector = init_detector(benchmark_cfg)

                # Sample the human-written code (apply augmentations if necessary)
                if cfg.augmentation_cfg.apply_augmentors and cfg.augmentation_cfg.apply_for_human_code:
                    sampled_human_answer = run_augmentors(task_wrapper.get_answers(task)[0], cfg.augmentation_cfg,
                                                          cfg.language)
                else:
                    sampled_human_answer = task_wrapper.get_answers(task)[0]

                # Sample the LLM-generated code (apply augmentations if necessary)
                if cfg.augmentation_cfg.apply_augmentors:
                    aigc_answer = run_augmentors(task['model_response'], cfg.augmentation_cfg, cfg.language)
                else:
                    aigc_answer = task['model_response']

                human_score, aigc_score = np.nan, np.nan
                if cfg.use_original_task:
                    # Compute the scores using the original task (for the experiment in section 4.4)
                    system, instruction_prompt = get_prompts(PromptVariant.REWRITING_METHOD_COT,
                                                             cfg.original_dataset_name)
                    system = system.format(language=cfg.language)
                    question = instruction_prompt.format(problem=task_wrapper.get_question(task), language=cfg.language)

                    messages = HFModel.build_messages_from_task(cfg.detector_cfg.base_model_name, system, question)

                    aigc_response = f"```{cfg.language}\n" + aigc_answer + "\n```"
                    human_response = f"```{cfg.language}\n" + sampled_human_answer + "\n```"
                    if cfg.calculate_aigc_scores and len(aigc_answer) > 0:
                        aigc_score = detector.compute_score(
                            messages + [HFModel.build_message("assistant", aigc_response)])
                    if cfg.calculate_human_scores and len(sampled_human_answer) > 0:
                        human_score = detector.compute_score(
                            messages + [HFModel.build_message("assistant", human_response)])
                else:
                    def score_function(code):
                        if cfg.infer_task:
                            # Compute the score using the conditional probability distribution (using the approximated task)
                            cached_task = cached_inferred_tasks_dict.get(task_wrapper.get_task_id(task), None)
                            score, inferred_task = detector.compute_score_infer_task(code, cached_task)
                            if cfg.infer_task_cfg.use_cache and cached_task is None:
                                RawDatasetSaver.append_to_jsonl_file({task_wrapper.get_task_id(task): inferred_task},
                                                                     infer_task_cache_file_path)
                            return score
                        else:
                            # Compute the score using the *un*conditional probability distribution
                            return detector.compute_score_without_task(code)

                    if cfg.calculate_aigc_scores and len(aigc_answer) > 0:
                        aigc_score = score_function(aigc_answer)
                    if cfg.calculate_human_scores and len(sampled_human_answer) > 0:
                        human_score = score_function(sampled_human_answer)

                json_line = {}
                if cfg.calculate_aigc_scores:
                    json_line['aigc_score'] = aigc_score
                if cfg.calculate_human_scores:
                    json_line['human_score'] = human_score
                results[results_file_path][task_wrapper.get_task_id(task)] = json_line
                RawDatasetSaver.append_to_jsonl_file({task_wrapper.get_task_id(task): json_line}, results_file_path)

                curr_task_count += 1
                if curr_task_count % cfg.memory_clean_interval == 0:
                    gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(0)

            if cfg.calculate_human_scores and not cfg.calculate_aigc_scores:
                # The human-written code is the same for all the task files, so we need to run it only once
                break

    return results


def choose_output_file_paths(cfg, dataset_canonical_name, scores_output_folder_path, tasks_file_name) -> Tuple[
    str, str]:
    """
    Choose the file paths for the approximated tasks cache and the results (scores) file based on the run configuration
    """
    canonical_pattern = cfg.detector_cfg.pattern_weight_mapping_str.replace(':', '_').replace(',', '_')
    if cfg.calculate_human_scores and not cfg.calculate_aigc_scores:
        # Order is important here
        infer_task_cache_file_path = os.path.join(os.path.dirname(scores_output_folder_path),
                                                  f'{dataset_canonical_name}_human_infer_task_cache.jsonl')
        if cfg.detector_cfg.pattern_weight_mapping_str:
            scores_output_folder_path = os.path.join(scores_output_folder_path, canonical_pattern)
            os.makedirs(scores_output_folder_path, exist_ok=True)
        results_file_path = os.path.join(scores_output_folder_path,
                                         f'{dataset_canonical_name}_human_scores.jsonl')

    else:
        infer_task_cache_file_path = os.path.join(os.path.dirname(scores_output_folder_path),
                                                  tasks_file_name.replace('_responses.jsonl',
                                                                          '_infer_task_cache.jsonl'))
        if cfg.detector_cfg.pattern_weight_mapping_str:
            scores_output_folder_path = os.path.join(scores_output_folder_path, canonical_pattern)
            os.makedirs(scores_output_folder_path, exist_ok=True)
        results_file_path = os.path.join(scores_output_folder_path, tasks_file_name.replace('_responses.jsonl',
                                                                                            '_scores.jsonl'))

    return str(infer_task_cache_file_path), str(results_file_path)


def load_cached_score_ids(results_file_path):
    if os.path.exists(results_file_path):
        cached_scores = RawDatasetSaver.load_from_jsonl_file(results_file_path)
        cached_scores_dict = {}
        [cached_scores_dict.update(x) for x in cached_scores]
        cached_scores_ids = [str(x) for x in cached_scores_dict.keys()]
    else:
        cached_scores_ids = []
    return cached_scores_ids


def load_infer_task_cache(cache_file_path: str):
    if os.path.exists(cache_file_path):
        cached_tasks = RawDatasetSaver.load_from_jsonl_file(cache_file_path)
        cached_tasks_dict = {}
        for tasks in cached_tasks:
            for task_id in tasks.keys():
                cached_tasks_dict[str(task_id)] = tasks[task_id]
        return cached_tasks_dict
    else:
        return {}


if __name__ == '__main__':
    benchmark_cfg = pyrallis.parse(config_class=BenchmarkConfig)
    main(benchmark_cfg)
