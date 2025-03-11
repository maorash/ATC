import argparse
import gc
import os.path

import torch
import tqdm

from code_datasets.response_generator import HFModel, GPT, Claude3Haiku, ResponseSanitizer
from code_datasets.loader import DatasetLoader, DatasetSanitizer, RawDatasetSaver
from code_datasets.prompts import get_prompts, PromptVariant


class CodeGenerationPipeline:
    def __init__(self, dataset_name: str, model_name: str, prompts_variant: PromptVariant, generation_config):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.prompts_variant = prompts_variant
        self.generation_config = generation_config

        if 'gpt' in self.model_name:
            self.llm = GPT(**generation_config, model_name=self.model_name)
        elif self.model_name == "claude3-haiku":
            self.llm = Claude3Haiku(**generation_config)
        else:
            self.llm = HFModel(model_name=self.model_name, **generation_config)

    def run(self):
        dataset_canonical_name = f"{self.dataset_name.replace('/', '_')}"
        model_canonical_name = self.model_name.replace('/', '_')
        os.makedirs(f"results/{dataset_canonical_name}", exist_ok=True)
        loader = DatasetLoader(self.dataset_name)
        loader.load_dataset()
        dataset = loader.get_test_set()

        sanitizer = DatasetSanitizer(dataset)
        sanitized_dataset = sanitizer.sanitize()

        RawDatasetSaver.save_to_json_file(sanitized_dataset,
                                          f"results/{dataset_canonical_name}/{dataset_canonical_name}_sanitized.json")

        results_filename = f"results/{dataset_canonical_name}/{dataset_canonical_name}_{model_canonical_name}_responses.jsonl"

        # Load previously processed tasks if the file exists
        processed_task_ids = set()
        if os.path.exists(results_filename):
            processed_task_ids = set([dataset.get_task_id(sample) for sample in
                                      RawDatasetSaver.load_from_jsonl_file(results_filename)])

        system_prompt, instruction_prompt = get_prompts(self.prompts_variant, self.dataset_name)
        system_prompt = system_prompt.format(language=loader.language)

        for sample in tqdm.tqdm(sanitized_dataset):
            if dataset.get_task_id(sample) in processed_task_ids:
                continue

            question = instruction_prompt.format(problem=dataset.get_question(sample), language=loader.language)

            response = ''
            tries = 0
            while len(response) < 1 and tries < 100:
                raw_response = self.llm.generate_code(system_prompt, question)

                sample['raw_model_response'] = raw_response

                response = ResponseSanitizer.sanitize_response(raw_response, self.model_name, language=loader.language)
                tries += 1
                if tries > 1:
                    print(f"Retrying to generate a response for task: {dataset.get_task_id(sample)}")

            if len(response) < 1:
                print(f"Failed to generate a response for task: {dataset.get_task_id(sample)}")
                continue

            sample['model_response'] = response

            RawDatasetSaver.append_to_jsonl_file(sample, results_filename)
            processed_task_ids.add(dataset.get_task_id(sample))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["google-research-datasets/mbpp"],
        choices=[
            "codeparrot/apps",
            "google-research-datasets/mbpp",
            "deepmind/code_contests_java",
            "deepmind/code_contests_cpp",
        ],
        help="List of datasets to use."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["codellama/CodeLlama-7b-Instruct-hf"],
        choices=[
            "HuggingFaceH4/starchat-alpha",
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-7b-Instruct-hf",
            "google/codegemma-7b-it",
            "claude3-haiku",
        ],
        help="List of models to use."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature setting for generation."
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        for model in args.models:
            pipeline = CodeGenerationPipeline(dataset, model,
                                              PromptVariant.REWRITING_METHOD_COT,
                                              generation_config={
                                                  "temperature": args.temperature,
                                                  "top_p": 0.95,
                                                  "torch_dtype": torch.float16
                                              })
            pipeline.run()
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
