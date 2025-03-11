import json
from typing import Optional

import datasets


class CodeDataset:
    def __init__(self, dataset_name: str, dataset: Optional[datasets.Dataset] = None, language: str = "python"):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.language = language

    def get_question(self, sample: dict):
        if self.dataset_name == "codeparrot/apps":
            return sample['question']
        elif self.dataset_name == "deepmind/code_contests":
            return sample['description']
        else:
            return sample['text']

    def get_answers(self, sample: dict):
        if self.dataset_name == "codeparrot/apps":
            solutions = sample['solutions']
            if len(solutions.strip()) == 0:
                return None
            else:
                return json.loads(solutions)
        elif self.dataset_name == "deepmind/code_contests":
            solutions = sample['solutions']
            LANGUAGE_MAP = {
                1: "python",
                2: "cpp",
                3: "python",
                4: "java",
            }
            solutions_in_language = [solutions['solution'][i] for i in range(len(solutions['language'])) if LANGUAGE_MAP[solutions['language'][i]] == self.language.lower()]
            return solutions_in_language
        else:
            return [sample['code']]

    def get_task_id(self, sample: dict):
        if self.dataset_name == "codeparrot/apps":
            return str(sample['problem_id'])
        elif self.dataset_name == "deepmind/code_contests":
            return f'{sample["source"]}_{sample["name"]}'
        else:
            return str(sample['task_id'])

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying dataset.
        """
        if name in self.__dict__:
            # Avoid infinite recursion by checking the wrapper's own attributes
            return self.__dict__[name]
        elif self.dataset is None:
            raise ValueError("Uninitialized dataset")
        return getattr(self.dataset, name)

    def __len__(self):
        """
        Delegate length to the underlying dataset.
        """
        if self.dataset is None:
            raise ValueError("Uninitialized dataset")
        return len(self.dataset)

    def __iter__(self):
        """
        Delegate iteration to the underlying dataset.
        """
        if self.dataset is None:
            raise ValueError("Uninitialized dataset")
        return iter(self.dataset)


class DatasetLoader:
    def __init__(self, dataset_name: str):
        if "deepmind/code_contests" in dataset_name:
            self.language = dataset_name.split('_')[-1]
            self.dataset_name = "deepmind/code_contests"
        else:
            self.language = "python"
            self.dataset_name = dataset_name
        self.dataset = None
        assert self.dataset_name in ["codeparrot/apps", "google-research-datasets/mbpp", "deepmind/code_contests"]

    def load_dataset(self):
        self.dataset = datasets.load_dataset(self.dataset_name)

    def get_test_set(self) -> CodeDataset:
        return CodeDataset(self.dataset_name, self.dataset['test'], self.language)


class DatasetSanitizer:
    def __init__(self, dataset: CodeDataset):
        self.dataset = dataset

    def sanitize(self):
        sanitized_data = []
        for item in self.dataset:
            if self._validate_item(item):
                sanitized_data.append(self._transform_item(item))
            # else:
            #     print(f"Item {self.dataset.get_task_id(item)} isn't valid")
        return sanitized_data

    def _validate_item(self, item):
        # no_image_markdown = '[Image]' not in self.dataset.get_question(item)
        no_link_markdown = '](http' not in self.dataset.get_question(item)
        answers = self.dataset.get_answers(item)
        answers_exist = answers is not None and len(answers) > 0
        return no_link_markdown and answers_exist

    def _transform_item(self, item):
        # Add transformations if needed
        return item


class RawDatasetSaver:
    @staticmethod
    def save_to_json_file(data: dict, filename: str):
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def append_to_jsonl_file(data: dict, filename: str):
        with open(filename, 'a') as f:
            f.write(json.dumps(data) + "\n")

    @staticmethod
    def load_from_jsonl_file(filename: str):
        tasks = []
        with open(filename, 'r') as f:
            for line in f:
                tasks.append(json.loads(line))
        return tasks
