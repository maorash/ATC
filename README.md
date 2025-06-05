# 🚀 Approximated Task Conditioning (ATC) 🚀
This repository contains the implementation for Approximated Task Conditioning (ATC).

## Setup 🛠️️
- Python 3.8+ (we used Python 3.8.20)
- Install the required packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
  
## Generating Tasks Datasets 📝
To generate the tasks dataset files, run the `code_datasets/generation_pipeline.py` script (see `code_datasets/README.md`). 

This script will create the necessary tasks to run the detection methods.

Each generated task file is a `.jsonl` file, containing the original fields in the dataset, alongside two additional fields `raw_model_response` and `model_response`.
The `raw_model_response` field contains the raw model response, while the `model_response` field contains the sanitized model response.

For the MBPP dataset, this is an example entry in the tasks `.jsonl` file:
```json
{
   "task_id":19,
   "text":"Write a function to find whether a given array of integers contains any duplicate element.",
   "code":"def test_duplicate(arraynums):\r\n    nums_set = set(arraynums)    \r\n    return len(arraynums) != len(nums_set)     ",
   "test_list":[
      "assert test_duplicate(([1,2,3,4,5]))==False",
      "assert test_duplicate(([1,2,3,4, 4]))==True",
      "assert test_duplicate([1,1,2,2,3,3,4,4,5])==True"
   ],
   "test_setup_code":"",
   "challenge_test_list":[
      
   ],
   "raw_model_response":"  ```\ndef contains_duplicate(nums):\n    return len(nums)!= len(set(nums))\n```\nExplanation:\n\nThe `set()` function creates a set from an iterable, which in this case is the `nums` array. The `len()` function returns the length of the set, which is the number of unique elements in the array.\n\nIf the array contains any duplicate elements, the length of the set will be less than the length of the array. In this case, the `len(nums)!= len(set(nums))` condition will be true, indicating that the array contains a duplicate element.\n\nThe function returns `True` if the condition is met, and `False` otherwise.\n\nFor example, if we call the function with the argument `nums = [1, 2, 3, 4, 5]`, it will return `False`, because the array does not contain any duplicate elements.\n\nIf we call the function with the argument `nums = [1, 2, 3, 4, 5, 5]`, it will return `True`, because the array contains a duplicate element (the number 5).",
   "model_response":"def contains_duplicate(nums):\n    return len(nums)!= len(set(nums))"
}
```

## Running the Detection Methods 🏃‍
The main script for running the detection methods is `detection/run.py`.
You can configure the detection method and other parameters using command-line arguments.

The main arguments for the `run.py` script:

- `--detector_cfg.base_model_name` (str): The Detector-LLM used for probability distribution estimation (and for task approximation in ATC).
- `--detector_cfg.method` (str): Detection method to use (one of the following: `entropy`, `mean_log_likelihood`, `log_rank`, `lrr`, `detectgpt`, `npr`, `supervised`).
- `--detector_cfg.pattern_weight_mapping_str` (str): Pattern weight mapping string (use 'comments:0,docstrings:0' to exclude comment tokens in the score computation, as done in ATC).


- `--infer_task` (bool): Whether to approximate the task during detection (used for ATC).
- `--use_original_task` (bool): Whether to use the original task for detection.

If both `infer_task` and `use_original_task` are set to `False`, the detector will sample from the unconditional probability distribution.

- `--calculate_human_scores` (bool): Whether to calculate scores for human-written code.
- `--calculate_aigc_scores` (bool): Whether to calculate scores for LLM-generated code.

Set one of the two arguments above to 'True' and the other to 'False' to calculate scores for either human-written or LLM-generated code.

- `--tasks_path` (str): Path to a directory containing the tasks files generated using `code_datasets/generation_pipeline.py`.
- `--scores_output_path` (str): Path to save the output scores.
- `--original_dataset_name` (str): Name of the original dataset.

### Example Command-Line Configurations 💻
In these examples we assume as a preqrequisite that MBPP task files were generated using `code_datasets/generation_pipeline.py` (and saved under `results/google-research-datasets_mbpp`).

You must run `run.py` twice: once for human-written code and once for LLM-generated code. 

Use the `--calculate_human_scores` and `--calculate_aigc_scores` flags to specify which type of code to process.

#### Approximated Task Conditioning (ATC) 🔍
  ```bash
  # Calculate human-written code scores
  python run.py  --calculate_human_scores True --calculate_aigc_scores False --use_original_task False --infer_task True --scores_output_path results/scores_atc --original_dataset_name google-research-datasets/mbpp --tasks_path results/google-research-datasets_mbpp --detector_cfg.method entropy
  # Calculate LLM-generated code scores
  python run.py  --calculate_human_scores False --calculate_aigc_scores True --use_original_task False --infer_task True --scores_output_path results/scores_atc --original_dataset_name google-research-datasets/mbpp --tasks_path results/google-research-datasets_mbpp --detector_cfg.method entropy
  ```

#### Without Task - Sampling from the unconditional probability distribution  🎲
  ```bash
  # In this example we use the entropy detection method
  
  # Calculate human-written code scores
  python run.py  --calculate_human_scores True --calculate_aigc_scores False --use_original_task False --infer_task False --scores_output_path results/scores_unconditional --original_dataset_name google-research-datasets/mbpp --tasks_path results/google-research-datasets_mbpp --detector_cfg.method entropy
  # Calculate LLM-generated code scores
  python run.py  --calculate_human_scores False --calculate_aigc_scores True --use_original_task False --infer_task False --scores_output_path results/scores_unconditional --original_dataset_name google-research-datasets/mbpp --tasks_path results/google-research-datasets_mbpp --detector_cfg.method entropy
  ```

#### Using the Original Task - Sampling from the conditional probability distribution 📊
  ```bash
  # In this example we use the entropy detection method
  
  # Calculate human-written code scores
  python run.py  --calculate_human_scores True --calculate_aigc_scores False --use_original_task True --infer_task False --scores_output_path results/scores_conditional --original_dataset_name google-research-datasets/mbpp --tasks_path results/google-research-datasets_mbpp --detector_cfg.method entropy
  # Calculate LLM-generated code scores
  python run.py  --calculate_human_scores False --calculate_aigc_scores True --use_original_task True --infer_task False --scores_output_path results/scores_conditional --original_dataset_name google-research-datasets/mbpp --tasks_path results/google-research-datasets_mbpp --detector_cfg.method entropy
  ```


### Notes Regarding Additional Experiments 🧪
* To recreate the ablation experiment `Score Computation with Comment Tokens`, set the `pattern_weight_mapping_str` to an empty string, or equivalently, to `comments:1,docstrings:1`
* To recreate the experiment in section 4.3 (`Robustness to Comment Removal`), use `--augmentation_cfg.apply_augmentors True` and `--augmentation_cfg.methods ['remove_comments']`.
* To recreate the experiment in section 4.5 (`Exploring Different Task Approximation Prompts`), use `--infer_task_cfg.prompt_style` and set it to one of the following: `'regular', 'short', 'long', 'friendly', 'critical', 'pseudocode', 'storytelling'`.
* To run an experiment in a different programming language (`cpp`, `java`), use the `--language` argument.


## Output Format 📂
The scores are saved in the specified `scores_output_path`. 
Each score file is saved as a `.jsonl` file, where each line contains the `task_id` and corresponding score.

Note that the final output folder is named based on additional configuration parameters, including the augmentations, token weight mapping and the detector method.

For example, when running ATC (using `--infer_task True` and `--detector_cfg.method entropy`) and setting `--scores_output_path results/scores_atc` , the folder structure will look like:
```
results/
└── scores_atc/
    └── entropy/
        └── comments_0_docstrings_0/
            ├── <dataset_prefix>_<generator_model>_scores.jsonl
            └── <dataset_prefix>_human_scores.jsonl
```

## Analyzing the Scores 📈
To process the `_scores.jsonl` files, use the `analyze_scores.ipynb` notebook. This notebook contains examples on how to calculate AUROC and generate ROC plots.

To demonstrate how scores are analyzed, we add a few score files generated in our MBPP experiments.
