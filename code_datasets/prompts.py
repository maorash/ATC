from enum import Enum
from typing import Tuple


class PromptVariant(Enum):
    REWRITING_METHOD_COT = 0
    REWRITING_METHOD_DIRECT = 1


class RewritingMethodConstants:
    """
    These prompts are taken from https://arxiv.org/abs/2405.16133
    """
    SYSTEM_PROMPT_WITH_IO = """You serve as a programming assistant. I will first give you a programming challenge. The challenge contains Problem Description, Input and Ouput specifications. (Note that it is in Markdown format, and the math formula is inline latex).
You need to provide a {language} solution for the challenge."""

    SYSTEM_PROMPT_WITHOUT_IO = """You serve as a programming assistant. I will first give you a programming challenge (Note that it is in Markdown format, and the math formula is inline latex).
You need to provide a {language} solution for the challenge."""

    INSTRUCTION_COT = """{problem}

Let's solve the problem step by step. You can first try to understand and analyze the problem. Then provide a {language} solution for the coding challenge above.
The {language} code should be organized in a single markdown block. Please do not add extra explanation for the code."""

    INSTRUCTION_DIRECT = """{problem}

Please provide a {language} solution for the coding challenge above. The {language} code should be organized in a single markdown block. Please do not add extra explanation for the code.
"""


def get_prompts(prompt_variant: PromptVariant, dataset_name: str) -> Tuple[str, str]:
    """
    Returns a tuple of system prompt and instruction prompt
    """
    if prompt_variant in [PromptVariant.REWRITING_METHOD_COT, PromptVariant.REWRITING_METHOD_DIRECT]:
        if dataset_name == "codeparrot/apps" or "code_contests" in dataset_name:
            system = RewritingMethodConstants.SYSTEM_PROMPT_WITH_IO
        elif dataset_name == "google-research-datasets/mbpp":
            system = RewritingMethodConstants.SYSTEM_PROMPT_WITHOUT_IO
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        if prompt_variant == PromptVariant.REWRITING_METHOD_COT:
            return system, RewritingMethodConstants.INSTRUCTION_COT
        else:
            return system, RewritingMethodConstants.INSTRUCTION_DIRECT
    else:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant.name}")
