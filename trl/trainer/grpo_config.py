from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class GRPOConfig(TrainingArguments):
    """
    Configuration class for GRPO training.
    """
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use VLLM for inference"}
    )
    num_generations: int = field(
        default=2,
        metadata={"help": "Number of generations per prompt"}
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "Maximum length of the prompt"}
    )
    max_completion_length: int = field(
        default=300,
        metadata={"help": "Maximum length of the completion"}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.2,
        metadata={"help": "GPU memory utilization for VLLM"}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam optimizer"}
    )
    adam_beta2: float = field(
        default=0.99,
        metadata={"help": "Beta2 for Adam optimizer"}
    )
    weight_decay: float = field(
        default=0.1,
        metadata={"help": "Weight decay"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    ) 
