# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset, Dataset
import argparse
from dataclasses import dataclass, field
from typing import Optional, Any, List, Callable
import datetime
import re
import json
from pathlib import Path
from PIL import Image
import deepspeed
import numpy as np
from torch.utils.data import Dataset

import ast
import copy
import logging
import pathlib
import time
import random
import math
import yaml
from packaging import version
import tokenizers
import torch.nn.functional as F

from llava.train.llava_grpo_trainer import LLaVAGRPOTrainer, LLaVAGRPOTrainerWithEWC
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaConfig
from llava.train.train import (
    ModelArguments, DataArguments, TrainingArguments, 
    smart_tokenizer_and_embedding_resize, _tokenize_fn, 
    preprocess_multimodal, preprocess, LazySupervisedDataset, DataCollatorForSupervisedDataset,
    make_supervised_data_module, get_model, maybe_zero_3,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3,
    get_mm_adapter_state_maybe_zero_3, find_all_linear_names, safe_save_model_for_hf_trainer
)
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.utils import rank0_print

from transformers import AutoProcessor
from trl import GRPOConfig
from trl.import_utils import is_xpu_available

torch.multiprocessing.set_sharing_strategy("file_system")
local_rank = None

# Define a local version of init_distributed_mode
def init_distributed_mode():
    """Initialize distributed training if available"""
    global local_rank
    
    # Check if process group is already initialized
    if dist.is_initialized():
        rank0_print("Process group already initialized, skipping initialization")
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        return
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        rank0_print(f"Initializing distributed training with rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set the device to the local rank's GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank0_print("Distributed training not initialized (RANK and WORLD_SIZE environment variables not set)")

# System prompt and XML format for responses
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

@dataclass
class GRPOTrainingArguments(TrainingArguments):
    """Configuration for GRPO training."""
    # Custom parameters for GRPO
    use_vllm: bool = field(default=False, metadata={"help": "Whether to use VLLM for faster generation."})
    
    # Parameters for GRPO with grpo_ prefix
    grpo_num_generations: int = field(default=2, metadata={"help": "Number of generations for GRPO."})
    grpo_max_prompt_length: int = field(default=256, metadata={"help": "Maximum prompt length for GRPO."})
    grpo_max_completion_length: int = field(default=300, metadata={"help": "Maximum completion length for GRPO."})
    grpo_learning_rate: float = field(default=5e-6, metadata={"help": "Learning rate for GRPO."})
    grpo_weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for GRPO."})
    grpo_warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio for GRPO."})
    grpo_lr_scheduler_type: str = field(default="cosine", metadata={"help": "LR scheduler type for GRPO."})
    grpo_max_grad_norm: float = field(default=0.1, metadata={"help": "Maximum gradient norm for GRPO."})
    grpo_report_to: str = field(default="tensorboard", metadata={"help": "Where to report GRPO training logs."})
    grpo_ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "Whether to find unused parameters in DDP."})
    
    # Keep the same parameters without prefix for backward compatibility
    num_generations: int = field(default=2, metadata={"help": "Number of generations for GRPO."})
    max_prompt_length: int = field(default=256, metadata={"help": "Maximum prompt length."})
    max_completion_length: int = field(default=300, metadata={"help": "Maximum completion length."})
    reward_model_path: Optional[str] = field(default=None, metadata={"help": "Path to a reward model. If not specified, will use the built-in reward functions."})
    grpo_beta: float = field(default=0.1, metadata={"help": "Beta parameter for GRPO, controls the impact of reward difference."})
    reward_scaling: float = field(default=1.0, metadata={"help": "Scaling factor for rewards."})
    
    # Standard GRPO parameters that match TRL (with our defaults)
    beta: float = field(default=0.1, metadata={"help": "Beta parameter for GRPO, used by TRL library. Will be set to grpo_beta value."})
    
    # Add tf32 parameter (similar to how it's defined in TrainingArguments)
    tf32: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    
    def __post_init__(self):
        """Set beta to match grpo_beta for compatibility with TRL"""
        super().__post_init__()
        self.beta = self.grpo_beta
        
        # Set the non-prefixed parameters to match the prefixed ones if they were provided
        self.num_generations = self.grpo_num_generations
        self.max_prompt_length = self.grpo_max_prompt_length
        self.max_completion_length = self.grpo_max_completion_length

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML formatted text."""
    match = re.search('<answer>(.*)</answer>', text, re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = ''
    return answer.strip()

def extract_xml_reasoning(text: str) -> str:
    """Extract reasoning from XML formatted text."""
    match = re.search('<reasoning>(.*)</reasoning>', text, re.DOTALL)
    if match:
        reasoning = match.group(1)
    else:
        reasoning = ''
    return reasoning.strip()

def add_xml_format_to_prompt(prompt: str) -> str:
    """Add XML format instruction to the prompt."""
    if not prompt.strip().startswith(SYSTEM_PROMPT.strip()):
        return SYSTEM_PROMPT + "\n\n" + prompt
    return prompt

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GRPOTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    if training_args.local_rank == 0:
        transformers.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()

    # 优化输出
    if hasattr(training_args, "log_level"):
        if training_args.log_level == "none":
            transformers.logging.set_verbosity_error()
            logging.basicConfig(level=logging.ERROR)
            rank0_print("设置日志级别为ERROR，减少输出量")

    # Log on each process the small summary
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", distributed training: {bool(training_args.local_rank != -1)}"
        + f", 16-bits training: {training_args.fp16}"
    )
    
    # 清理内存并设置运行配置
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            rank0_print("已清理GPU缓存")
    except Exception as e:
        rank0_print(f"清理缓存时出错: {e}")
    
    # 优化CUDA设置，提高性能
    if torch.cuda.is_available():
        # 设置CUDA分配器
        try:
            # Use cudnn benchmark for better performance
            torch.backends.cudnn.benchmark = True
            rank0_print("启用cudnn benchmark优化")
            
            # 使用TF32，对于Ampere及更新的GPU
            if hasattr(training_args, "tf32") and training_args.tf32:
                if torch.cuda.get_device_capability()[0] >= 8:  # Ampere及更新
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    rank0_print("启用TF32加速计算")
        except Exception as e:
            rank0_print(f"设置CUDA优化时出错: {e}")
    
    # 检查gradient_checkpointing和use_cache设置的兼容性
    if hasattr(training_args, "gradient_checkpointing") and training_args.gradient_checkpointing:
        # 始终禁用use_cache以兼容gradient_checkpointing
        model_args.use_cache = False
    
    # Initialize distributed training
    init_distributed_mode()
    
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        try:
            # Look for checkpoint in output dir
            last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        except Exception as e:
            logger.warning(f"Error checking for checkpoint: {e}")

    # Load model and tokenizer
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        
        # Ensure bf16_dtype attribute exists
        bf16_dtype = getattr(training_args, "bf16_dtype", "bfloat16")
        
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=eval("torch." + bf16_dtype),
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    # 处理中断恢复
    is_resume = last_checkpoint is not None or training_args.resume_from_checkpoint is not None
    
    # 添加模型预加载时的内存优化
    rank0_print("开始加载模型，优化内存使用...")
    torch.cuda.empty_cache()
    gc.collect()
        
    # Set up model
    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    
    # 确保在启用gradient checkpointing时禁用use_cache
    if hasattr(training_args, "gradient_checkpointing") and training_args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
    
    # Create the processor for image processing
    processor = None
    if model_args.vision_tower is not None:
        from transformers import CLIPImageProcessor
        
        # 尝试获取处理器
        try:
            # 从模型获取处理器
            if hasattr(model, "get_image_processor") and callable(model.get_image_processor):
                processor = model.get_image_processor()
                rank0_print(f"从模型获取图像处理器: {type(processor)}")
            else:
                # 基于vision tower加载处理器
                processor = CLIPImageProcessor.from_pretrained(model_args.vision_tower)
                rank0_print(f"从vision tower加载CLIPImageProcessor")
        except Exception as e:
            rank0_print(f"加载处理器时出错: {e}")
            # 使用基本处理器
            from transformers import ProcessorMixin
            class MinimalProcessor(ProcessorMixin):
                def __init__(self):
                    self.image_mean = [0.48145466, 0.4578275, 0.40821073]
                    self.image_std = [0.26862954, 0.26130258, 0.27577711]
                    self.size = {"height": 224, "width": 224, "shortest_edge": 224}
                
                def __call__(self, images, **kwargs):
                    import torch
                    import numpy as np
                    from PIL import Image
                    
                    if not isinstance(images, list):
                        images = [images]
                    
                    processed = []
                    for image in images:
                        if isinstance(image, torch.Tensor):
                            # Already a tensor
                            if image.dim() == 3 and image.shape[0] == 3:
                                processed.append(image)
                            else:
                                # Create placeholder
                                processed.append(torch.zeros(3, 224, 224))
                        elif isinstance(image, Image.Image):
                            # Convert PIL Image
                            image = image.convert("RGB").resize((224, 224))
                            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                            processed.append(image)
                        else:
                            processed.append(torch.zeros(3, 224, 224))
                    
                    return {"pixel_values": torch.stack(processed) if len(processed) > 1 else processed[0]}
            
            processor = MinimalProcessor()
            rank0_print("使用最小化处理器作为后备")
        
        # 识别SigLIP处理器并进行特殊处理
        is_siglip = "siglip" in str(type(processor)).lower()
        if is_siglip:
            rank0_print("检测到SigLIP图像处理器，设置图像大小为384x384")
            processor.size = {"height": 384, "width": 384, "shortest_edge": 384}
            
            # 确保SigLIP位置嵌入设置正确
            if hasattr(model, "get_vision_tower") and callable(model.get_vision_tower):
                vision_tower = model.get_vision_tower()
                if vision_tower is not None and hasattr(vision_tower, "vision_tower"):
                    try:
                        # 检查并修正位置嵌入大小
                        if hasattr(vision_tower.vision_tower, "embeddings") and hasattr(vision_tower.vision_tower.embeddings, "position_embedding"):
                            position_embedding = vision_tower.vision_tower.embeddings.position_embedding
                            if position_embedding.weight.shape[0] != 729:
                                rank0_print(f"警告: SigLIP位置嵌入大小 {position_embedding.weight.shape[0]} 与预期的729不匹配")
                    except Exception as e:
                        rank0_print(f"检查SigLIP位置嵌入时出错: {e}")
        
        # 使用包装器确保数据类型一致性
        processor = ProcessorWrapper(processor, model=model)
        print(f"使用ProcessorWrapper包装处理器以确保数据类型一致性")
    
        # 设置数据参数上的处理器用于数据集预处理
        data_args.image_processor = processor
    
    # 确保模型使用正确的处理器
    if hasattr(model, "config") and hasattr(model.config, "image_processor"):
        model.config.image_processor = processor
        print("设置model.config.image_processor为包装后的处理器")
    
    # Get tokenizer 
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        rank0_print("设置pad_token为unk_token")
    
    # 添加特殊token
    if model_args.version in {"v0", "v0-mmtag", "v1", "llama_v2"}:
        if DEFAULT_IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': [DEFAULT_IMAGE_TOKEN]})
            rank0_print(f"添加图像token: {DEFAULT_IMAGE_TOKEN}")
        
        if DEFAULT_IM_START_TOKEN not in tokenizer.get_vocab() or DEFAULT_IM_END_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]})
            rank0_print(f"添加图像边界token: {DEFAULT_IM_START_TOKEN}, {DEFAULT_IM_END_TOKEN}")
        
        # Resize token embeddings
        model.resize_token_embeddings(len(tokenizer))
    
    # 在加载数据集前再次清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Load dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 定义奖励函数
    def format_reward_func(trainer, completions=None, **kwargs):
        """检查响应是否遵循所需的XML格式"""
        try:
            if completions is None or len(completions) == 0:
                return [0.0]  # 默认奖励
                
            # 安全调用静态方法
            return LLaVAGRPOTrainer.format_reward_func(trainer, completions=completions, **kwargs)
        except Exception as e:
            rank0_print(f"format_reward_func出错: {e}")
            # 返回默认奖励
            return [0.0] if completions is None else [0.0] * len(completions)
    
    def reasoning_reward_func(trainer, completions=None, **kwargs):
        """评估推理质量"""
        try:
            if completions is None or len(completions) == 0:
                return [0.0]  # 默认奖励
                
            # 安全调用静态方法
            return LLaVAGRPOTrainer.reasoning_quality_reward_func(trainer, completions=completions, **kwargs)
        except Exception as e:
            rank0_print(f"reasoning_reward_func出错: {e}")
            # 返回默认奖励
            return [0.0] if completions is None else [0.0] * len(completions)
    
    def correctness_reward_func(trainer, prompts=None, completions=None, answer=None, **kwargs):
        """检查答案与地面真实情况的正确性"""
        try:
            if completions is None or len(completions) == 0:
                return [0.0]  # 默认奖励
                
            # 如果prompts为空，创建一个默认列表
            if prompts is None:
                prompts = [""] * len(completions)
                
            # 如果answer为空，创建一个默认列表
            if answer is None:
                answer = [None] * len(completions)
                
            # 安全调用静态方法
            return LLaVAGRPOTrainer.answer_correctness_reward_func(
                trainer, prompts=prompts, completions=completions, answer=answer, **kwargs
            )
        except Exception as e:
            rank0_print(f"correctness_reward_func出错: {e}")
            # 返回默认奖励
            return [0.0] if completions is None else [0.0] * len(completions)
    
    # 选择正确的trainer类
    if hasattr(training_args, "use_ewc") and training_args.use_ewc:
        rank0_print("使用弹性权重整合(EWC)训练器")
        trainer_class = LLaVAGRPOTrainerWithEWC
    else:
        trainer_class = LLaVAGRPOTrainer
    
    # 定义奖励函数列表
    reward_functions = [format_reward_func, reasoning_reward_func, correctness_reward_func]
    rank0_print(f"设置{len(reward_functions)}个奖励函数用于GRPO训练")
    
    # 添加更多的trainer参数以确保数据类型一致性
    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "processor": processor,
        "args": training_args,
        "train_dataset": data_module["train_dataset"] if data_module["train_dataset"] else None,
        "eval_dataset": data_module["eval_dataset"] if data_module["eval_dataset"] else None,
        "reward_funcs": reward_functions,
        "data_collator": data_module["data_collator"],
    }
    
    # 移除处理器配置的单独参数，确保processor本身已经有正确的数据类型设置
    if hasattr(model, "dtype"):
        model_dtype = model.dtype
        print(f"模型数据类型: {model_dtype}")
    else:
        # 尝试从模型参数获取dtype
        for param in model.parameters():
            model_dtype = param.dtype
            print(f"从模型参数获取dtype: {param.dtype}")
            break
    
    # 确保处理器设置正确的数据类型，但不传递额外参数给训练器
    if hasattr(processor, "_model_dtype") and model_dtype is not None:
        processor._model_dtype = model_dtype
        print(f"已将处理器的数据类型设置为: {model_dtype}")
    
    # 创建训练器
    trainer = trainer_class(**trainer_kwargs)
    
    # 在训练前再次检查gradient_checkpointing和use_cache的兼容性
    if hasattr(trainer.model, "gradient_checkpointing") and trainer.model.gradient_checkpointing:
        if hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = False
    
    # 对量化模型添加梯度钩子
    if training_args.bits in [4, 8]:
        from peft.utils import _get_submodules
        
        # 查找嵌入层
        for name, module in trainer.model.named_modules():
            if "embed_tokens" in name:
                # 添加钩子使输入需要梯度
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                
                module.register_forward_hook(make_inputs_require_grad)
    
    # 启用TF32加速大型矩阵乘法(如果支持)
    if hasattr(training_args, "tf32") and training_args.tf32 and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere及更新
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    
    # 开始训练
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存模型
    trainer.save_model()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 安全保存，适用于任何量化模型或DeepSpeed
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    return

class ProcessorWrapper:
    """包装图像处理器，确保处理图像时维持数据类型一致性"""
    
    def __init__(self, processor, model=None):
        self.processor = processor
        self.model = model
        self._model_dtype = None
        
        # 识别处理器类型
        self.is_siglip = False
        self.is_qwen = False
        
        # 识别模型类型
        self.is_qwen_model = False
        if model is not None:
            model_name = str(type(model)).lower()
            if "qwen" in model_name or (hasattr(model, "config") and hasattr(model.config, "model_type") and "qwen" in model.config.model_type.lower()):
                self.is_qwen_model = True
                print("检测到Qwen模型，将使用特殊的输入处理逻辑")
        
        if hasattr(processor, "__class__") and hasattr(processor.__class__, "__name__"):
            processor_name = processor.__class__.__name__.lower()
            if "siglip" in processor_name:
                self.is_siglip = True
                print(f"检测到SigLIP处理器: {processor_name}")
            elif "qwen" in processor_name:
                self.is_qwen = True
                print(f"检测到Qwen处理器: {processor_name}")
        
        # 设置正确的图像大小
        if self.is_siglip:
            self.image_size = 384
            print(f"SigLIP处理器: 设置图像大小为{self.image_size}x{self.image_size}")
        else:
            self.image_size = 448
            print(f"非SigLIP处理器: 设置图像大小为{self.image_size}x{self.image_size}")
            
        # 设置处理器的size属性，以便正确调整输入图像
        if hasattr(self.processor, "size"):
            if isinstance(self.processor.size, dict):
                self.processor.size["height"] = self.image_size
                self.processor.size["width"] = self.image_size
                if "shortest_edge" in self.processor.size:
                    self.processor.size["shortest_edge"] = self.image_size
            elif isinstance(self.processor.size, (list, tuple)):
                self.processor.size = (self.image_size, self.image_size)
    
    @property
    def model_dtype(self):
        """获取模型的数据类型"""
        if self._model_dtype is None and self.model is not None:
            # 获取模型的dtype
            try:
                if hasattr(self.model, "dtype"):
                    self._model_dtype = self.model.dtype
                    print(f"直接从模型获取数据类型: {self._model_dtype}")
                else:
                    for param in self.model.parameters():
                        self._model_dtype = param.dtype
                        break
                    print(f"从模型参数获取数据类型: {self._model_dtype}")
            except Exception as e:
                print(f"无法确定模型数据类型: {e}")
                self._model_dtype = torch.float32
        return self._model_dtype or torch.float32
    
    def ensure_correct_size_and_dtype(self, pixel_values):
        """确保图像具有正确的大小和数据类型"""
        import torch.nn.functional as F
        
        # 如果不是张量，无法处理
        if not torch.is_tensor(pixel_values):
            return pixel_values
            
        # 确保与模型数据类型一致
        if pixel_values.dtype != self.model_dtype:
            pixel_values = pixel_values.to(dtype=self.model_dtype)
            
        # 处理单张或批量图像
        if pixel_values.dim() == 3:  # 单张图像 [C, H, W]
            # 检查通道顺序
            if pixel_values.shape[0] != 3 and pixel_values.shape[-1] == 3:
                pixel_values = pixel_values.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                
            # 调整大小
            h, w = pixel_values.shape[1:]
            if h != self.image_size or w != self.image_size:
                pixel_values = F.interpolate(
                    pixel_values.unsqueeze(0),  # 添加批次维度 [1, C, H, W]
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)  # 移除批次维度 [C, H, W]
        
        elif pixel_values.dim() == 4:  # 批量图像 [B, C, H, W]
            # 检查通道顺序
            if pixel_values.shape[1] != 3 and pixel_values.shape[-1] == 3:
                pixel_values = pixel_values.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                
            # 调整大小
            h, w = pixel_values.shape[2:]
            if h != self.image_size or w != self.image_size:
                pixel_values = F.interpolate(
                    pixel_values,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False
                )
                
        return pixel_values
    
    def __call__(self, images, **kwargs):
        """确保处理后的图像具有正确的数据类型和大小"""
        result = self.processor(images, **kwargs)
        
        # 确保pixel_values有正确的数据类型和大小
        if hasattr(result, "pixel_values") and torch.is_tensor(result.pixel_values):
            result.pixel_values = self.ensure_correct_size_and_dtype(result.pixel_values)
            print(f"处理后图像: 形状={result.pixel_values.shape}, dtype={result.pixel_values.dtype}")
            
            # 对于Qwen模型，确保设置"images"属性
            if self.is_qwen_model or self.is_qwen:
                result.images = result.pixel_values
                print("为Qwen模型添加images属性")
        
        return result
        
    def __getattr__(self, name):
        """委托其他属性和方法到原始处理器"""
        if hasattr(self.processor, name):
            return getattr(self.processor, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

if __name__ == "__main__":
    train() 
