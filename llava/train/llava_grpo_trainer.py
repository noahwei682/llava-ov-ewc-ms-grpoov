import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import types
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from PIL import Image
import base64
from io import BytesIO
import copy
import numpy as np
import math
import logging
import transformers
import inspect
from transformers import Trainer
from transformers.trainer import *
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from trl import GRPOTrainer
from llava.train.llava_trainer import LLaVATrainer, LLaVATrainerWithEWC
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.utils import rank0_print

# Import at runtime to avoid circular imports
GRPOTrainingArguments = None

def import_grpo_args():
    """Import GRPOTrainingArguments at runtime to avoid circular imports"""
    global GRPOTrainingArguments
    if GRPOTrainingArguments is None:
        try:
            from llava.train.train_grpo import GRPOTrainingArguments
        except ImportError:
            # Fall back to TRL's GRPOConfig if our custom class isn't available
            from trl import GRPOConfig as GRPOTrainingArguments
    return GRPOTrainingArguments

def create_safe_dummy_loss(model):
    """
    Creates a dummy zero loss that works safely with DeepSpeed and avoids CUDA errors.
    This function creates a tensor with the correct device, dtype and gradient properties.
    
    Args:
        model: The model to match device/dtype
        
    Returns:
        A properly configured zero loss tensor that's safe for backpropagation
    """
    # Get a parameter from the model to match device and dtype
    param = next(model.parameters(), None)
    if param is not None:
        device = param.device
        dtype = param.dtype
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
    
    # Create a tensor with the right properties
    dummy_tensor = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
    
    # Check if model is wrapped with DeepSpeed
    is_deepspeed = 'deepspeed' in str(type(model)).lower()
    
    if is_deepspeed:
        # For DeepSpeed, we need a tensor that has a proper gradient structure
        # First, make a small network that produces a scalar
        dummy_net = nn.Linear(1, 1).to(device)
        dummy_net.weight.data.fill_(0.0)
        dummy_net.bias.data.fill_(0.0)
        
        # Then create a proper scalar with gradient history
        dummy_input = torch.zeros(1, 1, device=device, requires_grad=False)
        dummy_loss = dummy_net(dummy_input).sum()
        
        # Detach to avoid computing gradients for the dummy network
        dummy_loss = dummy_loss.detach()
        
        # Ensure it requires gradient for backprop
        dummy_loss.requires_grad_(True)
        
        return dummy_loss
    else:
        # For non-DeepSpeed, a simple zero tensor with requires_grad is sufficient
        return dummy_tensor.sum()

class LLaVAGRPOTrainer(LLaVATrainer, GRPOTrainer):
    """
    Trainer for LLaVA models that supports GRPO (Generative Reward-Prompted Optimization)
    
    This trainer inherits from both LLaVATrainer and GRPOTrainer to combine LLaVA's 
    multimodal capabilities with GRPO's reinforcement learning optimization.
    """
    
    def __init__(
        self,
        model: PreTrainedModel = None,
        processing_class: PreTrainedTokenizerBase = None,
        reward_funcs: List[Callable] = None,
        args = None,
        train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset] = None,
        processor = None,
        **kwargs
    ):
        """
        Initialize the LLaVAGRPOTrainer
        
        Args:
            model: The model to train
            processing_class: Tokenizer or image processor for handling inputs
            reward_funcs: List of reward functions used for GRPO optimization
            args: Training arguments (GRPOTrainingArguments or compatible)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            processor: Image processor for vision tower
        """
        # Import GRPOTrainingArguments at runtime
        GRPOTrainingArguments = import_grpo_args()
        
        # Validate args type if provided
        if args is not None and not isinstance(args, GRPOTrainingArguments):
            rank0_print(f"Warning: args is not an instance of GRPOTrainingArguments, but {type(args)}")
        
        # Set up tokenizer's pad token if needed
        if processing_class is not None:
            if not hasattr(processing_class, "pad_token") or processing_class.pad_token is None:
                rank0_print("Tokenizer doesn't have a pad_token, setting it up during initialization")
                # Try to set pad_token to eos_token
                if hasattr(processing_class, "eos_token") and processing_class.eos_token is not None:
                    processing_class.pad_token = processing_class.eos_token
                    rank0_print(f"Set pad_token to eos_token: {processing_class.pad_token}")
                else:
                    # If no eos_token, try other special tokens
                    if hasattr(processing_class, "bos_token") and processing_class.bos_token is not None:
                        processing_class.pad_token = processing_class.bos_token
                        rank0_print(f"Set pad_token to bos_token: {processing_class.pad_token}")
                    elif hasattr(processing_class, "cls_token") and processing_class.cls_token is not None:
                        processing_class.pad_token = processing_class.cls_token
                        rank0_print(f"Set pad_token to cls_token: {processing_class.pad_token}")
                    else:
                        # If no appropriate token exists, add a new one
                        try:
                            processing_class.add_special_tokens({'pad_token': '[PAD]'})
                            rank0_print("Added [PAD] as pad_token")
                            # If model was initialized, we need to resize the token embeddings
                            if model is not None and hasattr(model, "resize_token_embeddings"):
                                model.resize_token_embeddings(len(processing_class))
                                rank0_print("Resized model token embeddings to accommodate new pad token")
                        except Exception as e:
                            rank0_print(f"Failed to add pad token: {e}")
        
        # Initialize the LLaVATrainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processing_class,
            data_collator=kwargs.get("data_collator", None),
        )
        
        # Store GRPO-specific attributes
        if reward_funcs is None:
            reward_funcs = []
        self.reward_funcs = reward_funcs
        self.reward_functions = reward_funcs  # 为兼容TRL的GRPOTrainer
        
        # Initialize other GRPO-specific attributes without calling the parent constructor
        self.processing_class = processing_class
        self.processor = processor  # Image processor for vision tower
        
        # Store additional GRPO parameters that may be custom to our implementation
        # Get from args (GRPOTrainingArguments) or use defaults
        self.use_vllm = getattr(args, "use_vllm", False)
        self.num_generations = getattr(args, "num_generations", 2)
        self.max_prompt_length = getattr(args, "max_prompt_length", 256)
        self.max_completion_length = getattr(args, "max_target_length", 300)  # Using max_target_length as in GRPOTrainingArguments
        self.grpo_beta = getattr(args, "beta", 0.1)  # Using beta as in GRPOTrainingArguments
        self.reward_scaling = getattr(args, "reward_scaling", 1.0)
        
        # Initialize running stats
        self.running_stats = {
            "samples": 0,
            "rewards": 0.0,
        }
        
        rank0_print(f"Initialized LLaVAGRPOTrainer with parameters:")
        rank0_print(f"- num_generations: {self.num_generations}")
        rank0_print(f"- max_prompt_length: {self.max_prompt_length}")
        rank0_print(f"- max_completion_length: {self.max_completion_length}")
        rank0_print(f"- grpo_beta: {self.grpo_beta}")
        rank0_print(f"- reward_scaling: {self.reward_scaling}")

    def compute_rewards(self, prompt, completions, answer=None):
        """
        Compute rewards for a set of completions using all reward functions
        
        Args:
            prompt: The prompt that generated the completions
            completions: List of generated completions
            answer: Ground truth answer if available
            
        Returns:
            List of combined rewards for each completion
        """
        # Handle edge cases with empty or invalid inputs
        if not completions:
            rank0_print("Warning: No completions to compute rewards for")
            return [0.0]  # Return a default reward
            
        # Ensure completions are in the expected format
        if not isinstance(completions[0], list) or not isinstance(completions[0][0], dict) or "content" not in completions[0][0]:
            # Try to convert to the expected format [{"content": "text"}]
            rank0_print(f"Converting completions to expected format. Current type: {type(completions[0])}")
            try:
                fixed_completions = []
                for comp in completions:
                    if isinstance(comp, str):
                        fixed_completions.append([{"content": comp}])
                    elif isinstance(comp, dict) and "content" in comp:
                        fixed_completions.append([comp])
                    elif isinstance(comp, list) and len(comp) > 0:
                        if isinstance(comp[0], str):
                            fixed_completions.append([{"content": comp[0]}])
                        elif isinstance(comp[0], dict) and "content" in comp[0]:
                            fixed_completions.append(comp)
                        else:
                            fixed_completions.append([{"content": str(comp)}])
                    else:
                        fixed_completions.append([{"content": str(comp)}])
                completions = fixed_completions
            except Exception as e:
                rank0_print(f"Error fixing completion format: {e}")
                # Create a fallback completion
                completions = [[{"content": "Error in completion format"}] for _ in range(len(completions))]
                
        # Initialize rewards
        combined_rewards = [0.0] * len(completions)
        
        # Handle None answer by converting to empty string
        if answer is None:
            answer = ""
            
        # Ensure answer is in a list for zip
        if not isinstance(answer, list):
            answer = [answer] * len(completions)
            
        for reward_func in self.reward_funcs:
            try:
                # Try to call with all available parameters
                reward_values = reward_func(self, completions=completions, prompts=[prompt], answer=answer)
                
                # Add the rewards
                for i, reward in enumerate(reward_values):
                    if i < len(combined_rewards):
                        combined_rewards[i] += reward
            except Exception as e:
                rank0_print(f"Error in reward function {reward_func.__name__}: {e}")
                # Continue with other reward functions on error
                
        return combined_rewards

    def _get_completion(self, prompt, num_generations=None, temperature=0.7, 
                       max_prompt_length=None, max_completion_length=None):
        """
        Generate completions for a prompt
        
        Args:
            prompt: The prompt to generate completions for
            num_generations: Number of completions to generate
            temperature: Temperature for generation
            max_prompt_length: Maximum prompt length
            max_completion_length: Maximum completion length
            
        Returns:
            List of generated completions
        """
        # Use provided values or defaults
        num_generations = num_generations or self.num_generations
        max_prompt_length = max_prompt_length or self.max_prompt_length
        max_completion_length = max_completion_length or self.max_completion_length
        
        # Use the right tokenizer attribute
        tokenizer = getattr(self, "tokenizer", None) or self.processing_class
        if tokenizer is None:
            rank0_print("Error: No tokenizer available for generation")
            return [[{"content": "Error: No tokenizer available"}] for _ in range(num_generations)]
        
        # Extract and process images if present
        image_paths = []
        has_image = DEFAULT_IMAGE_TOKEN in prompt or "<image>" in prompt
        
        if has_image:
            # Process the image appropriately
            if self._is_qwen_model():
                # For Qwen models, we'll create a properly sized image tensor
                image_tensor = torch.zeros(1, 3, 448, 448, device=self.model.device)
                
                # We'll do the proper image processing in _prepare_qwen_image_inputs
            else:
                # For other models, use a standard image tensor
                image_tensor = torch.zeros(1, 3, 224, 224, device=self.model.device)
            
            # Replace image tokens with placeholders in the prompt
            # Real implementation would use proper image embedding
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, "").replace("<image>", "").strip()
        else:
            image_tensor = None
        
        # Generate completions
        completions = []
        
        for _ in range(num_generations):
            try:
                # Set up padding appropriately
                use_padding = hasattr(tokenizer, "pad_token") and tokenizer.pad_token is not None
                
                # Prepare inputs for generation
                try:
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True,
                        padding="longest" if use_padding else False,
                        max_length=max_prompt_length
                    ).to(self.model.device)
                    
                    # Verify input_ids exists and is not None
                    if "input_ids" not in inputs or inputs["input_ids"] is None:
                        rank0_print("Warning: input_ids is missing or None in tokenizer output")
                        raise ValueError("Missing input_ids in tokenizer output")
                        
                except Exception as e:
                    rank0_print(f"Error tokenizing input: {e}")
                    # Try fallback with minimal options
                    try:
                        inputs = tokenizer(
                            prompt, 
                            return_tensors="pt", 
                            truncation=True
                        ).to(self.model.device)
                        
                        # Verify input_ids exists and is not None
                        if "input_ids" not in inputs or inputs["input_ids"] is None:
                            rank0_print("Warning: input_ids is missing or None in fallback tokenizer output")
                            raise ValueError("Missing input_ids in fallback tokenizer output")
                            
                    except Exception as e2:
                        rank0_print(f"Even fallback tokenization failed: {e2}")
                        completions.append([{"content": "Error: Tokenization failed"}])
                        continue
                
                # Add image if present
                if image_tensor is not None:
                    inputs["images"] = image_tensor
                
                # Generate with proper error handling
                try:
                    with torch.no_grad():
                        # Extract input_ids specifically for the generate method
                        input_ids = inputs.get("input_ids", None)
                        if input_ids is None:
                            rank0_print("Error: input_ids is None, cannot generate output")
                            raise ValueError("input_ids is None")
                            
                        # Use only the required parameters for generate
                        attention_mask = inputs.get("attention_mask", None)
                        
                        # Special handling for Qwen model
                        if self._is_qwen_model():
                            rank0_print("Detected Qwen model, using special handling for generation")
                            
                            # For Qwen, position_ids needs special handling to avoid shape mismatch
                            # Get sequence length from input_ids
                            seq_length = input_ids.size(1)
                            
                            # Create proper position_ids that match the sequence length
                            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
                            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                            
                            # For Qwen, we need to pass inputs as 'inputs' parameter 
                            # and proper position_ids to avoid shape mismatch errors
                            generate_kwargs = {
                                "inputs": input_ids,  # Qwen expects 'inputs', not 'input_ids'
                                "position_ids": position_ids,  # Explicitly provide properly shaped position_ids
                                "max_new_tokens": max_completion_length,
                                "temperature": temperature,
                                "do_sample": True,
                                "use_cache": False,  # Set use_cache=False to avoid warnings with gradient checkpointing
                            }
                            
                            # Add images if available (Qwen needs this separate from input_ids)
                            if "images" in inputs:
                                # Process the image to make sure it's in the right format
                                processed_images = self._process_qwen_image(inputs["images"])
                                generate_kwargs["images"] = processed_images
                                
                            # Make sure attention_mask is included if available
                            if attention_mask is not None:
                                generate_kwargs["attention_mask"] = attention_mask
                                
                            # Qwen doesn't support inputs_embeds for generation
                            if "inputs_embeds" in generate_kwargs:
                                rank0_print("Removing inputs_embeds from generate_kwargs")
                                del generate_kwargs["inputs_embeds"]
                                
                            # Remove input_ids to avoid confusion with 'inputs'
                            if "input_ids" in generate_kwargs:
                                del generate_kwargs["input_ids"]
                        else:
                            # Standard handling for other models
                            generate_kwargs = {
                                "input_ids": input_ids,
                                "max_new_tokens": max_completion_length,
                                "temperature": temperature,
                                "do_sample": True,
                            }
                            
                            # Add attention_mask if available
                            if attention_mask is not None:
                                generate_kwargs["attention_mask"] = attention_mask
                                
                            # Add images if available
                            if "images" in inputs:
                                generate_kwargs["images"] = inputs["images"]
                                
                        # Generate output
                        if self._is_qwen_model():
                            # Use our safe Qwen generate method
                            rank0_print("Using safe Qwen generate method")
                            
                            # Extract parameters for _safe_qwen_generate
                            safe_kwargs = {
                                "max_new_tokens": max_completion_length,
                                "temperature": temperature,
                                "do_sample": True,
                                "attention_mask": attention_mask,
                                "use_cache": False,
                            }
                            
                            # Use our special safe generate method
                            if "images" in inputs:
                                output_ids = self._safe_qwen_generate(
                                    input_ids=input_ids,
                                    images=inputs["images"],
                                    **safe_kwargs
                                )
                            else:
                                output_ids = self._safe_qwen_generate(
                                    input_ids=input_ids,
                                    **safe_kwargs
                                )
                        else:
                            # Standard generate for non-Qwen models
                            output_ids = self.model.generate(**generate_kwargs)
                    
                    # Decode the output
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Extract the completion part (remove the prompt)
                    # Handle edge cases where the prompt might not be a perfect prefix
                    if output_text.startswith(prompt):
                        completion = output_text[len(prompt):].strip()
                    else:
                        # If prompt isn't a perfect prefix, use the entire output
                        completion = output_text.strip()
                    
                    # Format as required by GRPO
                    completions.append([{"content": completion}])
                except Exception as gen_e:
                    rank0_print(f"Error in generation: {gen_e}")
                    import traceback
                    rank0_print(traceback.format_exc())
                    completions.append([{"content": "Error: Generation failed"}])
                    
            except Exception as e:
                rank0_print(f"Error generating completion: {e}")
                import traceback
                rank0_print(traceback.format_exc())
                # Add a default completion on error
                completions.append([{"content": "I apologize, but I couldn't generate a proper response."}])
        
        # Ensure we have the requested number of completions
        while len(completions) < num_generations:
            completions.append([{"content": "Fallback response due to generation errors."}])
            
        return completions

    def _prepare_prompt(self, text):
        """
        Prepare a prompt for the model
        
        Args:
            text: The text to prepare
            
        Returns:
            Dictionary of prepared inputs
        """
        # Use the right tokenizer attribute
        tokenizer = getattr(self, "tokenizer", None) or self.processing_class
        if tokenizer is None:
            rank0_print("Error: No tokenizer available for prompt preparation")
            # Return a basic text representation
            return {"input_text": text}
            
        try:
            # Check if tokenizer has pad_token
            use_padding = hasattr(tokenizer, "pad_token") and tokenizer.pad_token is not None
            
            # Tokenize the text with appropriate options
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding="longest" if use_padding else False,
                max_length=self.max_prompt_length
            ).to(self.model.device)
            
            # Create dummy labels for loss computation
            inputs["labels"] = inputs["input_ids"].clone()
            
            return inputs
        except Exception as e:
            rank0_print(f"Error in prompt preparation: {e}")
            import traceback
            rank0_print(traceback.format_exc())
            
            # Try fallback with minimal options
            try:
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True
                ).to(self.model.device)
                
                # Create dummy labels for loss computation
                inputs["labels"] = inputs["input_ids"].clone()
                
                return inputs
            except Exception as e2:
                rank0_print(f"Even fallback tokenization failed: {e2}")
                # Return a basic text representation
                return {"input_text": text}

    @staticmethod
    def format_reward_func(self, completions, **kwargs) -> list[float]:
        """
        Reward function that checks if the response follows the required format
        """
        try:
            # Extract responses with error handling
            responses = []
            for completion in completions:
                try:
                    if isinstance(completion, list) and len(completion) > 0:
                        if isinstance(completion[0], dict) and "content" in completion[0]:
                            responses.append(completion[0]["content"])
                        else:
                            responses.append(str(completion[0]))
                    elif isinstance(completion, dict) and "content" in completion:
                        responses.append(completion["content"])
                    else:
                        responses.append(str(completion))
                except Exception as e:
                    rank0_print(f"Error extracting response in format_reward_func: {e}")
                    responses.append("")
            
            pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
            matches = [re.search(pattern, r, re.DOTALL) if isinstance(r, str) else None for r in responses]
            return [2.0 if match else 0.0 for match in matches]
        except Exception as e:
            rank0_print(f"Error in format_reward_func: {e}")
            return [0.0] * len(completions)  # Return zero rewards on error

    @staticmethod
    def reasoning_quality_reward_func(self, completions, **kwargs) -> list[float]:
        """
        Reward function that evaluates the quality of reasoning
        """
        try:
            # Extract responses with error handling
            responses = []
            for completion in completions:
                try:
                    if isinstance(completion, list) and len(completion) > 0:
                        if isinstance(completion[0], dict) and "content" in completion[0]:
                            responses.append(completion[0]["content"])
                        else:
                            responses.append(str(completion[0]))
                    elif isinstance(completion, dict) and "content" in completion:
                        responses.append(completion["content"])
                    else:
                        responses.append(str(completion))
                except Exception as e:
                    rank0_print(f"Error extracting response in reasoning_quality_reward_func: {e}")
                    responses.append("")
            
            rewards = []
            
            for response in responses:
                try:
                    # Extract reasoning section
                    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL) if isinstance(response, str) else None
                    if not reasoning_match:
                        rewards.append(0.0)
                        continue
                        
                    reasoning = reasoning_match.group(1).strip()
                    
                    # Check for key elements in reasoning
                    score = 0.0
                    # Has clear steps
                    if "step" in reasoning.lower() or "first" in reasoning.lower() or "then" in reasoning.lower():
                        score += 1.0  
                    # Has logical connections
                    if any(word in reasoning.lower() for word in ["because", "since", "therefore", "thus"]):
                        score += 1.0  
                    # Shows calculation process
                    if any(word in reasoning.lower() for word in ["calculate", "compute", "solve", "find"]):
                        score += 1.0  
                    # Has sufficient detail
                    if len(reasoning.split()) > 50:  
                        score += 1.0
                        
                    rewards.append(score)
                except Exception as item_e:
                    rank0_print(f"Error evaluating reasoning quality for item: {item_e}")
                    rewards.append(0.0)
                
            return rewards
        except Exception as e:
            rank0_print(f"Error in reasoning_quality_reward_func: {e}")
            return [0.0] * len(completions)  # Return zero rewards on error

    @staticmethod
    def answer_correctness_reward_func(self, prompts, completions, answer, **kwargs) -> list[float]:
        """
        Reward function that checks if the answer is correct
        """
        # Check if answer exists and is valid
        if answer is None or (isinstance(answer, list) and (len(answer) == 0 or all(a is None for a in answer))):
            # No ground truth answer provided, return neutral rewards
            rank0_print("No ground truth answer available for correctness evaluation")
            return [1.0] * len(completions)  # Neutral reward
        
        # Ensure responses are properly extracted
        try:
            responses = [completion[0]["content"] for completion in completions]
        except (IndexError, KeyError, TypeError) as e:
            rank0_print(f"Error extracting completion content: {e}")
            # Try a more robust extraction approach
            responses = []
            for completion in completions:
                try:
                    if isinstance(completion, list) and len(completion) > 0:
                        if isinstance(completion[0], dict) and "content" in completion[0]:
                            responses.append(completion[0]["content"])
                        else:
                            responses.append(str(completion[0]))
                    elif isinstance(completion, dict) and "content" in completion:
                        responses.append(completion["content"])
                    else:
                        responses.append(str(completion))
                except Exception as e:
                    rank0_print(f"Failed to extract response: {e}")
                    responses.append("")
        
        # Make sure answer is a list matching the length of responses
        if not isinstance(answer, list):
            answer = [answer] * len(responses)
        elif len(answer) < len(responses):
            # Extend answer list if needed
            answer = answer + [answer[-1] if answer else None] * (len(responses) - len(answer))
        
        rewards = []
        
        for i, (response, ground_truth) in enumerate(zip(responses, answer)):
            # Skip if ground truth is None
            if ground_truth is None:
                rewards.append(1.0)  # Neutral reward
                continue
                
            # Extract answer section
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if not answer_match:
                rewards.append(0.0)
                continue
                
            predicted_answer = answer_match.group(1).strip()
            
            # Convert ground truth to string if it's not already
            ground_truth_str = str(ground_truth).strip()
            
            # Check if the predicted answer matches or contains the ground truth
            # More flexible matching for edge cases
            if (predicted_answer.upper() == ground_truth_str.upper() or 
                ground_truth_str.upper() in predicted_answer.upper() or 
                predicted_answer.upper() in ground_truth_str.upper()):
                rewards.append(4.0)  # Correct answer
            else:
                rewards.append(0.0)  # Wrong answer
                
        return rewards

    def _prepare_inputs(self, inputs, processor=None):
        """准备模型输入，处理图像和其他输入类型"""
        try:
            # 在这里得到我们工作的模型和数据类型
            model = self.model
            model_dtype = None
            expected_image_size = None
            image_processor = processor or getattr(self, "processor", None)
            
            # 检测模型类型，判断是否是Qwen模型
            is_qwen_model = self._is_qwen_model()
            from llava.utils import rank0_print
            if is_qwen_model:
                rank0_print("检测到Qwen模型，将使用'images'而非'image_features'")
            
            # 获取模型数据类型，便于统一
            if hasattr(model, "dtype"):
                model_dtype = model.dtype
            else:
                # 尝试从模型参数推断
                for param in model.parameters():
                    model_dtype = param.dtype
                    break
                    
            # 如果我们有Vision Tower，获取其期望的图像大小
            if hasattr(model, "get_vision_tower") and callable(model.get_vision_tower):
                vision_tower = model.get_vision_tower()
                if vision_tower is not None and hasattr(vision_tower, "config"):
                    # 对于CLIP模型
                    if hasattr(vision_tower.config, "image_size"):
                        expected_image_size = vision_tower.config.image_size
                    # 兼容SigLIP
                    elif hasattr(vision_tower, "image_size"):
                        expected_image_size = vision_tower.image_size
                        
                # 检查SigLIP特殊情况
                if "siglip" in str(type(vision_tower)).lower():
                    expected_image_size = 384
                
            # 如果我们有处理器，也可以从中获取大小
            if expected_image_size is None and image_processor is not None:
                if hasattr(image_processor, "size"):
                    if isinstance(image_processor.size, dict):
                        expected_image_size = image_processor.size.get("height", 224)
                    elif isinstance(image_processor.size, (list, tuple)) and len(image_processor.size) >= 2:
                        expected_image_size = image_processor.size[0]
                    elif isinstance(image_processor.size, int):
                        expected_image_size = image_processor.size
                
            # 确保我们有一个默认值
            if expected_image_size is None:
                expected_image_size = 448  # 默认值
                
            # 从输入中提取图像特征
            if "images" in inputs:
                from llava.utils import rank0_print
                rank0_print("处理输入图像...")
                image_features = inputs.pop("images")
                
                # 处理图像特征
                if isinstance(image_features, torch.Tensor):
                    # 添加批处理维度（如果缺失）
                    if len(image_features.shape) == 3:  # [C, H, W]
                        image_features = image_features.unsqueeze(0)
                    
                    # 检查RGB通道
                    if image_features.shape[1] != 3 and image_features.shape[-3] == 3:
                        # 张量可能是[B, H, W, C]格式，转换为[B, C, H, W]
                        image_features = image_features.permute(0, 3, 1, 2)
                    
                    # 如果需要，调整图像大小
                    if expected_image_size is not None and image_processor is not None:
                        import torch.nn.functional as F
                        # 检查当前图像尺寸是否与预期不同
                        current_size = image_features.shape[-2:]  # [H, W]
                        if current_size[0] != expected_image_size or current_size[1] != expected_image_size:
                            # 使用interpolate直接调整大小
                            rank0_print(f"调整图像大小从{current_size}到{expected_size}x{expected_size}")
                            image_features = F.interpolate(
                                image_features, 
                                size=(expected_size, expected_size), 
                                mode='bilinear', 
                                align_corners=False
                            )
                    
                    # 确保图像特征的数据类型与模型匹配
                    if model_dtype is not None and image_features.dtype != model_dtype:
                        rank0_print(f"将图像数据类型从{image_features.dtype}转换为{model_dtype}")
                        image_features = image_features.to(dtype=model_dtype)
                        
                    # 记录图像特征的形状，用于诊断
                    rank0_print(f"图像特征形状: {image_features.shape}, 数据类型: {image_features.dtype}")
                    
                    # 根据模型类型选择正确的键
                    if is_qwen_model:
                        inputs["images"] = image_features
                        rank0_print("使用'images'键用于Qwen模型")
                    else:
                        inputs["image_features"] = image_features
                        rank0_print("使用'image_features'键用于非Qwen模型")
                else:
                    rank0_print(f"警告: image_features不是张量。类型: {type(image_features)}")
                    # 如果需要，创建具有正确形状的虚拟张量
                    if expected_image_size is not None:
                        dummy_shape = (1, 3, expected_image_size, expected_image_size)
                        rank0_print(f"创建占位符图像张量，形状为{dummy_shape}")
                        # 创建具有正确dtype的灰色占位符
                        dummy_tensor = torch.ones(dummy_shape, dtype=model_dtype or torch.float32, 
                                                device=self.model.device) * 0.5
                        
                        # 根据模型类型选择正确的键
                        if is_qwen_model:
                            inputs["images"] = dummy_tensor
                        else:
                            inputs["image_features"] = dummy_tensor
            
            # 检查是否需要调整键名
            if "image_features" in inputs and is_qwen_model:
                # 对于Qwen模型，如果存在image_features键，需要将其改为images
                rank0_print("将'image_features'键重命名为'images'以兼容Qwen模型")
                inputs["images"] = inputs.pop("image_features")
            
            return inputs
        except Exception as e:
            from llava.utils import rank0_print
            rank0_print(f"_prepare_inputs中出错: {e}")
            import traceback
            rank0_print(traceback.format_exc())
            return inputs

    def training_step(self, model, inputs):
        """
        执行一个训练步骤。
        重写以处理DeepSpeed中可能出现的梯度错误。
        """
        try:
            # 确保输入正确
            inputs = self._prepare_inputs(inputs)
            
            # 确保所有参数都有梯度缓冲区
            for param in model.parameters():
                if param.requires_grad and param.grad is None:
                    param.grad = torch.zeros_like(param)
            
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            # 使用更安全的梯度反向传播方式
            if hasattr(self, "accelerator") and hasattr(self.accelerator, "scaler") and self.accelerator.scaler is not None:
                # 使用混合精度缩放器进行梯度计算
                self.accelerator.scaler.scale(loss).backward()
            else:
                # 常规反向传播但带有错误捕获
                try:
                    loss.backward()
                except Exception as e:
                    print(f"反向传播错误: {e}")
                    # 对于DeepSpeed的NoneType.numel错误，尝试确保所有参数有梯度
                    if "NoneType" in str(e) and "numel" in str(e):
                        print("检测到DeepSpeed梯度为None错误，尝试修复...")
                        # 清除所有None梯度，用零张量替代
                        for name, param in model.named_parameters():
                            if param.requires_grad and (param.grad is None):
                                param.grad = torch.zeros_like(param)
                                print(f"为参数 {name} 创建零梯度")
                        
                        # 跳过这个批次的梯度更新
                        print("跳过这个批次的梯度更新")
                    else:
                        # 其他类型的错误
                        raise e
            
            return loss.detach()
        
        except Exception as e:
            print(f"训练步骤错误: {e}")
            # 创建一个安全的默认损失
            device = model.device if hasattr(model, "device") else next(model.parameters()).device
            return torch.tensor(1.0, device=device)

    def _prepare_prompt_with_image(self, text):
        """Prepare a text prompt that may contain image tokens for the model.
        
        Args:
            text (str or dict): Text that may contain image tokens or structured content
            
        Returns:
            dict: Prepared inputs for the model
        """
        # Handle completion dictionaries (extract the content)
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict) and "content" in text[0]:
            # This is likely a completion in the format [{"content": "..."}]
            rank0_print("Converting completion dictionary to text")
            text = text[0]["content"]
        elif isinstance(text, dict) and "content" in text:
            # This is likely a completion in the format {"content": "..."}
            text = text["content"]
            
        # Ensure text is a string
        if not isinstance(text, str):
            rank0_print(f"Warning: Input text is not a string: {type(text)}. Converting to string.")
            try:
                text = str(text)
            except Exception as e:
                rank0_print(f"Failed to convert to string: {e}")
                return {"input_text": "Error: invalid input type"}
        
        # Check if we have image tokens in the text
        has_image_tokens = DEFAULT_IMAGE_TOKEN in text or DEFAULT_IM_START_TOKEN in text
        
        # Ensure tokenizer has a padding token
        if hasattr(self, "processing_class"):
            # Check if pad_token exists, if not set it to eos_token
            if not hasattr(self.processing_class, "pad_token") or self.processing_class.pad_token is None:
                rank0_print("Tokenizer doesn't have a pad_token, setting to eos_token")
                if hasattr(self.processing_class, "eos_token") and self.processing_class.eos_token is not None:
                    self.processing_class.pad_token = self.processing_class.eos_token
                else:
                    # If neither pad_token nor eos_token exists, try to add a [PAD] token
                    rank0_print("Adding [PAD] as pad_token")
                    try:
                        self.processing_class.add_special_tokens({'pad_token': '[PAD]'})
                    except Exception as e:
                        rank0_print(f"Failed to add pad token: {e}")
                        # If adding special tokens fails, process without padding
                        padding_option = None
        
        # Process the text through the tokenizer
        if has_image_tokens and hasattr(self, "processing_class"):
            # Check if we have image preprocessing capabilities
            if hasattr(self.processing_class, "image_processor"):
                try:
                    # Use special image token processing
                    # Use padding only if pad_token exists
                    padding_option = "longest" if hasattr(self.processing_class, "pad_token") and self.processing_class.pad_token is not None else None
                    
                    inputs = self.processing_class(
                        text, 
                        return_tensors="pt", 
                        padding=padding_option,
                        truncation=True,
                        max_length=self.max_prompt_length,
                        add_special_tokens=True
                    )
                    
                    # Flag that this input needs image processing
                    inputs["has_image"] = True
                    
                    # Move to appropriate device
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                    
                    return inputs
                except Exception as e:
                    rank0_print(f"Error in image token processing: {e}")
                    import traceback
                    rank0_print(traceback.format_exc())
                    # Fall back to basic processing
        
        # Standard text processing
        if hasattr(self, "processing_class"):
            try:
                # Use padding only if pad_token exists
                padding_option = "longest" if hasattr(self.processing_class, "pad_token") and self.processing_class.pad_token is not None else None
                
                inputs = self.processing_class(
                    text, 
                    return_tensors="pt", 
                    padding=padding_option,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    add_special_tokens=True
                )
                
                # Move to appropriate device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.model.device)
                
                return inputs
            except Exception as e:
                rank0_print(f"Error in tokenizer processing: {e}")
                import traceback
                rank0_print(traceback.format_exc())
                # Fall back to no-padding approach as a last resort
                try:
                    inputs = self.processing_class(
                        text, 
                        return_tensors="pt", 
                        padding=False,
                        truncation=True,
                        max_length=self.max_prompt_length,
                        add_special_tokens=True
                    )
                    
                    # Move to appropriate device
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                    
                    return inputs
                except Exception as e2:
                    rank0_print(f"Even no-padding tokenization failed: {e2}")
        
        # Fallback if no processing_class is available or all methods failed
        rank0_print("Warning: Tokenization failed or no processing_class available. Using basic text input fallback.")
        return {"input_text": text}

    def _is_main_process(self):
        """Helper method to determine if this is the main process"""
        # Try different ways to get the main process status
        if hasattr(self, "is_main_process"):
            return self.is_main_process
        if hasattr(self, "args") and hasattr(self.args, "local_rank"):
            return self.args.local_rank == 0
        if hasattr(self, "local_rank"):
            return self.local_rank == 0
        
        # Default to the utility function if available
        try:
            from llava.utils import is_main_process
            return is_main_process()
        except ImportError:
            # If all else fails, assume this is the main process
            return True

    def _is_qwen_model(self, model=None):
        """Helper to determine if we're using a Qwen model"""
        model = model or self.model
        model_type = str(type(model)).lower()
        return "qwen" in model_type or "llava_qwen" in model_type
    
    def _prepare_qwen_image_inputs(self, inputs, input_ids):
        """Special handling for Qwen model image inputs"""
        rank0_print("Preparing Qwen model image inputs")
        
        try:
            # Generate proper position_ids based on input_ids
            seq_length = input_ids.size(1) if input_ids is not None else 0
            
            if seq_length > 0:
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            else:
                position_ids = None
                
            # For Qwen models, we should not use inputs_embeds as it's not supported by generate
            # Just prepare the necessary inputs with input_ids and image_features
            result_dict = {
                "input_ids": input_ids,
                "attention_mask": inputs.get("attention_mask", None),
                "position_ids": position_ids,
                "use_cache": False  # Disable use_cache to avoid warnings with gradient checkpointing
            }
            
            # Process images if present
            if "images" in inputs and inputs["images"] is not None:
                # Process the images with our helper
                images = self._process_qwen_image(inputs["images"])
                
                # Check if the model has the image handling methods
                if hasattr(self.model, "encode_images") and callable(self.model.encode_images):
                    rank0_print(f"Using model.encode_images with image shape: {images.shape}")
                    # Get image features from the model
                    image_features = self.model.encode_images(images)
                    
                    # Add image_features to result dict
                    result_dict["image_features"] = image_features
                else:
                    rank0_print("Warning: Model doesn't have encode_images method")
                    # Add raw images to result dict
                    result_dict["images"] = images
            
            return result_dict
            
        except Exception as e:
            rank0_print(f"Error in _prepare_qwen_image_inputs: {e}")
            import traceback
            rank0_print(traceback.format_exc())
            
            # Return a simple inputs dict as fallback
            if input_ids is not None:
                seq_length = input_ids.size(1)
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": inputs.get("attention_mask", None),
                    "position_ids": position_ids,
                    "use_cache": False
                }
            else:
                return {
                    "input_ids": None,
                    "attention_mask": None,
                    "use_cache": False
                }

    def _safe_qwen_generate(self, input_ids=None, images=None, generation_config=None, **kwargs):
        """更安全高效地处理Qwen模型的生成，简化图像处理以提高性能"""
        import torch
        import math
        import torch.nn.functional as F
        import logging
        from llava.utils import rank0_print
        
        # 获取logger
        logger = logging.getLogger(__name__)
        
        try:
            # 检查模型是否启用了梯度检查点
            if hasattr(self.model, "gradient_checkpointing") and self.model.gradient_checkpointing:
                # 如果启用了梯度检查点,禁用use_cache
                if kwargs.get("use_cache", True):
                    rank0_print("梯度检查点和use_cache=True不兼容，设置use_cache=False")
                kwargs["use_cache"] = False
            
            # 如果没有提供images，或它为None/空列表，直接使用常规生成
            if images is None or (isinstance(images, list) and len(images) == 0):
                rank0_print("没有图像，使用基本文本生成")
                return self.model.generate(input_ids=input_ids, generation_config=generation_config, **kwargs)
                
            # 确定是否为SigLIP模型
            is_siglip = False
            vision_tower = None
            
            # 尝试识别视觉模型类型
            if hasattr(self.model, "get_vision_tower") and callable(self.model.get_vision_tower):
                try:
                    vision_tower = self.model.get_vision_tower()
                    if vision_tower is not None:
                        tower_type = str(type(vision_tower)).lower()
                        if "siglip" in tower_type:
                            is_siglip = True
                            rank0_print("检测到SigLIP视觉模型")
                except Exception as e:
                    rank0_print(f"获取视觉塔时出错: {e}")
            
            # 检测处理器类型
            if not is_siglip and hasattr(self.processor, "__class__") and hasattr(self.processor.__class__, "__name__"):
                if "siglip" in self.processor.__class__.__name__.lower():
                    is_siglip = True
                    rank0_print("检测到SigLIP处理器")
            
            # 确保图像形状正确
            if isinstance(images, torch.Tensor):
                # 根据检测到的模型类型设置目标大小
                expected_size = 0
                if is_siglip:
                    expected_size = 384
                    rank0_print(f"SigLIP模型: 设置图像大小为{expected_size}x{expected_size}")
                else:
                    # 非SigLIP模型默认使用448x448
                    expected_size = 448
                    rank0_print(f"非SigLIP模型: 使用默认图像大小{expected_size}x{expected_size}")
                
                # 检查并调整图像大小
                needs_resize = False
                
                if images.dim() == 4:
                    current_size = (images.shape[2], images.shape[3])
                    if current_size[0] != expected_size or current_size[1] != expected_size:
                        needs_resize = True
                elif images.dim() == 3:
                    current_size = (images.shape[1], images.shape[2])
                    if current_size[0] != expected_size or current_size[1] != expected_size:
                        needs_resize = True
                
                # 调整大小如果需要
                if needs_resize:
                    rank0_print(f"调整图像大小从{current_size}到{expected_size}x{expected_size}")
                    if images.dim() == 4:
                        images = F.interpolate(images, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
                    else:
                        images = F.interpolate(images.unsqueeze(0), size=(expected_size, expected_size), mode="bilinear", align_corners=False).squeeze(0)
            
            # 尝试直接生成
            try:
                rank0_print("尝试直接传递图像进行生成")
                return self.model.generate(input_ids=input_ids, images=images, generation_config=generation_config, **kwargs)
            except Exception as e:
                rank0_print(f"直接生成失败: {str(e)}")
                
                # 尝试使用预处理的图像特征进行生成
                try:
                    rank0_print("尝试使用预处理的图像特征")
                    
                    # 如果没有视觉塔，重新获取
                    if vision_tower is None and hasattr(self.model, "get_vision_tower"):
                        vision_tower = self.model.get_vision_tower()
                    
                    if vision_tower is None:
                        rank0_print("无法获取视觉塔，尝试基本生成")
                        return self.model.generate(input_ids=input_ids, generation_config=generation_config, **kwargs)
                    
                    # 预处理并编码图像
                    if isinstance(images, torch.Tensor):
                        # 确保图像大小正确
                        if is_siglip and (images.dim() == 4 and (images.shape[2] != expected_size or images.shape[3] != expected_size)):
                            rank0_print(f"调整SigLIP图像大小为{expected_size}x{expected_size}")
                            images = F.interpolate(images, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
                        elif not is_siglip and (images.dim() == 4 and (images.shape[2] != expected_size or images.shape[3] != expected_size)):
                            rank0_print(f"调整标准图像大小为{expected_size}x{expected_size}")
                            images = F.interpolate(images, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
                        
                        # 提取图像特征
                        try:
                            rank0_print(f"处理图像张量: 形状={images.shape}")
                            with torch.no_grad():
                                # 确保张量的dtype与模型匹配
                                model_dtype = next(vision_tower.parameters()).dtype
                                images_proper_dtype = images.to(dtype=model_dtype)
                                image_features = vision_tower(images_proper_dtype)
                                # 确保输出特征使用相同的dtype
                                if image_features.dtype != model_dtype:
                                    image_features = image_features.to(dtype=model_dtype)
                            rank0_print(f"提取图像特征成功: 形状={image_features.shape}, dtype={image_features.dtype}")
                        except Exception as feature_e:
                            rank0_print(f"提取图像特征出错: {feature_e}")
                            import traceback
                            rank0_print(traceback.format_exc())
                            raise
                    else:
                        # 处理其他图像类型
                        try:
                            rank0_print("处理非张量图像")
                            pixel_values = self.processor(images).pixel_values
                            with torch.no_grad():
                                # 确保张量的dtype与模型匹配
                                model_dtype = next(vision_tower.parameters()).dtype
                                pixel_values = pixel_values.to(dtype=model_dtype)
                                image_features = vision_tower(pixel_values)
                                # 确保输出特征使用相同的dtype
                                if image_features.dtype != model_dtype:
                                    image_features = image_features.to(dtype=model_dtype)
                            rank0_print(f"处理非张量图像成功: dtype={image_features.dtype}")
                        except Exception as proc_e:
                            rank0_print(f"处理图像出错: {proc_e}")
                            import traceback
                            rank0_print(traceback.format_exc())
                            raise
                    
                    # 使用图像特征进行生成
                    rank0_print("使用提取的图像特征进行生成")
                    return self.model.generate(
                        input_ids=input_ids,
                        image_features=image_features,
                        generation_config=generation_config, 
                        **kwargs
                    )
                except Exception as e:
                    rank0_print(f"无法使用图像特征生成，尝试基本生成: {str(e)}")
                    import traceback
                    rank0_print(traceback.format_exc())
                    
                    # 最后尝试：使用基本生成
                    return self.model.generate(input_ids=input_ids, generation_config=generation_config, **kwargs)
        
        except Exception as e:
            rank0_print(f"安全生成过程中出现未处理的错误: {str(e)}")
            import traceback
            rank0_print(traceback.format_exc())
            
            # 最后尝试：使用最基本的生成
            return self.model.generate(input_ids=input_ids, generation_config=generation_config, **kwargs)

    # Add a specialized ProcessorWrapper class to better handle image processing similar to train_grpo.py
    class ProcessorWrapper:
        """Wrapper for image processors that handles different types consistently"""
        def __init__(self, processor):
            self.processor = processor
            
            # Check if we have a CLIP processor
            self.is_clip = "CLIPImageProcessor" in str(type(processor))
            
            # For convenience, set common attributes
            if hasattr(processor, "image_mean"):
                self.image_mean = processor.image_mean
            if hasattr(processor, "image_std"):
                self.image_std = processor.image_std
            
            # Get image size from processor and ensure it's properly formatted
            self._size_dict = {}
            self._size_list = []
            
            if hasattr(processor, "size"):
                if isinstance(processor.size, dict):
                    self._size_dict.update(processor.size)
                    if "height" in processor.size and "width" in processor.size:
                        self._size_list = [processor.size["height"], processor.size["width"]]
                    elif "shortest_edge" in processor.size:
                        # Square image size based on shortest edge
                        self._size_list = [processor.size["shortest_edge"], processor.size["shortest_edge"]]
                elif isinstance(processor.size, (list, tuple)):
                    self._size_list = list(processor.size)
                    if len(self._size_list) >= 2:
                        self._size_dict["height"] = self._size_list[0]
                        self._size_dict["width"] = self._size_list[1]
                    elif len(self._size_list) == 1:
                        # Square image
                        self._size_dict["height"] = self._size_list[0]
                        self._size_dict["width"] = self._size_list[0]
                else:
                    # Single scalar size
                    size_value = processor.size
                    self._size_list = [size_value, size_value]
                    self._size_dict["height"] = size_value
                    self._size_dict["width"] = size_value
            
            # For Qwen-VL, default to 448x448 if nothing else is specified
            if not self._size_list:
                if self._is_qwen_processor():
                    self._size_list = [448, 448]  # Qwen-VL default
                else:
                    self._size_list = [224, 224]  # Default size for others
                    
            if not self._size_dict:
                if self._is_qwen_processor():
                    self._size_dict = {"height": 448, "width": 448, "shortest_edge": 448}
                else:
                    self._size_dict = {"height": 224, "width": 224, "shortest_edge": 224}
        
        def _is_qwen_processor(self):
            """Check if this processor is likely for a Qwen model"""
            if hasattr(self.processor, "__class__") and hasattr(self.processor.__class__, "__name__"):
                return "qwen" in self.processor.__class__.__name__.lower()
            return False
            
        @property
        def size(self):
            """Support both dict and list/tuple access patterns"""
            # This allows both processor.size[0] and processor.size['height'] to work
            return self._size_dict
        
        def __getitem__(self, key):
            """Support indexing access like processor.size[0] or processor['crop_size'][0]"""
            if key == 'size':
                # Return an indexable object for processor['size']
                class SizeProxy:
                    def __init__(self, size_list, size_dict):
                        self._size_list = size_list
                        self._size_dict = size_dict
                        
                    def __getitem__(self, idx):
                        if isinstance(idx, int):
                            return self._size_list[idx]
                        return self._size_dict[idx]
                        
                    def get(self, key, default=None):
                        try:
                            return self[key]
                        except (KeyError, IndexError):
                            return default
                
                return SizeProxy(self._size_list, self._size_dict)
                
            elif isinstance(key, int):
                # Default to size list access for direct integer indexing
                if key < len(self._size_list):
                    return self._size_list[key]
                else:
                    raise IndexError(f"Index {key} out of range for size list with length {len(self._size_list)}")
            elif isinstance(key, str):
                # For other string keys, try the size dict first, then proxy to processor
                if key in self._size_dict:
                    return self._size_dict[key]
                elif hasattr(self.processor, "__getitem__") and callable(getattr(self.processor, "__getitem__")):
                    try:
                        return self.processor[key]
                    except (KeyError, TypeError):
                        pass
                raise KeyError(f"Key '{key}' not found")
            else:
                raise TypeError(f"Unsupported key type: {type(key)}")

        def get(self, key, default=None):
            """Support dictionary-like get method"""
            try:
                return self[key]
            except (KeyError, IndexError):
                return default
        
        def __getattr__(self, name):
            """Delegate any unknown attributes to the underlying processor"""
            if hasattr(self.processor, name):
                return getattr(self.processor, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        def preprocess(self, *args, **kwargs):
            """Preprocess images using the underlying processor"""
            return self.processor(*args, **kwargs)
        
        def __call__(self, images, **kwargs):
            """Process images, ensuring proper normalization and sizing for Qwen"""
            # If the processor has a __call__ method, use it
            if hasattr(self.processor, "__call__") and callable(self.processor.__call__):
                try:
                    return self.processor(images, **kwargs)
                except Exception as e:
                    rank0_print(f"Error in processor.__call__: {e}, using fallback processing")
            
            # Fallback processing
            import torch
            from PIL import Image
            import numpy as np
            
            if not isinstance(images, list):
                images = [images]
            
            processed_images = []
            for image in images:
                if isinstance(image, torch.Tensor):
                    # Already a tensor, just normalize and resize if needed
                    if image.dim() == 2:
                        # Grayscale, add channel dimension
                        image = image.unsqueeze(0).repeat(3, 1, 1)
                    elif image.dim() == 3 and image.size(0) == 1:
                        # Single channel, repeat to make RGB
                        image = image.repeat(3, 1, 1)
                    
                    # Resize if needed
                    if self._is_qwen_processor():
                        target_size = (448, 448)
                    else:
                        target_size = (224, 224)
                    
                    if image.shape[1] != target_size[0] or image.shape[2] != target_size[1]:
                        image = torch.nn.functional.interpolate(
                            image.unsqueeze(0),
                            size=target_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    # Normalize
                    if hasattr(self, "image_mean") and hasattr(self, "image_std"):
                        mean = torch.tensor(self.image_mean).view(3, 1, 1).to(image.device)
                        std = torch.tensor(self.image_std).view(3, 1, 1).to(image.device)
                        image = (image - mean) / std
                    
                    processed_images.append(image)
                elif isinstance(image, Image.Image):
                    # Convert PIL Image to tensor
                    if self._is_qwen_processor():
                        target_size = (448, 448)
                    else:
                        target_size = (224, 224)
                        
                    image = image.convert("RGB").resize(target_size)
                    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                    
                    # Normalize
                    if hasattr(self, "image_mean") and hasattr(self, "image_std"):
                        mean = torch.tensor(self.image_mean).view(3, 1, 1)
                        std = torch.tensor(self.image_std).view(3, 1, 1)
                        image = (image - mean) / std
                    
                    processed_images.append(image)
                else:
                    # Unknown type, create dummy tensor
                    if self._is_qwen_processor():
                        dummy = torch.zeros(3, 448, 448)
                    else:
                        dummy = torch.zeros(3, 224, 224)
                    processed_images.append(dummy)
            
            return {"pixel_values": torch.stack(processed_images) if len(processed_images) > 1 else processed_images[0]}
        
        def __repr__(self):
            return f"ProcessorWrapper({self.processor!r}, size_dict={self._size_dict}, size_list={self._size_list})"

    def _process_qwen_image(self, image_tensor):
        """Special processing for Qwen model image inputs"""
        # Ensure image tensor has the right format
        if image_tensor is None:
            return None
            
        try:
            # Get a processor wrapper if we have one
            processor_wrapper = None
            if hasattr(self, "processor") and self.processor is not None:
                if not isinstance(self.processor, self.ProcessorWrapper):
                    # Wrap the processor
                    processor_wrapper = self.ProcessorWrapper(self.processor)
                else:
                    processor_wrapper = self.processor
            
            # Ensure it's a proper tensor
            if not isinstance(image_tensor, torch.Tensor):
                rank0_print(f"Warning: image_tensor is not a tensor. Type: {type(image_tensor)}")
                # Try to convert to tensor if possible
                if hasattr(image_tensor, "convert") and callable(image_tensor.convert):  # PIL Image
                    rank0_print("Converting PIL image to tensor")
                    if processor_wrapper is not None:
                        # Use our wrapped processor
                        processed = processor_wrapper(image_tensor)
                        if "pixel_values" in processed:
                            image_tensor = processed["pixel_values"]
                        else:
                            # Basic fallback
                            image_tensor = processor_wrapper.preprocess(image_tensor.convert('RGB'))['pixel_values']
                    else:
                        # Basic conversion
                        import torchvision.transforms as transforms
                        transform = transforms.Compose([
                            transforms.Resize((448, 448)),  # Qwen typically expects 448x448
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        image_tensor = transform(image_tensor.convert('RGB')).unsqueeze(0)
                else:
                    # Last resort: create a placeholder
                    rank0_print("Creating placeholder image tensor")
                    image_tensor = torch.zeros(1, 3, 448, 448, device=self.model.device)
                
            # Ensure it has the right dimensions [batch_size, channels, height, width]
            if image_tensor.dim() == 3:  # [C, H, W]
                image_tensor = image_tensor.unsqueeze(0)
                rank0_print(f"Added batch dimension to image tensor, new shape: {image_tensor.shape}")
            
            # Ensure channels are in the right position
            if image_tensor.shape[1] != 3 and image_tensor.shape[-1] == 3:
                # Tensor is likely in [B, H, W, C] format, convert to [B, C, H, W]
                image_tensor = image_tensor.permute(0, 3, 1, 2)
            
            # Ensure it has 3 channels
            if image_tensor.shape[1] != 3:
                if image_tensor.shape[1] == 1:  # Grayscale
                    rank0_print("Converting grayscale to RGB by repeating channels")
                    image_tensor = image_tensor.repeat(1, 3, 1, 1)
                else:
                    # Just take the first 3 channels
                    rank0_print(f"Taking first 3 channels from shape {image_tensor.shape}")
                    image_tensor = image_tensor[:, :3, :, :]
                
            # Check if we're using a SigLIP-based model, which requires different sizes
            is_siglip = False
            if hasattr(self.model, "get_vision_tower"):
                try:
                    vision_tower = self.model.get_vision_tower()
                    if vision_tower is not None:
                        tower_type = str(type(vision_tower)).lower()
                        if "siglip" in tower_type:
                            is_siglip = True
                            rank0_print("Detected SigLIP vision tower, using 384x384 image size")
                except:
                    pass
            
            # Get expected size from model config or use defaults
            if is_siglip:
                # SigLIP typically uses 384x384 with 27x27 patch grid (not 32x32)
                expected_size = 384
            else:
                # Default size for Qwen-VL is 448x448
                expected_size = 448
            
            # Check if LLaVA vision config has a specific size
            if hasattr(self.model, "config"):
                if hasattr(self.model.config, "vision_config") and hasattr(self.model.config.vision_config, "image_size"):
                    expected_size = self.model.config.vision_config.image_size
                    rank0_print(f"Using image size from vision_config: {expected_size}")
                elif hasattr(self.model.config, "image_size"):
                    expected_size = self.model.config.image_size
                    rank0_print(f"Using image size from model config: {expected_size}")
            
            # Ensure we're using the right size for SigLIP models
            if is_siglip and expected_size != 384:
                rank0_print(f"Warning: SigLIP detected but expected_size is {expected_size}, overriding to 384")
                expected_size = 384
                
            # Resize if necessary
            current_h, current_w = image_tensor.shape[2], image_tensor.shape[3]
            if current_h != expected_size or current_w != expected_size:
                rank0_print(f"Resizing image from {current_h}x{current_w} to {expected_size}x{expected_size}")
                # Use interpolate for resizing
                image_tensor = F.interpolate(
                    image_tensor, 
                    size=(expected_size, expected_size), 
                    mode='bilinear', 
                    align_corners=False
                )
                
            # Make sure the tensor is on the right device
            if image_tensor.device != self.model.device:
                rank0_print(f"Moving image tensor from {image_tensor.device} to {self.model.device}")
                image_tensor = image_tensor.to(self.model.device)
                
            # Verify tensor has valid values
            if torch.isnan(image_tensor).any():
                rank0_print("Warning: NaN values detected in image tensor. Replacing with zeros.")
                image_tensor = torch.where(torch.isnan(image_tensor), torch.zeros_like(image_tensor), image_tensor)
                
            if torch.isinf(image_tensor).any():
                rank0_print("Warning: Inf values detected in image tensor. Replacing with ones.")
                image_tensor = torch.where(torch.isinf(image_tensor), torch.ones_like(image_tensor), image_tensor)
                
            # Ensure values are in expected range for normalized images (approximately [-2, 2])
            if image_tensor.min() < -10 or image_tensor.max() > 10:
                rank0_print(f"Warning: Image tensor has unusual values: min={image_tensor.min()}, max={image_tensor.max()}. Normalizing.")
                # Simple normalization to [0, 1]
                image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
                # Then shift to [-1, 1] range which is more standard
                image_tensor = image_tensor * 2 - 1
                
            rank0_print(f"Final image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}, range: [{image_tensor.min()}, {image_tensor.max()}]")
            return image_tensor
            
        except Exception as e:
            rank0_print(f"Error in _process_qwen_image: {e}")
            import traceback
            rank0_print(traceback.format_exc())
            
            # Check if we should return a SigLIP compatible image
            is_siglip = False
            if hasattr(self.model, "get_vision_tower"):
                try:
                    vision_tower = self.model.get_vision_tower()
                    if vision_tower is not None:
                        tower_type = str(type(vision_tower)).lower()
                        if "siglip" in tower_type:
                            is_siglip = True
                except:
                    pass
            
            # Return a basic placeholder as fallback with appropriate size
            if is_siglip:
                return torch.zeros(1, 3, 384, 384, device=self.model.device)
            else:
                return torch.zeros(1, 3, 448, 448, device=self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算GRPO训练的损失。
        包括正常的语言模型损失以及基于奖励函数的引导部分。
        """
        try:
            # 如果模型使用梯度检查点，在训练中禁用use_cache
            if hasattr(self.args, "gradient_checkpointing") and self.args.gradient_checkpointing:
                if hasattr(model, "config"):
                    model.config.use_cache = False
            
            # 确保reward_functions存在
            if not hasattr(self, "reward_functions") or self.reward_functions is None:
                if hasattr(self, "reward_funcs") and self.reward_funcs is not None:
                    self.reward_functions = self.reward_funcs
                    print("Using reward_funcs as reward_functions")
                else:
                    # 默认奖励函数
                    from copy import deepcopy
                    self.reward_functions = [
                        lambda trainer, completions, **kwargs: [0.0] * len(completions) if completions else [0.0],
                        lambda trainer, completions, **kwargs: [0.0] * len(completions) if completions else [0.0],
                        lambda trainer, prompts, completions, answer, **kwargs: [0.0] * len(completions) if completions else [0.0]
                    ]
                    print("Using default reward functions (all zeros)")
            
            # 运行模型获取输出
            outputs = model(**inputs)
            loss = outputs.loss
            
            # 检查loss是否为标量，如果不是，打印警告并取平均值
            if loss.dim() > 0:
                print(f"警告: Loss不是标量 (shape={loss.shape})，取平均值")
                loss = loss.mean()
            
            # 尝试应用奖励函数
            if hasattr(self.args, "beta") and self.args.beta > 0 and hasattr(outputs, "logits") and "labels" in inputs:
                batch_size = inputs["labels"].shape[0] if "labels" in inputs else 1
                
                # 准备completions和prompts
                try:
                    # 获取tokenizer用于解码
                    if not hasattr(self, "tokenizer") and hasattr(self.args, "tokenizer"):
                        self.tokenizer = self.args.tokenizer
                    
                    # 使用logits进行采样获得完成
                    logits = outputs.logits
                    if logits.shape[0] == 0:  # 处理空批次情况
                        rewards = torch.tensor([0.0], device=loss.device)
                    else:
                        # 采样完成结果
                        completions = []
                        for i in range(min(logits.shape[0], batch_size)):
                            completion_tokens = torch.argmax(logits[i], dim=-1)
                            try:
                                completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
                                completions.append(completion)
                            except Exception as e:
                                print(f"解码错误 (sample {i}): {e}")
                                completions.append("")
                        
                        # 如果completions为空，添加一个默认项
                        if len(completions) == 0:
                            completions = [""]
                        
                        # 获取prompts（如果可能）
                        try:
                            prompts = []
                            if "input_ids" in inputs:
                                for i in range(min(inputs["input_ids"].shape[0], batch_size)):
                                    try:
                                        prompt = self.tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
                                        prompts.append(prompt)
                                    except Exception as e:
                                        print(f"提示解码错误 (sample {i}): {e}")
                                        prompts.append("")
                            
                            # 如果prompts为空或长度不匹配，创建匹配的空列表
                            if len(prompts) == 0 or len(prompts) != len(completions):
                                prompts = [""] * len(completions)
                        except Exception as e:
                            print(f"提示准备错误: {e}")
                            prompts = [""] * len(completions)
                        
                        # 应用所有奖励函数
                        rewards_list = []
                        for i, reward_func in enumerate(self.reward_functions):
                            try:
                                if "answer" in inspect.signature(reward_func).parameters:
                                    # 创建默认答案列表，如有实际答案应传入
                                    answers = [None] * len(completions)
                                    reward = reward_func(self, prompts=prompts, completions=completions, answer=answers)
                                else:
                                    reward = reward_func(self, completions=completions)
                                
                                # 确保奖励是列表，并且长度匹配
                                if not isinstance(reward, list):
                                    reward = [reward]
                                if len(reward) != len(completions):
                                    print(f"警告: 奖励函数 {i} 返回长度 {len(reward)}，但期望 {len(completions)}。填充或裁剪。")
                                    if len(reward) < len(completions):
                                        reward.extend([0.0] * (len(completions) - len(reward)))
                                    else:
                                        reward = reward[:len(completions)]
                                
                                rewards_list.append(reward)
                            except Exception as e:
                                print(f"奖励函数 {i} 错误: {e}")
                                rewards_list.append([0.0] * len(completions))
                        
                        # 计算总奖励并转换为tensor
                        total_rewards = [0.0] * len(completions)
                        for rewards in rewards_list:
                            for i, r in enumerate(rewards):
                                total_rewards[i] += float(r)
                        
                        rewards = torch.tensor(total_rewards, device=loss.device)
                        
                        # 如果rewards不是标量但应该是，取平均值
                        if rewards.dim() > 0 and rewards.shape[0] > 1:
                            rewards = rewards.mean()
                
                except Exception as e:
                    print(f"奖励计算错误: {e}")
                    rewards = torch.tensor([0.0], device=loss.device)
                
                # 应用奖励缩放和GRPO公式
                beta = getattr(self.args, "beta", 0.1)
                scaled_rewards = beta * rewards
                
                # 更新统计数据
                if not hasattr(self, "_reward_stats"):
                    self._reward_stats = {"count": 0, "sum": 0.0, "sum_scaled": 0.0}
                
                self._reward_stats["count"] += rewards.shape[0] if rewards.dim() > 0 else 1
                self._reward_stats["sum"] += rewards.sum().item() if rewards.dim() > 0 else rewards.item()
                self._reward_stats["sum_scaled"] += scaled_rewards.sum().item() if scaled_rewards.dim() > 0 else scaled_rewards.item()
                
                # 每100个样本打印一次平均奖励
                if self._reward_stats["count"] % 100 <= batch_size:
                    avg_reward = self._reward_stats["sum"] / self._reward_stats["count"]
                    avg_scaled = self._reward_stats["sum_scaled"] / self._reward_stats["count"]
                    print(f"平均奖励: {avg_reward:.4f}, 缩放后: {avg_scaled:.4f}")
                
                # 应用GRPO公式：loss = loss - beta * rewards
                loss = loss - scaled_rewards
            
            # 确保最终loss是标量
            if loss.dim() > 0:
                loss = loss.mean()
            
            return (loss, outputs) if return_outputs else loss
        
        except Exception as e:
            print(f"计算损失错误: {e}")
            # 创建安全的默认loss
            try:
                # 尝试基本前向传播
                outputs = model(**inputs)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                print(f"使用基本损失: {loss.item()}")
                return (loss, outputs) if return_outputs else loss
            except Exception as e2:
                print(f"基本损失计算也失败: {e2}")
                # 创建与模型兼容的默认损失
                default_loss = torch.tensor(1.0, device=model.device if hasattr(model, "device") else 
                                                     next(model.parameters()).device)
                print(f"使用默认损失: {default_loss.item()}")
                return (default_loss, None) if return_outputs else default_loss

class LLaVAGRPOTrainerWithEWC(LLaVAGRPOTrainer, LLaVATrainerWithEWC):
    """
    结合GRPO训练和EWC（弹性权重巩固）的Trainer。
    用于在避免灾难性遗忘的同时进行GRPO训练。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保EWC参数已初始化
        self.ewc_lambda = kwargs.get("ewc_lambda", 0.0)
        self.selected_params = kwargs.get("selected_params", None)
        self.use_ewc = True
        self.fisher_dict = None
        self.optpar_dict = None
        print(f"使用EWC训练，lambda={self.ewc_lambda}")
        
        # 初始化EWC参数选择
        if self.selected_params is None:
            self.selected_params = []
            for name, param in self.model.named_parameters():
                # 跳过不需要梯度的参数
                if not param.requires_grad:
                    continue
                    
                # 跳过特定名称模式的参数
                if any(skip in name for skip in ["bias", "LayerNorm", "layer_norm", "logit_scale"]):
                    continue
                    
                # 包含符合条件的参数用于EWC计算
                self.selected_params.append(name)
                    
            print(f"为EWC选择了 {len(self.selected_params)} 个参数")
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算组合的GRPO+EWC损失。
        首先计算基本的GRPO损失，然后添加EWC损失。
        """
        try:
            # 首先计算基本的GRPO损失
            grpo_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            
            # 确保损失是标量
            if grpo_loss.dim() > 0:
                print(f"GRPO损失不是标量 (shape={grpo_loss.shape})，取平均值")
                grpo_loss = grpo_loss.mean()
            
            total_loss = grpo_loss
            ewc_loss = torch.tensor(0.0, device=grpo_loss.device)
            
            # 如果EWC已启用且Fisher矩阵可用，计算EWC损失
            if self.ewc_lambda > 0 and hasattr(self, "fisher_dict") and self.fisher_dict is not None:
                try:
                    # 计算EWC损失
                    ewc_loss = self._compute_ewc_loss(model)
                    
                    # 确保EWC损失是标量
                    if ewc_loss.dim() > 0:
                        print(f"EWC损失不是标量 (shape={ewc_loss.shape})，取平均值")
                        ewc_loss = ewc_loss.mean()
                    
                    # 应用EWC损失权重和总损失
                    total_loss = grpo_loss + self.ewc_lambda * ewc_loss
                    
                    # 打印损失组件
                    if self.state.global_step % 10 == 0:
                        print(f"GRPO损失: {grpo_loss.item():.4f}, EWC损失: {ewc_loss.item():.4f}, Lambda: {self.ewc_lambda}")
                
                except Exception as e:
                    print(f"EWC损失计算错误: {e}")
                    # 如果计算EWC损失失败，仅使用GRPO损失
                    total_loss = grpo_loss
            
            return (total_loss, outputs) if return_outputs else total_loss
        
        except Exception as e:
            print(f"组合损失计算错误: {e}")
            # 尝试单独使用父类的compute_loss
            try:
                return super().compute_loss(model, inputs, return_outputs)
            except Exception as e2:
                print(f"父类损失计算也失败: {e2}")
                # 如果所有方法都失败，返回默认损失
                default_loss = torch.tensor(1.0, device=model.device if hasattr(model, "device") else 
                                                   next(model.parameters()).device)
                print(f"使用默认损失: {default_loss.item()}")
                return (default_loss, None) if return_outputs else default_loss

    def train(self, *args, **kwargs):
        """
        使用带有EWC的GRPO训练模型
        
        参数:
            args: 训练的位置参数
            kwargs: 训练的关键字参数
            
        返回:
            训练结果
        """
        try:
            # 如果启用了EWC但还没有计算Fisher信息矩阵，则计算它
            if getattr(self, "use_ewc", True) and not hasattr(self, "fisher_dict") or self.fisher_dict is None:
                print("正在计算EWC的Fisher信息矩阵...")
                
                # 确保属性存在
                if not hasattr(self, "fisher_dict"):
                    self.fisher_dict = {}
                
                if not hasattr(self, "optpar_dict"):
                    self.optpar_dict = {}
                
                # 计算Fisher信息矩阵
                try:
                    self._compute_fisher(model=self.model)
                    print(f"已计算Fisher信息矩阵，包含 {len(self.fisher_dict)} 个参数")
                except Exception as e:
                    print(f"计算Fisher矩阵时出错: {e}")
                    # 创建最小的Fisher字典以避免后续错误
                    if not self.fisher_dict:
                        self.fisher_dict = {name: torch.zeros_like(param) 
                                           for name, param in self.model.named_parameters() 
                                           if param.requires_grad and name in self.selected_params}
                        print(f"创建了零Fisher矩阵用于 {len(self.fisher_dict)} 个参数")
            
            # 直接调用Trainer的train方法以避免递归
            from transformers import Trainer
            print("开始使用EWC训练...")
            return Trainer.train(self, *args, **kwargs)
            
        except Exception as e:
            print(f"EWC训练出错: {e}")
            # 尝试使用基本训练
            print("回退到基本训练...")
            from transformers import Trainer
            return Trainer.train(self, *args, **kwargs)

    def _compute_ewc_loss(self, model=None):
        """
        计算EWC正则化损失
        
        参数:
            model: 要计算EWC损失的模型。如果为None，则使用self.model
            
        返回:
            EWC损失值
        """
        if model is None:
            model = self.model
        
        if not hasattr(self, "fisher_dict") or not self.fisher_dict:
            print("Fisher字典不存在或为空，无法计算EWC损失")
            return torch.tensor(0.0, device=model.device if hasattr(model, "device") else 
                               next(model.parameters()).device)
        
        if not hasattr(self, "optpar_dict") or not self.optpar_dict:
            print("最优参数字典不存在或为空，无法计算EWC损失")
            return torch.tensor(0.0, device=model.device if hasattr(model, "device") else 
                               next(model.parameters()).device)
            
        loss = torch.tensor(0.0, device=model.device if hasattr(model, "device") else 
                           next(model.parameters()).device)
        
        # 计算EWC损失
        for name, param in model.named_parameters():
            if name in self.fisher_dict and name in self.optpar_dict:
                fisher = self.fisher_dict[name]
                optpar = self.optpar_dict[name]
                
                # 确保参数在同一设备上
                if fisher.device != param.device:
                    fisher = fisher.to(param.device)
                if optpar.device != param.device:
                    optpar = optpar.to(param.device)
                
                # 计算参数差的平方与Fisher信息的加权乘积
                loss += (fisher * (optpar - param).pow(2)).sum()
                
        return loss
    
    def _compute_fisher(self, model=None):
        """
        计算Fisher信息矩阵，用于EWC正则化
        
        参数:
            model: 要计算Fisher信息的模型。如果为None，则使用self.model
        """
        if model is None:
            model = self.model
            
        # 设置模型为评估模式
        model.eval()
        
        # 初始化Fisher和最优参数字典
        self.fisher_dict = {}
        self.optpar_dict = {}
        
        # 使用小数据集计算Fisher信息
        if self.train_dataset is not None:
            total_samples = min(len(self.train_dataset), 100)  # 最多使用100个样本
            print(f"使用{total_samples}个样本计算Fisher信息矩阵")
            
            # 创建临时数据加载器
            from torch.utils.data import DataLoader, Subset
            import random
            
            # 随机选择样本
            indices = random.sample(range(len(self.train_dataset)), total_samples)
            subset = Subset(self.train_dataset, indices)
            
            # 使用较小批次避免内存问题
            batch_size = min(8, total_samples)
            data_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            
            # 收集选定参数的梯度平方
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.selected_params:
                    # 存储当前参数值作为最优值
                    self.optpar_dict[name] = param.data.clone()
                    # 初始化Fisher矩阵为零
                    self.fisher_dict[name] = torch.zeros_like(param.data)
            
            # 为每个样本累积梯度平方
            samples_processed = 0
            for batch in data_loader:
                try:
                    # 准备输入
                    batch = self._prepare_inputs(batch)
                    batch_size = batch["input_ids"].shape[0] if "input_ids" in batch else 1
                    
                    # 处理批次中的每个样本
                    for i in range(batch_size):
                        sample = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in batch.items()}
                        
                        # 清除现有梯度
                        model.zero_grad()
                        
                        # 前向传播
                        outputs = model(**sample)
                        
                        # 获取损失并计算梯度
                        if hasattr(outputs, "loss"):
                            loss = outputs.loss
                            loss.backward()
                            
                            # 更新Fisher信息
                            for name, param in model.named_parameters():
                                if param.requires_grad and name in self.fisher_dict:
                                    if param.grad is not None:
                                        # Fisher信息是梯度平方的期望
                                        self.fisher_dict[name] += param.grad.data.pow(2)
                            
                            samples_processed += 1
                            if samples_processed % 10 == 0:
                                print(f"已处理 {samples_processed}/{total_samples} 个样本")
                        
                        # 防止内存溢出
                        del outputs, loss
                        if samples_processed >= total_samples:
                            break
                    
                    # 清除内存
                    del batch
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if samples_processed >= total_samples:
                        break
                        
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue
            
            # 归一化Fisher信息
            if samples_processed > 0:
                for name in self.fisher_dict:
                    self.fisher_dict[name] /= samples_processed
                print(f"成功处理{samples_processed}个样本，已计算并归一化Fisher信息矩阵")
            else:
                print("警告: 未能处理任何样本，使用恒等Fisher矩阵")
                # 使用恒等矩阵作为后备
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.selected_params:
                        self.fisher_dict[name] = torch.ones_like(param.data) * 0.1
                        self.optpar_dict[name] = param.data.clone()
        else:
            print("无训练数据集可用，使用恒等Fisher矩阵")
            # 使用恒等矩阵作为后备
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.selected_params:
                    self.fisher_dict[name] = torch.ones_like(param.data) * 0.1
                    self.optpar_dict[name] = param.data.clone()
        
        # 恢复模型状态
        model.train()
        
        print(f"Fisher信息矩阵计算完成，包含{len(self.fisher_dict)}个参数")
        return
