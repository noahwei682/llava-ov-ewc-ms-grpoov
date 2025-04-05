import torch
from typing import List, Optional, Tuple, Dict, Any, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer

class GRPOConfig:
    def __init__(
        self,
        beta: float = 0.1,
        reward_functions: Optional[List[Callable]] = None,
        **kwargs
    ):
        self.beta = beta
        self.reward_functions = reward_functions or []
        for key, value in kwargs.items():
            setattr(self, key, value)

class GRPOTrainer(Trainer):
    """
    GRPO (Generative Reward-Powered Optimization) Trainer.
    Extends Trainer with GRPO-specific functionality.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_functions: Optional[List[Callable]] = None,
        **kwargs
    ):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.reward_functions = reward_functions or []
        self.config = GRPOConfig(**kwargs)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute GRPO loss for training.
        
        Args:
            model: The model to train
            inputs: Training inputs
            return_outputs: Whether to return model outputs
            
        Returns:
            Tuple of (loss, metrics)
        """
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Calculate rewards if reward functions are provided
        rewards = torch.zeros(logits.size(0), device=logits.device)
        for reward_fn in self.reward_functions:
            rewards += reward_fn(model, inputs, outputs)
        
        # GRPO loss calculation
        loss = -self.config.beta * rewards.mean()
        
        if return_outputs:
            return loss, outputs
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[List[str], torch.FloatTensor]:
        """
        Generate responses and calculate rewards.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Tuple of (generated texts, rewards)
        """
        outputs = self.model.generate(input_ids, **kwargs)
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Calculate rewards for generated responses
        rewards = torch.zeros(len(responses), device=self.model.device)
        for reward_fn in self.reward_functions:
            rewards += reward_fn(self.model, {"input_ids": input_ids}, {"logits": outputs})
            
        return responses, rewards 
