import random
from re import L
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from constants import TEXT_BETWEEN_SHOTS

from logits_processor import RestrictiveTokensLogitsProcessor


def pad_left(t, value, dim):
    shape = list(t.shape)
    shape[dim] = 1
    return torch.cat([t.new_full(shape, value), t], dim=dim)


class PermCWModelWrapper:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        context_window_size: int,
        right_indentation: bool = False,
        version: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.context_window_size = context_window_size
        self.device = device
        # Left indentation is the default behavior as explained in the paper.
        self.right_indentation = right_indentation
        
        self.version = version

    def get_contexts_cache(self, contexts: List[str]) -> Dict:

        assert len(contexts) == 1
        contexts = contexts[0].split(TEXT_BETWEEN_SHOTS)
        
        encoded_input_context = self.tokenizer(
            [text + TEXT_BETWEEN_SHOTS for text in contexts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
    
        window_position_ids = torch.arange(encoded_input_context.input_ids.shape[1], device=self.device)
        max_window_length = encoded_input_context.input_ids.shape[1]
        window_length = encoded_input_context.attention_mask.sum(1)
        total_len = window_length.sum()

        if self.version == 1:
            inp = {}
            inp["input_ids"] = pad_left(encoded_input_context.input_ids.flatten(), self.tokenizer.bos_token_id, dim=0)
            inp["attention_mask"] = pad_left(encoded_input_context.attention_mask.flatten(), 1, dim=0)

            context_position_ids = inp['attention_mask'].cumsum(0)
            suffix_position_ids = context_position_ids.new_full((len(contexts), max_window_length), total_len) + window_position_ids.unsqueeze(0)
            prefix_position_ids = suffix_position_ids - window_length.unsqueeze(1)
            suffix_position_ids = pad_left(suffix_position_ids.flatten() + 2, 1, 0)
            prefix_position_ids = pad_left(prefix_position_ids.flatten() + 2, 1, 0)

            context_mask = torch.arange(len(contexts), device=self.device).unsqueeze(1).repeat(1, max_window_length)
            context_mask = pad_left(context_mask.flatten(), -1, 0)
            context_mask = context_mask[:, None] == context_mask[None, :]

            inp['input_ids'] = inp['input_ids'].unsqueeze(0)
            inp['attention_mask'] = inp['attention_mask'].unsqueeze(0)
            inp["position_ids"] = (
                context_position_ids.unsqueeze(0),
                suffix_position_ids.unsqueeze(0),
                prefix_position_ids.unsqueeze(0),
                context_mask.unsqueeze(0),
                1
            )

        elif self.version == 2:
            ...

        elif self.version == 3:
            inp = {}
            inp["input_ids"] = pad_left(encoded_input_context.input_ids.flatten(), self.tokenizer.bos_token_id, dim=0)
            inp["attention_mask"] = pad_left(encoded_input_context.attention_mask.flatten(), 1, dim=0)

            context_position_ids = inp['attention_mask'].cumsum(0)
            suffix_position_ids = context_position_ids.new_full((len(contexts), max_window_length), total_len)
            prefix_position_ids = suffix_position_ids + window_position_ids.unsqueeze(0) - window_length.unsqueeze(1)
            suffix_position_ids = pad_left(suffix_position_ids.flatten(), 0, 0)
            suffix_position_ids = suffix_position_ids + context_position_ids
            prefix_position_ids = pad_left(prefix_position_ids.flatten() + 2, 1, 0)

            context_mask = torch.arange(len(contexts), device=self.device).unsqueeze(1).repeat(1, max_window_length)
            context_mask = pad_left(context_mask.flatten(), -1, 0)
            context_mask = context_mask[:, None] < context_mask[None, :]
            
            inp['input_ids'] = inp['input_ids'].unsqueeze(0)
            inp['attention_mask'] = inp['attention_mask'].unsqueeze(0)
            inp["position_ids"] = (
                context_position_ids.unsqueeze(0),
                suffix_position_ids.unsqueeze(0),
                prefix_position_ids.unsqueeze(0),
                context_mask,
                3
            )

        context = self.model(**inp)

        context["past_attention_mask"] = inp["attention_mask"]
        # context["max_window_size"] = inp["input_ids"].shape[1]
        context["sum_windows_size"] = inp["attention_mask"].shape[1]
        return context

    def pcw_generate(
        self,
        contexts: Optional[List[str]] = None,
        task_text: Optional[str] = None,
        contexts_cache: Optional[Dict] = None,
        restrictive_logit_preprocessor: Optional[RestrictiveTokensLogitsProcessor] = None,
        **kwargs,
    ) -> str:
        """Note: Batching is not supported by PCW at the moment."""
        assert (contexts is None) != (
            contexts_cache is None
        ), "pcw_generate should work with contexts or cache, not with both!"
        cache = contexts_cache or self.get_contexts_cache(contexts)
        encoded_task_text = self.tokenizer(task_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        if restrictive_logit_preprocessor:
            restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_task_text["input_ids"].shape[1])
            kwargs["logits_processor"] = [restrictive_logit_preprocessor]
        combined_attention_mask = torch.cat(
            (cache["past_attention_mask"], encoded_task_text["attention_mask"]), dim=1
        ).to(self.device)
        res = self.model.generate(
            input_ids=encoded_task_text["input_ids"],
            attention_mask=combined_attention_mask,
            windows_key_values=cache["past_key_values"],
            # max_window_size=cache["max_window_size"],
            sum_windows_size=cache["sum_windows_size"],
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )[0]
        res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
        return self.tokenizer.decode(res[encoded_task_text["input_ids"].shape[1] :])
