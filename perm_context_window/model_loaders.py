import torch
from transformers import (
    AutoConfig,
    LlamaTokenizer,
    PreTrainedTokenizerBase,
)

from perm_context_window.model_wrapper import PermCWModelWrapper
from perm_context_window.modeling_llama_with_permcw import LlamaForCausalLMPermCW

LLAMA_WINDOW_SIZE = 2048


def validate_model_name(model_name: str) -> None:
    assert "llama" in model_name, f"Unknown model: {model_name}"


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    if "llama" in model_name:
        if model_name == "seanmor5/tiny-llama-test" or "decapoda-research" in model_name:  # debug mode:
            tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
            # In case you load those models, we must override an incorrect config:
            # see: https://huggingface.co/decapoda-research/llama-7b-hf/discussions/12
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
        else:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            if model_name == "data/tiny-llama-test":
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_permcw_wrapper(
    model_name: str,
    cache_dir: str = None,
    right_indentation: bool = False,
    version: int=1,
) -> PermCWModelWrapper:
    validate_model_name(model_name)
    config = AutoConfig.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpus = torch.cuda.device_count() > 1
    model_args = {"cache_dir": cache_dir}
    if multi_gpus:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype

    #  Note that some LLaMa versions located in HF have an incorrect token mapping, we correct it here:
    # see: https://huggingface.co/decapoda-research/llama-7b-hf/discussions/12
    # also: https://github.com/tloen/alpaca-lora/issues/279
    model_args["bos_token_id"] = 1
    model_args["eos_token_id"] = 2
    model_obj = LlamaForCausalLMPermCW
    context_window_size = LLAMA_WINDOW_SIZE

    tokenizer = load_tokenizer(model_name)
    model = model_obj.from_pretrained(model_name, **model_args).eval()
    if not multi_gpus:
        model = model.to(device).half()

    return PermCWModelWrapper(model, tokenizer, device, context_window_size, right_indentation, version)
