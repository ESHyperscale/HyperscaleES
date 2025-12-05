from .tokenizer import GptTokenizer, WorldTokenizer, QwenTokenizer

# from . import rwkv4, rwkv5, rwkv5_2, rwkv6, rwkv7
from . import rwkv7, qrwkv6

from huggingface_hub.constants import HF_HOME
from huggingface_hub import hf_hub_download

from transformers import AutoModelForCausalLM

from pathlib import Path

import pickle

import jax
import jax.numpy as jnp

suffix = ".model"

versions = {
    # "4": rwkv4,
    # "5": rwkv5,
    # "5_2": rwkv5_2,
    # "6": rwkv6,
    "7": rwkv7,
    "6q": qrwkv6
}

models = {
    # "4w0.1B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-0.1B-v1-20230520-ctx4096.pth")), None),
    # "4w0.4B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-0.4B-v1-20230529-ctx4096.pth")), None),
    # "4w1.5B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth")), None),
    # "4w3B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-3B-v1-20230619-ctx4096.pth")), None),
    # "4w7B": (rwkv4, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-4-world", filename="RWKV-4-World-7B-v1-20230626-ctx4096.pth")), None),

    # "5w0.1B": (rwkv5, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-0.1B-v1-20230803-ctx4096.pth")), None),
    # "5w0.4B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-0.4B-v2-20231113-ctx4096.pth")), None),
    # "5w1.5B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-1B5-v2-20231025-ctx4096.pth")), None),
    # "5w3B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-3B-v2-20231113-ctx4096.pth")), None),
    # "5w7B": (rwkv5_2, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename="RWKV-5-World-7B-v2-20240128-ctx4096.pth")), None),

    # "6g0.1B": (rwkv6, GptTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/temp-latest-training-models", filename="temp/rwkv-x060-173m-pile-20240515-ctx4k.pth")), None),
    # "6w1.5B": (rwkv6, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-6-world", filename="RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth")), None),
    # "6w3B": (rwkv6, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-6-world", filename="RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth")), None),
    # "6w7B": (rwkv6, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-6-world", filename="RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth")), None),
    # "6w14B": (rwkv6, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-6-world", filename="RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth")), None),

    "7w0.1B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth")), None),
    "7w0.4B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")), None),
    "7w1.5B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth")), None),
    "7w3B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-world", filename="RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth")), None),

    "7n0.1B": (rwkv7, GptTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-pile", filename="RWKV-x070-Pile-168M-20241120-ctx4096.pth")), None),
    "7n0.4B": (rwkv7, GptTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-pile", filename="RWKV-x070-Pile-421M-20241127-ctx4096.pth")), None),
    "7n1.5B": (rwkv7, GptTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv-7-pile", filename="RWKV-x070-Pile-1.47B-20241210-ctx4096.pth")), None),

    "7g0.1B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1a-0.1b-20250728-ctx4096.pth")), None),
    "7g0.4B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1-0.4b-20250324-ctx4096.pth")), None),
    "7g1.5B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1-1.5b-20250429-ctx4096.pth")), None),
    "7g2.9B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g1-2.9b-20250519-ctx4096.pth")), None),
    "7g7B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g0-7.2b-20250722-ctx4096.pth")), None),
    "7g14B": (rwkv7, WorldTokenizer, (lambda : hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename="rwkv7-g0a3-13.3b-20251031-ctx4096.pth")), None),

    # "6q7B": (qrwkv6, QwenTokenizer, (lambda : AutoModelForCausalLM.from_pretrained("recursal/QRWKV6-7B-Base", trust_remote_code=True, dtype="auto")), (lambda: {"head_size": 128})),
    # "6q32B": (qrwkv6, QwenTokenizer, (lambda : AutoModelForCausalLM.from_pretrained("featherless-ai/QRWKV-QwQ-32B", trust_remote_code=True, dtype="auto")), (lambda: {"head_size": 128})),
}

# def get_rand_model(seed, version, n_layer, n_embd, vocab_size, config=None, dtype=None, rwkv_type="ScanRWKV", verbose=False):
#     rwkv = versions[version]
#     RWKV = getattr(rwkv, rwkv_type)
#     if dtype is None:
#         dtype = jnp.float32 if version.startswith('m') else jnp.bfloat16
#     elif isinstance(dtype, str):
#         dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
#     if verbose:
#         print(dtype)
    
#     model_name = f"{seed}_{version}_{n_layer}_{n_embd}_{vocab_size}"

#     path = Path(HF_HOME, "jaxrwkv_cache", f"{model_name}_{str(dtype.dtype)}.model")
#     if path.is_file():
#         if verbose:
#             print("loading from", path)
#         rwkv_params, config = load(path)
#     else:
#         key = jax.random.key(seed)
#         with jax.default_device(jax.local_devices(backend="cpu")[0]):
#             rwkv_params, config = RWKV.randomize_weights(key, n_layer, n_embd, vocab_size, config, dtype)
#         if verbose:
#             print("saving to", path)
#         save((rwkv_params, config), path)

#     return RWKV, rwkv_params, config

def get_model(model_name, dtype=None, rwkv_type="BaseRWKV", verbose=False, reload_cache=False):
    rwkv, tok_cls, model_name_fn, config_fn = models[model_name]
    RWKV = getattr(rwkv, rwkv_type)
    rwkv_tokenizer = tok_cls()

    if dtype is None:
        dtype = jnp.float32 if model_name.startswith('m') else jnp.bfloat16
    elif isinstance(dtype, str):
        dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
    if verbose:
        print(dtype)

    path = Path(HF_HOME, "hyperscalees_cache", f"{model_name}_{str(dtype.dtype)}.model")
    if path.is_file() and not reload_cache:
        if verbose:
            print("loading from", path)
        rwkv_full_params = load(path)
    else:
        import torch
        MODEL_NAME = model_name_fn()
        if isinstance(MODEL_NAME, torch.nn.Module):
            rwkv_full_params = MODEL_NAME.state_dict()
        else:
            rwkv_full_params = torch.load(MODEL_NAME, map_location='cpu', weights_only=True)
        config = config_fn() if config_fn is not None else None
        rwkv_full_params = RWKV.load_from_torch(rwkv_full_params, config, dtype=dtype)
        if verbose:
            print("saving to", path)
        save(rwkv_full_params, path, True)
    return RWKV, rwkv_full_params, rwkv_tokenizer

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "WorldTokenizer":
        return WorldTokenizer
    if tokenizer_name == "GptTokenizer":
        return GptTokenizer
    raise NotImplementedError(f"No such tokenizer {tokenizer_name}")
    

def save(model: any, path: str | Path, overwrite: bool = False):
    """
    Save the Any model as a file given a path.

    See https://github.com/google/jax/issues/2116#issuecomment-580322624

    :param model: The Any model you want to save
    :param path: The path to save the model to
    :param overwrite: Set to true to allow overwriting over existing file
    """
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load(path: str | Path) -> any:
    """
    Read the Any model from a file

    See https://github.com/google/jax/issues/2116#issuecomment-580322624

    :param path: The path to read the model from
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
