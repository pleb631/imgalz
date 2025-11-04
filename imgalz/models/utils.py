
import json
import inspect
import os
from functools import wraps
from huggingface_hub import hf_hub_download
def auto_download(category, local_cache_dir="./ckpt"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mapping_json_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../cfg/mapping.json")
            )

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            model_key = bound_args.arguments.get("model_path")
            if model_key is None:
                raise ValueError("model_path is not provided")

            if os.path.exists(model_key):
                return func(*args, **kwargs)
            model_key = Path(model_key).stem

            with open(mapping_json_path, "r", encoding="utf-8") as f:
                mapping = json.load(f).get(category, {})

            if model_key not in mapping:
                raise ValueError(
                    f"model key '{model_key}' is not found in the mapping of {category}"
                )

            hf_info = mapping[model_key]
            repo_id = hf_info["repo_id"]
            filename = hf_info["filename"]

            os.makedirs(local_cache_dir, exist_ok=True)

            local_path = os.path.join(local_cache_dir, filename)
            if not os.path.exists(local_path):
                print(f"Downloading {filename} from {repo_id}...")
                hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=local_cache_dir
                )
            else:
                print(f"Found existing model file: {local_path}")

            bound_args.arguments["model_path"] = local_path
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
