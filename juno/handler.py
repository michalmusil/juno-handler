import os
import re
import sys
import time
import uuid

import base64
from io import BytesIO
from PIL import Image

import runpod
from runpod.serverless import log
from runpod.serverless.utils.rp_validator import validate
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

from juno.schema import VALIDATIONS

MODEL = os.getenv("MODEL_NAME")
# DTYPE = os.getenv("MODEL_DTYPE")
# QUANTIZATION = os.getenv("MODEL_QUANTIZATION")
# TRUST_REMOTE_CODE = os.getenv("MODEL_TRUST_REMOTE_CODE", "").lower() in ("true", "1", "yes")
# TOKENIZER = os.getenv("MODEL_TOKENIZER")
# CONFIG_FORMAT = os.getenv("MODEL_CONFIG_FORMAT")
# LOAD_FORMAT = os.getenv("MODEL_LOAD_FORMAT")

# MAX_MODEL_LEN = int(os.getenv("MODEL_MAX_LEN")) if os.getenv("MODEL_MAX_LEN") else None
# MAX_NUM_SEQS = int(os.getenv("MODEL_MAX_NUM_SEQS")) if os.getenv("MODEL_MAX_NUM_SEQS") else None
# DISTRIBUTED_EXECUTOR_BACKEND = os.getenv("DISTRIBUTED_EXECUTOR_BACKEND")

# DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE") or "0.15")
# DEFAULT_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS") or "32768")
# DEFAULT_TOP_P = float(os.getenv("MODEL_TOP_P") or "0.95")

model = None


def handler(job):
    input_validation = validate(job["input"], VALIDATIONS)

    if "errors" in input_validation:
        return {
            "error": {
                "type": "validation_error",
                "message": "Invalid input",
                "details": input_validation["errors"],
            }
        }
    job_input = input_validation["validated_input"]

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    temperature = job_input.get("temperature")
    max_tokens = job_input.get("max_tokens")
    top_p = job_input.get("top_p")

    if messages and prompt:
        return {
            "error": {
                "type": "validation_error",
                "message": "Provide either 'messages' or 'prompt', not both",
            }
        }

    if not messages and not prompt:
        return {
            "error": {
                "type": "validation_error",
                "message": "Either 'messages' or 'prompt' is required",
            }
        }
    input = None
    if prompt:
        input = [{"prompt": prompt}]

    if messages:
        input = []
        for msg in messages:
            # 1. Get the base64 string from the nested dictionary
            mm_data = msg.get("multi_modal_data", {})
            image_b64 = mm_data.get("image")

            if image_b64:
                try:
                    # 2. Decode the base64 string
                    image_bytes = base64.b64decode(image_b64)

                    # 3. Create PIL Image and convert to RGB
                    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")

                    # 4. Update the dictionary with the actual PIL object
                    msg["multi_modal_data"]["image"] = pil_img
                except Exception as e:
                    log.error(f"Failed to decode image: {e}")
                    # You might want to return an error response here

        input.append(msg)

    sampler = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )

    model_output = model.generate(input, sampler)

    result = model_output[0]
    output = result.outputs[0]

    text = output.text
    reasoning_content = None

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    message = {
        "role": "assistant",
        "reasoning_content": reasoning_content,
        "content": text,
    }

    if hasattr(output, "tool_calls") and output.tool_calls:
        message["tool_calls"] = output.tool_calls

    return {
        "id": os.getenv("RUNPOD_REQUEST_ID") or f"rp-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": output.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(output.token_ids),
            "total_tokens": len(result.prompt_token_ids) + len(output.token_ids),
        },
    }


if __name__ == "__main__":

    log.info("Loading...")

    model = LLM(
        model=MODEL,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    runpod.serverless.start({"handler": handler})
