import os
import re
import time
import uuid
import base64
from io import BytesIO
from PIL import Image

import runpod
from runpod.serverless import log
from runpod.serverless.utils.rp_validator import validate
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from juno.schema import VALIDATIONS

MODEL = os.getenv("MODEL_NAME")
model = None
processor = None

def clean_repeated_substrings(text):
    """Clean repeated substrings in text - required for HunyuanOCR output stability"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]  
    return text

def handler(job):
    input_validation = validate(job["input"], VALIDATIONS)
    if "errors" in input_validation:
        return {"error": {"type": "validation_error", "details": input_validation["errors"]}}
    
    job_input = input_validation["validated_input"]
    messages_input = job_input.get("messages")
    
    vllm_inputs = []

    if messages_input:
        for msg in messages_input:
            user_text = msg.get("prompt")
            mm_data = msg.get("multi_modal_data", {})
            image_b64 = mm_data.get("image")
            
            pil_images = []
            
            content = []
            if image_b64:
                try:
                    image_bytes = base64.b64decode(image_b64)
                    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    pil_images.append(pil_img)
                    # Note: We pass a placeholder here for the processor to replace
                    content.append({"type": "image", "image": "placeholder"})
                except Exception as e:
                    log.error(f"Image decode failed: {e}")

            if user_text:
                content.append({"type": "text", "text": user_text})

            # 2. Apply Processor Chat Template (Matching official tutorial)
            template_msgs = [
                {"role": "system", "content": ""},
                {"role": "user", "content": content}
            ]
            
            prompt_string = processor.apply_chat_template(
                template_msgs, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 3. Construct the vLLM dict
            vllm_inputs.append({
                "prompt": prompt_string,
                "multi_modal_data": {"image": pil_images}
            })

    sampler = SamplingParams(temperature=0, max_tokens=16384)
    model_outputs = model.generate(vllm_inputs, sampler)

    # Process first output
    result = model_outputs[0]
    raw_text = result.outputs[0].text
    
    # Apply official cleaning function
    final_text = clean_repeated_substrings(raw_text)

    return {
        "id": os.getenv("RUNPOD_REQUEST_ID") or f"rp-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_text
                },
                "finish_reason": result.outputs[0].finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(result.outputs[0].token_ids),
            "total_tokens": len(result.prompt_token_ids) + len(result.outputs[0].token_ids),
        },
    }

if __name__ == "__main__":
    log.info(f"Loading HunyuanOCR from {MODEL}...")
    
    model = LLM(
        model=MODEL,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    runpod.serverless.start({"handler": handler})