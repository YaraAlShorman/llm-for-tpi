"""
Qwen3 Inference Server
OpenAI-compatible API with streaming support.
"""

import os
import json
import uuid
import time
import asyncio
from threading import Thread
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-14B")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading model: {MODEL_ID}")
print(f"Device: {DEVICE} | Dtype: {DTYPE}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully.")

# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=MAX_MODEL_LEN)
    stream: bool = False
    enable_thinking: Optional[bool] = None

class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=MAX_MODEL_LEN)
    stream: bool = False

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3 Inference Server")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


def _build_chat_input(messages: list[Message], enable_thinking: Optional[bool]):
    """Apply the chat template and return input_ids on the correct device."""
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    text = tokenizer.apply_chat_template(
        msg_dicts,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    return tokenizer([text], return_tensors="pt").to(model.device)


# ---- Chat completions (non-streaming) ------------------------------------
def _generate_sync(inputs, temperature, top_p, max_tokens):
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        top_p=top_p,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    # Trim the prompt tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    inputs = _build_chat_input(req.messages, req.enable_thinking)

    if req.stream:
        return StreamingResponse(
            _stream_chat(inputs, req),
            media_type="text/event-stream",
        )

    text = await asyncio.to_thread(
        _generate_sync, inputs, req.temperature, req.top_p, req.max_tokens
    )

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[-1],
            "completion_tokens": len(tokenizer.encode(text)),
            "total_tokens": inputs["input_ids"].shape[-1] + len(tokenizer.encode(text)),
        },
    }


# ---- Chat completions (streaming) ----------------------------------------
async def _stream_chat(inputs, req: ChatRequest):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        top_p=req.top_p,
        streamer=streamer,
    )
    if req.temperature > 0:
        gen_kwargs["temperature"] = req.temperature

    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    for token_text in streamer:
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


# ---- Text completions -----------------------------------------------------
@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    inputs = tokenizer([req.prompt], return_tensors="pt").to(model.device)
    text = await asyncio.to_thread(
        _generate_sync, inputs, req.temperature, req.top_p, req.max_tokens
    )
    resp_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": resp_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "text": text, "finish_reason": "stop"}
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
