import asyncio
import multiprocessing as mp
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from vllm import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

# hardcoded demo configuration
HARDCODED_CONFIG = {
    "model_name": "willcb/Qwen3-1.7B",
    "dtype": "auto",
    "kv_cache_dtype": "auto",
    "max_model_len": 4096,
    "quantization": None,
    "enforce_eager": False,
    "device": "auto",
    "tensor_parallel_size": 1,
}


# request/response models following OpenAI API schema
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    # vLLM specific parameters
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    # vLLM specific parameters
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]
    extra: Dict[str, Any] = {"proof": b""}


class WeightUpdateRequest(BaseModel):
    path: str


class WeightUpdateResponse(BaseModel):
    status: str = "success"


# Global variables for engine
engine: Optional[AsyncLLM] = None
model_name: str = ""


async def initialize_engine():
    """Initialize the vLLM AsyncLLM engine."""
    global engine, model_name

    # Create engine args from hardcoded config
    engine_args = AsyncEngineArgs(
        model=HARDCODED_CONFIG["model_name"],
        dtype=HARDCODED_CONFIG["dtype"],
        kv_cache_dtype=HARDCODED_CONFIG["kv_cache_dtype"],
        max_model_len=HARDCODED_CONFIG["max_model_len"],
        quantization=HARDCODED_CONFIG["quantization"],
        enforce_eager=HARDCODED_CONFIG["enforce_eager"],
        device=HARDCODED_CONFIG["device"],
        tensor_parallel_size=HARDCODED_CONFIG["tensor_parallel_size"],
    )

    # Initialize AsyncLLM
    engine = AsyncLLM.from_engine_args(engine_args)
    model_name = HARDCODED_CONFIG["model_name"]

    print(f"Engine initialized with model: {model_name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    await initialize_engine()
    yield
    # Shutdown
    if engine is not None:
        # Add any cleanup code here if needed
        pass


app = FastAPI(title="vLLM OpenAI-Compatible Server", lifespan=lifespan)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Handle completion requests."""
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "Engine not initialized"})

    # Convert request to vLLM sampling params
    sampling_params = SamplingParams(
        n=request.n or 1,
        temperature=request.temperature or 1.0,
        top_p=request.top_p or 1.0,
        top_k=request.top_k or -1,
        min_p=request.min_p or 0.0,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0,
        repetition_penalty=request.repetition_penalty or 1.0,
        max_tokens=request.max_tokens,
        stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
        logprobs=request.logprobs,
        include_stop_str_in_output=False,
    )

    # Handle single or multiple prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    request_id = f"cmpl-{uuid.uuid4().hex}"

    # Non-streaming response
    all_outputs = []
    for prompt in prompts:
        final_output = None
        async for output in engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=f"{request_id}-{prompts.index(prompt)}"
        ):
            if output.finished:
                final_output = output
        if final_output:
            all_outputs.append(final_output)

    # Format response
    choices = []
    total_tokens = 0
    n = request.n or 1
    for i, output in enumerate(all_outputs):
        for j, completion_output in enumerate(output.outputs):
            choice = CompletionChoice(
                text=completion_output.text,
                index=i * n + j,
                finish_reason=completion_output.finish_reason,
                logprobs=None,  # TODO: format logprobs if needed
            )
            choices.append(choice)
            total_tokens += len(completion_output.token_ids)

    response = CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model or model_name,
        choices=choices,
        usage={
            "prompt_tokens": sum(len(output.prompt_token_ids) if output.prompt_token_ids else 0 for output in all_outputs),
            "completion_tokens": total_tokens,
            "total_tokens": sum(len(output.prompt_token_ids) if output.prompt_token_ids else 0 for output in all_outputs) + total_tokens,
        },
    )
    return response


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    if engine is None:
        return JSONResponse(status_code=500, content={"error": "Engine not initialized"})

    # Get tokenizer and apply chat template
    tokenizer = await engine.get_tokenizer()

    # Convert messages to the format expected by tokenizer
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Apply chat template
    prompt_result = tokenizer.apply_chat_template(
        messages,  # type: ignore
        tokenize=False,
        add_generation_prompt=True,
    )

    # Since tokenize=False, this should be a string, but ensure it
    prompt = str(prompt_result)

    # Convert to sampling params
    sampling_params = SamplingParams(
        n=request.n or 1,
        temperature=request.temperature or 1.0,
        top_p=request.top_p or 1.0,
        top_k=request.top_k or -1,
        min_p=request.min_p or 0.0,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0,
        repetition_penalty=request.repetition_penalty or 1.0,
        max_tokens=request.max_tokens,
        stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
        logprobs=request.top_logprobs if request.logprobs else None,
        include_stop_str_in_output=False,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Non-streaming response - collect all outputs
    final_output = None
    async for output in engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
        if output.finished:
            final_output = output

    if final_output is None:
        return JSONResponse(status_code=500, content={"error": "No output generated"})

    choices = []
    for i, completion_output in enumerate(final_output.outputs):
        choice = ChatCompletionChoice(
            index=i, message=ChatCompletionMessage(content=completion_output.text), finish_reason=completion_output.finish_reason
        )
        choices.append(choice)

    response = ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model or model_name,
        choices=choices,
        usage={
            "prompt_tokens": len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0,
            "completion_tokens": sum(len(out.token_ids) for out in final_output.outputs),
            "total_tokens": (len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0)
            + sum(len(out.token_ids) for out in final_output.outputs),
        },
    )
    return response


_reload_lock = asyncio.Lock()


async def reload_model_weights(engine: AsyncLLM, path: str):
    async with _reload_lock:
        await engine.sleep()
        try:
            loop = asyncio.get_running_loop()
            # TODO: update weights
        finally:
            await engine.wake_up()


@app.post("/update_weights")
async def update_weights(request: WeightUpdateRequest):
    path = request.path
    if path == "":
        return JSONResponse(status_code=400, content={"error": "Path is empty"})

    if engine is None:
        return JSONResponse(status_code=500, content={"error": "Engine not initialized"})

    await reload_model_weights(engine, path)
    return JSONResponse(status_code=200, content={"status": "success"})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if engine is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "Engine not initialized"})
    return {"status": "healthy"}


if __name__ == "__main__":
    mp.set_start_method("spawn")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
