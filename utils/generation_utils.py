# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
import re
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import httpx
import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8-sig") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

def _normalize_text_api_mode(mode: str) -> str:
    """Normalize configured text endpoint mode to: responses | chat."""
    mode_norm = (mode or "").strip().lower()
    if mode_norm == "":
        return "chat"
    if mode_norm in ("chat", "chat.completions", "chat_completions"):
        return "chat"
    if mode_norm in ("responses", "response"):
        return "responses"
    print(f"[Warning] Unknown text API mode '{mode}', defaulting to 'chat'.")
    return "chat"


DEFAULT_OPENAI_COMPAT_BASE_URL = "https://api-slb.packyapi.com/v1/chat/completions"


def _normalize_openai_compat_base_url(base_url: str, default_base_url: str) -> str:
    """Normalize OpenAI-compatible base URL to a root path (typically .../v1)."""
    raw = (base_url or default_base_url or "").strip()
    if not raw:
        return ""

    normalized = raw.rstrip("/")
    lower = normalized.lower()
    if lower.endswith("/responses"):
        normalized = normalized[: -len("/responses")]
    elif lower.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]

    return normalized.rstrip("/")

# Initialize clients lazily or with robust defaults
gemini_client = None
anthropic_client = None
openai_client = None
openrouter_client = None
openrouter_api_key = ""
openrouter_base_url = _normalize_openai_compat_base_url(
    DEFAULT_OPENAI_COMPAT_BASE_URL,
    DEFAULT_OPENAI_COMPAT_BASE_URL,
)
openai_text_api_mode = "chat"
openrouter_text_api_mode = "chat"


def reinitialize_clients():
    """(Re)build all API clients from current env vars / config file.

    Called once at module load and can be called again at runtime
    (e.g. after the user sets new API keys via the Gradio UI).

    Returns a list of client names that were successfully initialized.
    """
    global gemini_client, anthropic_client, openai_client
    global openrouter_client, openrouter_api_key, openrouter_base_url
    global openai_text_api_mode, openrouter_text_api_mode

    initialized = []
    openai_text_api_mode = _normalize_text_api_mode(
        get_config_val("api_modes", "openai_text_api", "OPENAI_TEXT_API", "chat")
    )
    openrouter_text_api_mode = _normalize_text_api_mode(
        get_config_val("api_modes", "openrouter_text_api", "OPENROUTER_TEXT_API", "chat")
    )

    api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print("Initialized Gemini Client with API Key")
        initialized.append("Gemini")
    else:
        gemini_client = None

    key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
    if key:
        anthropic_client = AsyncAnthropic(api_key=key)
        print("Initialized Anthropic Client with API Key")
        initialized.append("Anthropic")
    else:
        anthropic_client = None

    key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
    if key:
        openai_base_url = _normalize_openai_compat_base_url(
            get_config_val("endpoints", "openai_base_url", "OPENAI_BASE_URL", DEFAULT_OPENAI_COMPAT_BASE_URL),
            DEFAULT_OPENAI_COMPAT_BASE_URL,
        )
        openai_kwargs = {"api_key": key}
        if openai_base_url:
            openai_kwargs["base_url"] = openai_base_url
        openai_client = AsyncOpenAI(**openai_kwargs)
        print(f"Initialized OpenAI Client with API Key (text mode: {openai_text_api_mode})")
        initialized.append("OpenAI")
    else:
        openai_client = None

    openrouter_api_key = get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
    if openrouter_api_key:
        openrouter_base_url = _normalize_openai_compat_base_url(
            get_config_val(
                "endpoints", "openrouter_base_url", "OPENROUTER_BASE_URL", DEFAULT_OPENAI_COMPAT_BASE_URL
            ),
            DEFAULT_OPENAI_COMPAT_BASE_URL,
        )
        openrouter_client = AsyncOpenAI(
            base_url=openrouter_base_url,
            api_key=openrouter_api_key,
        )
        print(
            f"Initialized OpenRouter Client with API Key "
            f"(base_url: {openrouter_base_url}, text mode: {openrouter_text_api_mode})"
        )
        initialized.append("OpenRouter")
    else:
        openrouter_client = None
        openrouter_base_url = _normalize_openai_compat_base_url(
            DEFAULT_OPENAI_COMPAT_BASE_URL,
            DEFAULT_OPENAI_COMPAT_BASE_URL,
        )

    return initialized


# Run once at import time (preserves original behaviour)
reinitialize_clients()



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["image_base64"]),
                        mime_type="image/jpeg",
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client was not initialized: missing Google API key. "
            "Please set GOOGLE_API_KEY in environment, or configure api_keys.google_api_key in configs/model_config.yaml."
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client

            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                    if part.text is not None
                ]
            result_list.extend([r for r in raw_response_list if r and r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    
    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    
    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


def _convert_to_responses_user_content(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert generic content list to Responses API user content blocks."""
    responses_content = []
    for item in contents:
        if item.get("type") == "text":
            text = item.get("text", "")
            if text:
                responses_content.append({"type": "input_text", "text": text})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                if data:
                    responses_content.append(
                        {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}
                    )
            elif "image_base64" in item:
                data = item.get("image_base64", "")
                if data:
                    responses_content.append(
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}"}
                    )
    return responses_content


def _build_responses_input(system_prompt: str, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build OpenAI Responses API input payload."""
    payload = []
    if system_prompt:
        payload.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        )

    user_content = _convert_to_responses_user_content(contents)
    if not user_content:
        user_content = [{"type": "input_text", "text": ""}]

    payload.append({"role": "user", "content": user_content})
    return payload


def _extract_text_from_responses(response: Any) -> str:
    """Extract text output from an OpenAI Responses API object."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    if isinstance(output_text, list):
        merged = "".join(x for x in output_text if isinstance(x, str))
        if merged.strip():
            return merged

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")

    texts = []
    for out_item in output or []:
        content = out_item.get("content") if isinstance(out_item, dict) else getattr(out_item, "content", None)
        for part in content or []:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type not in ("output_text", "text"):
                continue
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                texts.append(text)

    return "".join(texts)


def _responses_unsupported_error(exc: Exception) -> bool:
    """Heuristic to detect endpoints that do not support /responses."""
    msg = str(exc).lower()
    tokens = [
        "/responses",
        "responses",
        "unsupported",
        "not found",
        "unknown url",
        "unrecognized request",
        "no such endpoint",
        "method not allowed",
    ]
    return any(t in msg for t in tokens)


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []
    use_responses = openai_text_api_mode != "chat"

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            if use_responses:
                responses_input = _build_responses_input(system_prompt, current_contents)
                first_response = await openai_client.responses.create(
                    model=model_name,
                    input=responses_input,
                    temperature=temperature,
                    max_output_tokens=max_completion_tokens,
                )
                content = _extract_text_from_responses(first_response)
            else:
                openai_contents = _convert_to_openai_format(current_contents)
                first_response = await openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": openai_contents}
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                content = first_response.choices[0].message.content or ""

            if not content.strip():
                print(f"OpenAI returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            if use_responses and _responses_unsupported_error(e):
                print("OpenAI /responses endpoint unavailable; falling back to chat.completions.")
                use_responses = False
                continue
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        if use_responses:
            valid_input = _build_responses_input(system_prompt, current_contents)
            tasks = [
                openai_client.responses.create(
                    model=model_name,
                    input=valid_input,
                    temperature=temperature,
                    max_output_tokens=max_completion_tokens,
                )
                for _ in range(remaining_candidates)
            ]
        else:
            valid_openai_contents = _convert_to_openai_format(current_contents)
            tasks = [
                openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": valid_openai_contents}
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                for _ in range(remaining_candidates)
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                if use_responses:
                    text = _extract_text_from_responses(res)
                    response_text_list.append(text or "Error")
                else:
                    response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_openrouter_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter API (OpenAI-compatible) with asynchronous retry logic.
    """
    if openrouter_client is None:
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key. "
            "Please set OPENROUTER_API_KEY in environment, or configure "
            "api_keys.openrouter_api_key in configs/model_config.yaml."
        )

    model_name = _to_openrouter_model_id(model_name)
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []
    use_responses = openrouter_text_api_mode != "chat"

    current_contents = contents

    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            if use_responses:
                responses_input = _build_responses_input(system_prompt, current_contents)
                first_response = await openrouter_client.responses.create(
                    model=model_name,
                    input=responses_input,
                    temperature=temperature,
                    max_output_tokens=max_completion_tokens,
                )
                content = _extract_text_from_responses(first_response)
            else:
                openai_contents = _convert_to_openai_format(current_contents)
                first_response = await openrouter_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": openai_contents},
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                content = first_response.choices[0].message.content or ""

            if not content.strip():
                print(f"OpenRouter returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break

        except Exception as e:
            if use_responses and _responses_unsupported_error(e):
                print("OpenRouter /responses endpoint unavailable; falling back to chat.completions.")
                use_responses = False
                continue
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"OpenRouter attempt {attempt + 1} failed{context_msg}: {error_str}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(f"Error: All {max_attempts} OpenRouter attempts failed{context_msg}.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        if use_responses:
            valid_input = _build_responses_input(system_prompt, current_contents)
            tasks = [
                openrouter_client.responses.create(
                    model=model_name,
                    input=valid_input,
                    temperature=temperature,
                    max_output_tokens=max_completion_tokens,
                )
                for _ in range(remaining_candidates)
            ]
        else:
            valid_openai_contents = _convert_to_openai_format(current_contents)
            tasks = [
                openrouter_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": valid_openai_contents},
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )
                for _ in range(remaining_candidates)
            ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent OpenRouter candidate: {res}")
                response_text_list.append("Error")
            else:
                if use_responses:
                    text = _extract_text_from_responses(res)
                    response_text_list.append(text or "Error")
                else:
                    response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


def _extract_b64_from_data_url(value: str) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    # Direct data URL
    if text.startswith("data:image") and "," in text:
        return text.split(",", 1)[1].strip()

    # Markdown-wrapped image, e.g. ![image](data:image/png;base64,...)
    m = re.search(r"data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=\r\n]+)", text)
    if m:
        return m.group(1).replace("\n", "").replace("\r", "").strip()
    return ""


def _extract_image_ref_from_obj(obj: Any) -> tuple[str, str]:
    """Return first image ref found in obj: ("b64", data) | ("url", http_url) | ("", "")."""
    if obj is None:
        return "", ""

    if isinstance(obj, str):
        b64 = _extract_b64_from_data_url(obj)
        if b64:
            return "b64", b64
        if obj.startswith("http://") or obj.startswith("https://"):
            return "url", obj
        return "", ""

    if isinstance(obj, list):
        for item in obj:
            t, v = _extract_image_ref_from_obj(item)
            if t:
                return t, v
        return "", ""

    if not isinstance(obj, dict):
        return "", ""

    for key in ("b64_json", "image_base64", "base64", "b64"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return "b64", val.strip()

    inline_data = obj.get("inline_data")
    if isinstance(inline_data, dict):
        data = inline_data.get("data")
        if isinstance(data, str) and data.strip():
            return "b64", data.strip()

    image_url = obj.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str) and url.strip():
            b64 = _extract_b64_from_data_url(url)
            if b64:
                return "b64", b64
            if url.startswith("http://") or url.startswith("https://"):
                return "url", url
    elif isinstance(image_url, str) and image_url.strip():
        b64 = _extract_b64_from_data_url(image_url)
        if b64:
            return "b64", b64
        if image_url.startswith("http://") or image_url.startswith("https://"):
            return "url", image_url

    for key in ("url", "data"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            b64 = _extract_b64_from_data_url(val)
            if b64:
                return "b64", b64
            if key == "url" and (val.startswith("http://") or val.startswith("https://")):
                return "url", val

    for key in ("images", "content", "message", "delta", "output", "parts"):
        if key in obj:
            t, v = _extract_image_ref_from_obj(obj.get(key))
            if t:
                return t, v

    for val in obj.values():
        t, v = _extract_image_ref_from_obj(val)
        if t:
            return t, v

    return "", ""


def _extract_image_ref_from_chat_response(data: Dict[str, Any]) -> tuple[str, str]:
    """Extract image from chat/non-chat response payload."""
    if not isinstance(data, dict):
        return "", ""

    choices = data.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            t, v = _extract_image_ref_from_obj(ch)
            if t:
                return t, v

    if "output" in data:
        t, v = _extract_image_ref_from_obj(data.get("output"))
        if t:
            return t, v

    return "", ""


async def _download_image_url_as_b64(url: str) -> str:
    if not url:
        return ""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        if not resp.content:
            return ""
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        print(f"[Warning]: Failed to download image URL returned by gateway: {e}")
        return ""


async def _stream_openrouter_image_b64(
    url: str, headers: Dict[str, str], payload: Dict[str, Any]
) -> str:
    """Try streaming chat completion and return first image base64 found."""
    stream_payload = dict(payload)
    stream_payload["stream"] = True

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", url, headers=headers, json=stream_payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    break
                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                ref_type, ref_val = _extract_image_ref_from_chat_response(evt)
                if ref_type == "b64" and ref_val:
                    return ref_val
                if ref_type == "url" and ref_val:
                    b64 = await _download_image_url_as_b64(ref_val)
                    if b64:
                        return b64
    return ""


async def _safe_http_error_detail(e: httpx.HTTPStatusError, max_chars: int = 1200) -> str:
    """Safely read HTTP error body without triggering ResponseNotRead."""
    resp = e.response
    if resp is None:
        return "<no response body>"

    try:
        raw = await resp.aread()
        if not raw:
            return "<empty response body>"
        text = raw.decode("utf-8", errors="replace")
    except Exception as read_err:
        # Fallback for already-buffered non-stream responses.
        try:
            text = resp.text
        except Exception:
            return f"<response body unavailable: {read_err}>"

    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "...(truncated)"
    return text or "<empty response body>"


async def call_openrouter_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter image generation via chat completions.
    Supports streaming-first parsing (when available) and non-stream fallback.
    """
    if not openrouter_api_key:
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key."
        )

    system_prompt = config.get("system_prompt", "")
    temperature = config.get("temperature", 1.0)
    aspect_ratio = config.get("aspect_ratio", "1:1")
    image_size = config.get("image_size", "1K")
    use_stream = bool(config.get("stream", True))

    model_name = _to_openrouter_model_id(model_name)
    openai_contents = _convert_to_openai_format(contents)

    image_config = {}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": openai_contents},
        ],
        "temperature": temperature,
        "modalities": ["image", "text"],
    }
    if image_config:
        payload["image_config"] = image_config

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }
    post_url = f"{openrouter_base_url}/chat/completions"

    for attempt in range(max_attempts):
        try:
            if use_stream:
                try:
                    b64_stream = await _stream_openrouter_image_b64(post_url, headers, payload)
                    if b64_stream:
                        return [b64_stream]
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code if e.response is not None else None
                    if status_code in (400, 404, 405, 422):
                        print("[Info]: Streaming image endpoint unsupported; falling back to non-stream response.")
                        use_stream = False
                    elif status_code is not None and status_code >= 500:
                        print(
                            f"[Warning]: Streaming image endpoint returned HTTP {status_code}; "
                            "falling back to non-stream response."
                        )
                        use_stream = False
                    else:
                        raise
                except Exception as e:
                    print(f"[Warning]: Streaming parse failed, falling back to non-stream: {e}")

            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    post_url,
                    headers=headers,
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                print(f"[Warning]: OpenRouter image generation returned no choices, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

            ref_type, ref_val = _extract_image_ref_from_chat_response(data)
            if ref_type == "b64" and ref_val:
                return [ref_val]
            if ref_type == "url" and ref_val:
                b64_data = await _download_image_url_as_b64(ref_val)
                if b64_data:
                    return [b64_data]

            if attempt == 0:
                top_keys = list(data.keys()) if isinstance(data, dict) else []
                msg_keys = []
                try:
                    msg_obj = data.get("choices", [{}])[0].get("message", {})
                    if isinstance(msg_obj, dict):
                        msg_keys = list(msg_obj.keys())
                except Exception:
                    pass
                print(
                    f"[Debug]: No image parsed from gateway response. "
                    f"top_keys={top_keys}, message_keys={msg_keys}"
                )

            print(f"[Warning]: OpenRouter image generation returned no images, retrying...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            continue

        except httpx.HTTPStatusError as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            status_code = e.response.status_code if e.response is not None else "unknown"
            error_detail = await _safe_http_error_detail(e)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: "
                f"HTTP {status_code} - {error_detail}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]

def _to_openrouter_model_id(model_name: str) -> str:
    """Normalize model IDs for OpenAI-compatible routers.

    Behaviour:
    - If model already contains "/", keep it unchanged.
    - If using the official openrouter.ai endpoint and model starts with "gemini",
      auto-prefix to "google/<model>" for compatibility.
    - For custom gateways (e.g. Packy), keep bare model names unchanged.
    """
    if "/" in model_name:
        return model_name

    base = (openrouter_base_url or "").lower()
    if "openrouter.ai" in base and model_name.startswith("gemini"):
        return f"google/{model_name}"
    return model_name


async def call_model_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    Unified router that dispatches to the correct provider based on model_name.

    Routing rules:
      1. Explicit prefix overrides: "openrouter/" -> OpenRouter, "claude-" -> Anthropic,
         "gpt-"/"o1-"/"o3-"/"o4-" -> OpenAI
      2. No prefix: auto-detect based on which API key is configured.
         Priority: OpenRouter > Gemini > Anthropic > OpenAI
    """
    # Explicit provider prefix overrides auto-detection
    if model_name.startswith("openrouter/"):
        provider = "openrouter"
        actual_model = model_name[len("openrouter/"):]
    elif model_name.startswith("claude-"):
        provider = "anthropic"
        actual_model = model_name
    elif any(model_name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        provider = "openai"
        actual_model = model_name
    else:
        # Auto-detect provider based on which API key is configured
        actual_model = model_name
        if openrouter_client is not None:
            provider = "openrouter"
            actual_model = _to_openrouter_model_id(model_name)
        elif gemini_client is not None:
            provider = "gemini"
        elif anthropic_client is not None:
            provider = "anthropic"
        elif openai_client is not None:
            provider = "openai"
        else:
            raise RuntimeError(
                "No API client available. Please configure at least one API key "
                "in configs/model_config.yaml or via environment variables."
            )

    if provider == "gemini":
        return await call_gemini_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    # Convert Gemini GenerateContentConfig -> dict for OpenAI/Claude/OpenRouter
    cfg_dict = {
        "system_prompt": config.system_instruction if hasattr(config, "system_instruction") else "",
        "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
        "candidate_num": config.candidate_count if hasattr(config, "candidate_count") else 1,
        "max_completion_tokens": config.max_output_tokens if hasattr(config, "max_output_tokens") else 50000,
    }

    call_fn = {
        "openrouter": call_openrouter_with_retry_async,
        "anthropic": call_claude_with_retry_async,
        "openai": call_openai_with_retry_async,
    }[provider]

    return await call_fn(
        model_name=actual_model,
        contents=contents,
        config=cfg_dict,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )
