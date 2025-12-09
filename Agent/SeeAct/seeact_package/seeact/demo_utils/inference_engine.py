# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time

import backoff
import openai
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
)
import requests
from dotenv import load_dotenv
import litellm
from litellm.exceptions import BadRequestError, OpenAIError
import base64

# Optional verbose tracing for LiteLLM requests. Enable with
# SEEACT_LITELLM_VERBOSE=true to print request metadata and let litellm log.
DEBUG_LITELLM = os.getenv("SEEACT_LITELLM_VERBOSE", "").lower() in ("1", "true", "yes", "y", "on")
if DEBUG_LITELLM:
    try:
        litellm.set_verbose(True)
    except Exception:
        pass

EMPTY_API_KEY="Your API KEY Here"

def load_openai_api_key():
    load_dotenv()
    assert (
            os.getenv("OPENAI_API_KEY") is not None and
            os.getenv("OPENAI_API_KEY") != EMPTY_API_KEY
    ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
    return os.getenv("OPENAI_API_KEY")


def load_gemini_api_key():
    load_dotenv()
    assert (
            os.getenv("GEMINI_API_KEY") is not None and
            os.getenv("GEMINI_API_KEY") != EMPTY_API_KEY
    ), "must pass on the api_key or set GEMINI_API_KEY in the environment"
    return os.getenv("GEMINI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def engine_factory(api_key=None, model=None, **kwargs):
    """
    Hardcoded to LiteLLM engine: all models route through LiteLLMEngine so you
    can test any provider/model string supported by litellm.
    """
    if model is None:
        raise Exception("No model provided to engine_factory")
    return LiteLLMEngine(model=model, **kwargs)

class Engine:
    def __init__(
            self,
            stop=["\n\n"],
            rate_limit=-1,
            model=None,
            temperature=0,
            **kwargs,
    ) -> None:
        """
            Base class to init an engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        self.time_slots = [0]
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.time_slots)
        self.current_key_idx = 0
        print(f"Initializing model {self.model}")        

    def tokenize(self, input):
        return self.tokenizer(input)


class LiteLLMEngine(Engine):
    """
    Generic Litellm-powered engine; supports any provider/model string that
    litellm understands, including vision models when an image is supplied.
    """

    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None,
                 image_path=None, ouput_0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)

        prompt0, prompt1, prompt2 = prompt if prompt else ("", "", "")
        target_model = model if model else self.model

        # Handle different providers with their preferred message formats
        if target_model.startswith("ollama/"):
            # Ollama format - images in separate "images" field
            base64_image = encode_image(image_path) if image_path else None
            if turn_number == 0:
                prompt_input = [
                    {"role": "assistant", "content": prompt0},
                    {"role": "user", "content": prompt1, "images": [base64_image] if base64_image else []},
                ]
            else:
                prompt_input = [
                    {"role": "assistant", "content": prompt0},
                    {"role": "user", "content": prompt1, "images": [base64_image] if base64_image else []},
                    {"role": "assistant", "content": f"\n\n{ouput_0}"},
                    {"role": "user", "content": prompt2},
                ]
        else:
            # OpenAI-style format for vision models
            base64_image = encode_image(image_path) if image_path else None
            user_content = [{"type": "text", "text": prompt1}]
            if base64_image:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                )

            if turn_number == 0:
                prompt_input = [
                    {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                    {"role": "user", "content": user_content},
                ]
            else:
                prompt_input = [
                    {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt2}]},
                ]

        token_value = max_new_tokens if max_new_tokens else 4096
        base_kwargs = {
            "model": target_model,
            "messages": prompt_input,
            **kwargs,
        }
        temp_value = self.temperature if temperature is None else temperature
        # GPT-5 family currently only supports the default temperature (1) and
        # rejects custom temperature values. Drop temperature for these models.
        if target_model.startswith("gpt-5"):
            temp_value = None

        # GPT-4.1 uses max_output_tokens; GPT-5-mini currently rejects it, so
        # keep the new param only for 4.1 variants.
        uses_new_output_param = (
            target_model.startswith("gpt-4.1") or "gpt-4.1" in target_model
        )
        if target_model.startswith("gpt-5"):
            token_param_candidates = [
                ("max_completion_tokens", token_value),
            ]
        else:
            token_param_candidates = (
                [
                    ("max_output_tokens", token_value),
                    ("max_completion_tokens", token_value),
                    ("max_tokens", token_value),
                ]
                if uses_new_output_param
                else [
                    ("max_tokens", token_value),
                    ("max_completion_tokens", token_value),
                ]
            )
        if target_model.startswith("gpt-5"):
            # GPT-5 only accepts the default temperature; omit the param.
            temperature_candidates = [None]
        else:
            temperature_candidates = [temp_value]
            if temp_value is not None:
                temperature_candidates.append(None)  # retry without temperature if needed

        last_error = None
        for token_param_name, token_param_val in token_param_candidates:
            for temp_candidate in temperature_candidates:
                completion_kwargs = dict(base_kwargs)
                if temp_candidate is not None:
                    completion_kwargs["temperature"] = temp_candidate
                if DEBUG_LITELLM:
                    debug_payload = {
                        "model": completion_kwargs.get("model"),
                        "token_param": token_param_name,
                        "token_val": token_param_val,
                        "temperature": completion_kwargs.get("temperature", "(default)"),
                        "messages_count": len(completion_kwargs.get("messages", [])),
                        "has_image": bool(base64_image),
                    }
                    print("[SeeAct][LiteLLM] request", debug_payload)
                try:
                    response = litellm.completion(**completion_kwargs, **{token_param_name: token_param_val})
                    last_error = None
                    break
                except (BadRequestError, OpenAIError) as e:
                    last_error = e
                    msg = str(e).lower()
                    if DEBUG_LITELLM:
                        print("[SeeAct][LiteLLM] error", {"token_param": token_param_name, "temperature": temp_candidate, "error": msg})
                    # If the error is unrelated to token/temperature, propagate immediately.
                    if (
                        "max_tokens" not in msg
                        and "max_completion_tokens" not in msg
                        and "max_output_tokens" not in msg
                        and "temperature" not in msg
                    ):
                        raise
            if last_error is None:
                break
        # Final minimal attempt: drop token/temperature entirely to satisfy
        # newer models with strict param requirements (e.g., GPT-5).
        if last_error:
            try:
                minimal_kwargs = dict(base_kwargs)
                if DEBUG_LITELLM:
                    print("[SeeAct][LiteLLM] request (minimal)", {
                        "model": minimal_kwargs.get("model"),
                        "messages_count": len(minimal_kwargs.get("messages", [])),
                        "has_image": bool(base64_image),
                    })
                response = litellm.completion(**minimal_kwargs)
                last_error = None
            except (BadRequestError, OpenAIError) as e:
                last_error = e
        if last_error:
            raise last_error

        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )

        return [choice["message"]["content"] for choice in response.choices][0]


class OllamaEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init an Ollama engine
            To use Ollama, dowload and install Ollama from https://ollama.com/
            After Ollama start, pull llava with command: ollama pull llava
        """
        super().__init__(**kwargs)
        self.api_url = "http://localhost:11434/api/chat"


    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            wait_time = self.next_avil_time[self.current_key_idx] - start_time
            print(f"Wait {wait_time} for rate limitting")
            time.sleep(wait_time)
        prompt0, prompt1, prompt2 = prompt

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                {"role": "user", "content": prompt1, "images": [f"{base64_image}"]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                {"role": "user", "content": prompt1, "images": [f"{base64_image}"]},
                {"role": "assistant", "content": f"\n\n{ouput_0}"},
                {"role": "user", "content": prompt2}, 
            ]

        options = {"temperature": self.temperature, "num_predict": max_new_tokens}
        data = {
            "model": self.model,
            "messages": prompt_input,
            "options": options,
            "stream": False,
        }
        _request = {
            "url": f"{self.api_url}",
            "json": data,
        }
        response = requests.post(**_request)  # type: ignore
        if response.status_code != 200:
            raise Exception(f"Ollama API Error: {response.status_code}, {response.text}")
        response_json = response.json()
        return response_json["message"]["content"]


class GeminiEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init a Gemini engine
            To use this engine, please provide the GEMINI_API_KEY in the environment
            Supported Model             Rate Limit
            gemini-1.5-pro-latest    	2 queries per minute, 1000 queries per day
        """
        super().__init__(**kwargs)


    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            wait_time = self.next_avil_time[self.current_key_idx] - start_time
            print(f"Wait {wait_time} for rate limitting")
        prompt0, prompt1, prompt2 = prompt
        litellm.set_verbose=True

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "system", "content": prompt0},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": image_path,
                                                                                                    "detail": "high"},
                                                                }]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "system", "content": prompt0},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": image_path,
                                                                                                    "detail": "high"}, 
                                                                }]},
                {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt_input,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature if temperature else self.temperature,
            **kwargs,
        )
        return [choice["message"]["content"] for choice in response.choices][0]


class OpenAIEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init an OpenAI GPT/Codex engine
            To find your OpenAI API key, visit https://platform.openai.com/api-keys
        """
        super().__init__(**kwargs)

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        prompt0, prompt1, prompt2 = prompt
        # litellm.set_verbose=True

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url":
                                                                                                        f"data:image/jpeg;base64,{base64_image}",
                                                                                                    "detail": "high"},
                                                                 }]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url":
                                                                                                        f"data:image/jpeg;base64,{base64_image}",
                                                                                                    "detail": "high"}, }]},
                {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt_input,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature if temperature else self.temperature,
            **kwargs,
        )
        return [choice["message"]["content"] for choice in response.choices][0]


class OpenaiEngine_MindAct(Engine):
    def __init__(self, **kwargs) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        super().__init__(**kwargs)
    #
    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(self, prompt, max_new_tokens=50, temperature=0, model=None, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        if isinstance(prompt, str):
            # Assume one turn dialogue
            prompt = [
                {"role": "user", "content": prompt},
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )
        return [choice["message"]["content"] for choice in response["choices"]]
