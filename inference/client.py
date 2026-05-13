"""
LLM Client for batch inference with multiple provider support.
Copied from heterogeneous_knowledge_graph/llm_client.py
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM inference"""
    provider: str = "huggingface"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    batch_size: int = 10
    max_concurrent: int = 5
    timeout: int = 60
    max_requests_per_minute: int = 500
    max_tokens_per_minute: int = 90000


class InferenceClient:
    """Unified LLM client supporting batch inference with rate limiting"""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = self._load_config_from_env()

        self.config = config
        self.client = None
        
        # Rate limiters
        self._request_limiter = AsyncLimiter(
            max_rate=self.config.max_requests_per_minute,
            time_period=60
        )
        self._token_limiter = AsyncLimiter(
            max_rate=self.config.max_tokens_per_minute,
            time_period=60
        )
        
        self._setup_client()

    def _load_config_from_env(self) -> LLMConfig:
        """Load configuration from environment variables"""
        hf_model = os.getenv("HF_MODEL", os.getenv("HUGGINGFACE_MODEL",
                    os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")))
        hf_api_key = (os.getenv("HF_TOKEN")
                      or os.getenv("HUGGINGFACE_API_KEY")
                      or os.getenv("OPENAI_API_KEY"))
        hf_base_url = (os.getenv("HF_BASE_URL")
                       or os.getenv("HUGGINGFACE_BASE_URL")
                       or os.getenv("OPENAI_BASE_URL")
                       or "https://router.huggingface.co/v1")
        provider = os.getenv("LLM_PROVIDER",
                             os.getenv("USE_PROVIDER", "huggingface"))
        return LLMConfig(
            provider=provider,
            model=hf_model,
            api_key=hf_api_key,
            base_url=hf_base_url,
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_concurrent=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
            timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
            max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "500")),
            max_tokens_per_minute=int(os.getenv("MAX_TOKENS_PER_MINUTE", "90000"))
        )

    def _setup_client(self):
        """Initialize OpenAI-compatible client (HuggingFace, OpenAI, or any compatible endpoint)."""
        base_url = (self.config.base_url or "").strip() or "https://router.huggingface.co/v1"

        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=base_url,
            timeout=self.config.timeout,
            max_retries=3
        )
        logger.info(
            "Initialized OpenAI-compatible client with base_url: %s model: %s",
            base_url,
            self.config.model,
        )

    async def _call_openai(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024
    ) -> str:
        """Call OpenAI-compatible HuggingFace endpoint with rate limiting."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 3
        retry_count = 0
        
        async with self._request_limiter:
            while True:
                try:
                    request_params = {
                        "model": self.config.model,
                        "messages": messages,
                        "max_tokens": max_tokens
                    }
                    
                    response = await self.client.chat.completions.create(**request_params)
                    return response.choices[0].message.content
                except Exception as e:
                    error_str = str(e)
                    if ("429" in error_str or "rate" in error_str.lower() or 
                        "500" in error_str or "503" in error_str):
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Max retries exceeded: {e}")
                            raise
                        
                        delay = 2 ** retry_count
                        logger.warning(f"Rate limit/server error, retrying in {delay}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Error calling HuggingFace endpoint: {e}")
                        raise

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024
    ) -> str:
        """Generate a single response"""
        return await self._call_openai(prompt, system_prompt, max_tokens)

    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: str = "",
        system_prompts: Optional[List[str]] = None,
        max_tokens: int = 1024,
        show_progress: bool = True,
        desc: str = "Processing prompts"
    ) -> List[str]:
        """
        Generate responses for a batch of prompts with concurrency control.
        Supports either a single system prompt for all, or a list of system prompts.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def generate_with_semaphore(prompt: str, s_prompt: str, index: int) -> tuple[int, str]:
            async with semaphore:
                try:
                    result = await self.generate(prompt, s_prompt, max_tokens)
                    return index, result
                except Exception as e:
                    logger.error(f"Error processing prompt {index}: {str(e)}")
                    return index, ""

        # Determine system prompt for each item
        final_system_prompts = []
        if system_prompts:
            if len(system_prompts) != len(prompts):
                raise ValueError("system_prompts list must match prompts list length")
            final_system_prompts = system_prompts
        else:
            final_system_prompts = [system_prompt] * len(prompts)

        # Create tasks with their indices to maintain order
        tasks = [
            generate_with_semaphore(prompt, final_system_prompts[i], i)
            for i, prompt in enumerate(prompts)
        ]

        # Execute with progress tracking if requested
        if show_progress:
            try:
                from tqdm.asyncio import tqdm as async_tqdm
                results = await async_tqdm.gather(*tasks, desc=desc)
            except ImportError:
                logger.warning("tqdm not available, processing without progress bar")
                results = await asyncio.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks)

        # Sort by index and extract responses
        results.sort(key=lambda x: x[0])
        return [response for _, response in results]
