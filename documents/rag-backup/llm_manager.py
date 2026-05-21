""""
LLM management module
Handles loading and inference with Qwen locally using LangChain.
"""

import sys
# OLD HUGGINGFACE IMPORTS:
# import torch
from typing import List, Dict, Optional

# OLD HUGGINGFACE IMPORTS:
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# LangChain
# OLD HUGGINGFACE IMPORTS:
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# NEW OLLAMA IMPORT:
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from .config import RAGConfig


def _safe_print(*args, **kwargs):
    """Unicode-safe print for Windows"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        print(text.encode('utf-8', errors='replace').decode('utf-8'), **kwargs)


class LLMManager:
    """Manages local Qwen model using LangChain"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # OLD HUGGINGFACE CODE:
        # self.model = None
        # self.tokenizer = None
        # self.llm: Optional[ChatHuggingFace] = None
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # NEW OLLAMA CODE:
        self.llm: Optional[ChatOllama] = None
        self.device = "Ollama (Auto-CUDA)"

    def load_model(self):
        """Load Qwen model locally"""
        if self.llm is not None:
            _safe_print("LLM already loaded")
            return

        # OLD HUGGINGFACE CODE:
        # _safe_print(f"🚀 Loading model: {self.config.LLM_MODEL}")
        # _safe_print("⏳ First run will download model (~6–8GB)...")
        #
        # model_name = self.config.LLM_MODEL
        #
        # # Load tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     model_name,
        #     trust_remote_code=True
        # )
        #
        # # Load model (GPU optimized)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #     torch_dtype=torch.float16,
        #     trust_remote_code=True
        # )
        #
        # # Fix padding token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #
        # # Create pipeline
        # text_pipeline = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     max_new_tokens=self.config.MAX_NEW_TOKENS,
        #     temperature=self.config.TEMPERATURE,
        #     top_p=self.config.TOP_P,
        #     repetition_penalty=self.config.REPETITION_PENALTY,
        #     do_sample=True,
        #     return_full_text=False
        # )
        #
        # # Wrap into LangChain
        # hf_llm = HuggingFacePipeline(pipeline=text_pipeline)
        # self.llm = ChatHuggingFace(llm=hf_llm)
        #
        # _safe_print("✅ Qwen2.5 loaded successfully on GPU")

        # NEW OLLAMA CODE:
        model_name = self.config.LLM_MODEL
        _safe_print(f"🚀 Connecting to Ollama model: {model_name}")

        self.llm = ChatOllama(
            model=model_name,
            temperature=self.config.TEMPERATURE,
            top_p=self.config.TOP_P,
            repeat_penalty=self.config.REPETITION_PENALTY
        )

        _safe_print(f"✅ Successfully connected to Ollama ({model_name})")

    def _to_langchain_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convert dict messages to LangChain messages"""
        lc_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        return lc_messages

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = None,
        temperature: float = None
    ) -> str:
        """Generate response"""

        if self.llm is None:
            self.load_model()

        # lc_messages = self._to_langchain_messages(messages)

        lc_messages = self._to_langchain_messages(messages)

        system_prompt = self.config.get_active_system_prompt()
        lc_messages = [SystemMessage(content=system_prompt)] + lc_messages

        response = self.llm.invoke(lc_messages, **params)


        # OLD HUGGINGFACE CODE:
        # params = {}
        # if max_new_tokens:
        #     params["max_tokens"] = max_new_tokens
        # if temperature:
        #     params["temperature"] = temperature
        
        # NEW OLLAMA CODE:
        options = {}
        # Comment out or remove these lines so num_predict is not sent to Ollama
        # Commenting out because I do not want the responses to be cut short in between, I want the LLM to finish its response naturally.
        # if max_new_tokens:
        #     options["num_predict"] = max_new_tokens
        if temperature is not None:
            options["temperature"] = temperature

        # Bundle them inside the 'options' kwarg that Ollama expects
        params = {"options": options} if options else {}

        response = self.llm.invoke(lc_messages, **params)

        # return response.content.strip()
        return response.content.encode("utf-8", errors="replace").decode("utf-8").strip()

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = None,
        temperature: float = None
    ):
        """Stream response"""

        if self.llm is None:
            self.load_model()

        lc_messages = self._to_langchain_messages(messages)

        # OLD HUGGINGFACE CODE:
        # params = {}
        # if max_new_tokens:
        #     params["max_tokens"] = max_new_tokens
        # if temperature is not None:
        #     params["temperature"] = temperature
        
        # NEW OLLAMA CODE:
        system_prompt = self.config.get_active_system_prompt()
        lc_messages = [SystemMessage(content=system_prompt)] + lc_messages
        options = {}
        # Comment out or remove these lines so num_predict is not sent to Ollama
        # Commenting out because I do not want the responses to be cut short in between, I want the LLM to finish its response naturally.
        # if max_new_tokens:
        #     options["num_predict"] = max_new_tokens
        if temperature is not None:
            options["temperature"] = temperature

        params = {"options": options} if options else {}

        for chunk in self.llm.stream(lc_messages, **params):
            if chunk.content:
                yield chunk.content.encode("utf-8", errors="replace").decode("utf-8")


    def rewrite_question(self, question: str, chat_history: List) -> str:
        """Rewrite follow-up question"""

        if self.llm is None:
            self.load_model()

        if not chat_history:
            return question

        history_text = ""

        for msg in chat_history[-self.config.REWRITE_MAX_HISTORY:]:

            if isinstance(msg, BaseMessage):
                content = msg.content
                role = "Q" if isinstance(msg, HumanMessage) else "A"

            elif isinstance(msg, dict):
                content = msg.get("content", "")
                role = "Q" if msg.get("role") == "user" else "A"

            else:
                continue

            history_text += f"{role}: {content[:100]}\n"

        messages = [
            SystemMessage(content=self.config.REWRITE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{history_text}\n\nCurrent question: {question}\n\nRewrite:"
            )
        ]

        # OLD HUGGINGFACE CODE:
        # params = {
        #     "max_tokens": self.config.REWRITE_MAX_TOKENS,
        #     "temperature": self.config.REWRITE_TEMPERATURE,
        # }
        
        # NEW OLLAMA CODE:
        params = {
            "options": {
                "num_predict": self.config.REWRITE_MAX_TOKENS,
                "temperature": self.config.REWRITE_TEMPERATURE,
            }
        }

        rewritten = self.llm.invoke(messages, **params).content.strip()

        rewritten = rewritten.split("\n")[0].strip()

        if len(rewritten) < 5 or len(rewritten) > 250:
            _safe_print("⚠️ Rewrite failed, using original")
            return question

        _safe_print(f"🔄 Rewritten: {rewritten}")
        return rewritten

    def get_model_info(self) -> Dict:
        """Return model info"""

        # OLD HUGGINGFACE CODE:
        # if self.model is None:
        #     return {"loaded": False}
        #
        # return {
        #     "loaded": True,
        #     "model_name": self.config.LLM_MODEL,
        #     "device": self.device,
        #     "memory_footprint_gb": self.model.get_memory_footprint() / 1e9,
        #     "backend": "Local Qwen2.5 (Transformers + LangChain)"
        # }

        # NEW OLLAMA CODE:
        if self.llm is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.llm.model,
            "device": self.device,
            "backend": "Local Ollama API"
        }