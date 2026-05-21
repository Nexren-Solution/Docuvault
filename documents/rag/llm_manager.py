"""
LLM management module
Handles loading and inference with Qwen locally using LangChain + Ollama.
"""

import sys
from typing import List, Dict, Optional

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
    """Manages local Qwen model via Ollama using LangChain"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.llm: Optional[ChatOllama] = None
        self.device = "Ollama (Auto-CUDA)"

    # ────────────────────────────────────────────────────────────────────────
    # Loading
    # ────────────────────────────────────────────────────────────────────────
    def load_model(self):
        """Connect to Ollama (no actual model load — Ollama handles that server-side)."""
        if self.llm is not None:
            _safe_print("LLM already loaded")
            return

        model_name = self.config.LLM_MODEL
        _safe_print(f"🚀 Connecting to Ollama model: {model_name}")

        self.llm = ChatOllama(
            model=model_name,
            temperature=self.config.TEMPERATURE,
            top_p=self.config.TOP_P,
            repeat_penalty=self.config.REPETITION_PENALTY,
        )

        _safe_print(f"✅ Successfully connected to Ollama ({model_name})")

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────
    def _to_langchain_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convert dict messages to LangChain messages."""
        lc_messages: List[BaseMessage] = []

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

    def _build_ollama_params(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        Build the kwargs dict for ChatOllama.invoke / stream.
        Note: max_new_tokens is intentionally ignored so the LLM finishes naturally.
        Re-enable by uncommenting the num_predict block below.
        """
        options: Dict = {}

        # if max_new_tokens:
        #     options["num_predict"] = max_new_tokens
        if temperature is not None:
            options["temperature"] = temperature

        return {"options": options} if options else {}

    def _prepend_system_prompt(self, lc_messages: List[BaseMessage]) -> List[BaseMessage]:
        """Prepend the active system prompt if one is configured."""
        system_prompt = self.config.get_active_system_prompt()
        if not system_prompt:
            return lc_messages
        return [SystemMessage(content=system_prompt)] + lc_messages

    # ────────────────────────────────────────────────────────────────────────
    # Public generation API
    # ────────────────────────────────────────────────────────────────────────
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""

        if self.llm is None:
            self.load_model()

        lc_messages = self._to_langchain_messages(messages)
        lc_messages = self._prepend_system_prompt(lc_messages)

        params = self._build_ollama_params(max_new_tokens, temperature)

        response = self.llm.invoke(lc_messages, **params)

        return response.content.encode("utf-8", errors="replace").decode("utf-8").strip()

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = None,
        temperature: float = None,
    ):
        """Stream a response token-by-token."""

        if self.llm is None:
            self.load_model()

        lc_messages = self._to_langchain_messages(messages)
        lc_messages = self._prepend_system_prompt(lc_messages)

        params = self._build_ollama_params(max_new_tokens, temperature)

        for chunk in self.llm.stream(lc_messages, **params):
            if chunk.content:
                yield chunk.content.encode("utf-8", errors="replace").decode("utf-8")

    # ────────────────────────────────────────────────────────────────────────
    # Question rewriting
    # ────────────────────────────────────────────────────────────────────────
    def rewrite_question(self, question: str, chat_history: List) -> str:
        """Rewrite follow-up questions to be self-contained, using chat history."""

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
            ),
        ]

        params = {
            "options": {
                "num_predict": self.config.REWRITE_MAX_TOKENS,
                "temperature": self.config.REWRITE_TEMPERATURE,
            }
        }

        try:
            rewritten = self.llm.invoke(messages, **params).content.strip()
        except Exception as exc:
            _safe_print(f"⚠️ Rewrite invoke failed ({exc}), using original")
            return question

        rewritten = rewritten.split("\n")[0].strip()

        if len(rewritten) < 5 or len(rewritten) > 250:
            _safe_print("⚠️ Rewrite produced invalid output, using original")
            return question

        _safe_print(f"🔄 Rewritten: {rewritten}")
        return rewritten

    # ────────────────────────────────────────────────────────────────────────
    # Introspection
    # ────────────────────────────────────────────────────────────────────────
    def get_model_info(self) -> Dict:
        """Return model info."""
        if self.llm is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.llm.model,
            "device": self.device,
            "backend": "Local Ollama API",
        }