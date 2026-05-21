"""
Enhanced Configuration Settings for Multi-Modal RAG System
Optimized for LOCAL GPU (Qwen2.5 + RTX 5070)
"""

import sys
import os
from pathlib import Path


def _safe_print(*args, **kwargs):
    """Unicode-safe print for Windows"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        print(text.encode('utf-8', errors='replace').decode('utf-8'), **kwargs)


class RAGConfig:
    """Configuration for local RAG system"""

    # ==================== MODEL CONFIG ====================

    # 🔥 BEST embedding for RAG
    EMBEDDING_MODEL = "BAAI/bge-base-en"

    # 🔥 LOCAL LLM (NO API)
    # LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MODEL = "gpt-oss:20b" # switching to ollama support 

    # Optional multimodal
    IMAGE_MODEL = "Salesforce/blip2-opt-2.7b"

    # ==================== TEXT PROCESSING ====================

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 100
    TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    # ==================== RETRIEVAL ====================

    N_RESULTS = 6
    SIMILARITY_THRESHOLD = 0.50

    USE_HYBRID_SEARCH = True
    SEMANTIC_WEIGHT = 0.75

    ENABLE_RERANKING = True

    # Knowledge mode
    ALLOW_GENERAL_KNOWLEDGE = True
    STRICT_DOCUMENT_MODE = False
    INDICATE_KNOWLEDGE_SOURCE = False

    # ==================== LLM GENERATION ====================

    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3
    TOP_P = 0.9
    REPETITION_PENALTY = 1.1

    # Rewrite
    REWRITE_MAX_TOKENS = 50
    REWRITE_TEMPERATURE = 0.1
    REWRITE_MAX_HISTORY = 4

    # ==================== MEMORY ====================

    MAX_MEMORY_TOKEN_LIMIT = 3000
    MAX_HISTORY_TURNS = 8
    MAX_CONTEXT_LENGTH = 4000

    # ==================== MULTIMODAL ====================

    ENABLE_TABLE_EXTRACTION = True
    TABLE_EXTRACTION_METHOD = "pdfplumber"

    ENABLE_OCR = False
    OCR_DPI = 300
    OCR_LANG = "eng"

    ENABLE_IMAGE_DESCRIPTION = False
    IMAGE_DESCRIPTION_MAX_TOKENS = 100

    # ==================== VECTOR DB ====================

    CHROMA_DB_PATH = None
    COLLECTION_NAME = "docuvault_documents"

    # ==================== PERFORMANCE ====================

    DEVICE = None
    EMBEDDING_BATCH_SIZE = 32

    # ❌ DISABLED (Windows safe)
    USE_8BIT_QUANTIZATION = False
    LLM_INT8_THRESHOLD = 6.0

    # ==================== SYSTEM PROMPTS ====================

    SYSTEM_PROMPT = """You are a helpful AI assistant for DocuVault.

PRIORITY:
1. Use document context first
2. Use general knowledge only if needed

RULES:
- Do NOT mention documents explicitly (unless handling an ambiguous query as described below)
- Answer clearly and concisely
- Use bullet points if needed
- Respond in same language as user
- Use the HANDLING AMBIGUOUS QUERIES section below to handle ambiguous queries

HANDLING AMBIGUOUS QUERIES:
If the user asks an ambiguous general question (especially at the beginning of the chat) such as:
1. "give me a summary"
2. "what are the key points"
3. "provide the recommendations"
4. "summarize this"
5. "give me the highlights"
6. "what is this about"
7. "explain this document"
8. "provide a brief overview"
9. "what are the main takeaways"
10. "give me the conclusion"
11. "what does it say"
12. "summarize the key findings"
13. "outline the main points"
14. "can you simplify this"
15. "give me the tl;dr"

You must provide the response using the available context, but you MUST conclude your response with this exact note:
"I have responded using the available documents in the system. If you'd like to get results for some specific document or concept, you may ask about those as well."
"""

    STRICT_SYSTEM_PROMPT = """Answer ONLY from provided context.

If answer not found:
"I cannot find that information in the available documents"
"""

    REWRITE_SYSTEM_PROMPT = """Rewrite the question as standalone.

Rules:
- Keep short
- Preserve intent
- Output ONLY rewritten question
"""

    STOP_TOKEN_IDS = [151645]

    # ==================== TEMPLATES ====================

    NO_CONTEXT_TEMPLATE = """Question: {question}

Answer using general knowledge.
"""

    WITH_CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}

Answer using the provided context first. If the provided context does not contain the answer, use your general knowledge to answer the question. Do NOT say "the context does not contain the information", just answer the question directly.

CRITICAL INSTRUCTION: If the Question above is a generic or ambiguous request (like "give me a summary", "what are the key points", "explain this document", etc.), you MUST conclude your entire response with exactly this note:
"I have responded using the available documents in the system. If you'd like to get results for some specific document or concept, you may ask about those as well."
"""

    STRICT_NO_CONTEXT_RESPONSE = "I cannot find relevant information in documents."

    # ==================== METHODS ====================

    @classmethod
    def set_chroma_path(cls, base_path: str):
        cls.CHROMA_DB_PATH = os.path.join(base_path, 'chroma_db')
        os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)

    @classmethod
    def set_device(cls, device: str):
        cls.DEVICE = device

    @classmethod
    def get_active_system_prompt(cls):
        if cls.STRICT_DOCUMENT_MODE:
            return cls.STRICT_SYSTEM_PROMPT
        return cls.SYSTEM_PROMPT

    @classmethod
    def get_config_summary(cls):
        return {
            "embedding": cls.EMBEDDING_MODEL,
            "llm": cls.LLM_MODEL,
            "device": cls.DEVICE or "auto",
            "hybrid": cls.USE_HYBRID_SEARCH
        }

    # ==================== MODES ====================

    @classmethod
    def enable_lightweight(cls):
        cls.ENABLE_OCR = False
        cls.N_RESULTS = 4
        cls.CHUNK_SIZE = 256
        cls.EMBEDDING_BATCH_SIZE = 16
        _safe_print("✅ Lightweight mode")

    @classmethod
    def enable_strict_mode(cls):
        cls.STRICT_DOCUMENT_MODE = True
        cls.ALLOW_GENERAL_KNOWLEDGE = False
        _safe_print("✅ Strict document mode")

    @classmethod
    def enable_hybrid_mode(cls):
        cls.STRICT_DOCUMENT_MODE = False
        cls.ALLOW_GENERAL_KNOWLEDGE = True
        _safe_print("✅ Hybrid mode")