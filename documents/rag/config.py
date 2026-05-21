"""
Enhanced Configuration Settings for Multi-Modal RAG System
Optimized for LOCAL GPU (Qwen2.5 / gpt-oss:20b + RTX 5070) with Ollama support

CHANGES (v2 — anti-hallucination):
  - EXCEL_STRICT_SYSTEM_PROMPT added: forces LLM to only narrate/format
    precomputed pandas results, never compute or infer on its own.
  - _PRESENT_SYSTEM prompt in excel_tool.py also tightened (see excel_tool.py).
  - SYSTEM_PROMPT updated with stronger grounding rules.
  - Added EXCEL_ANALYTICS_ENABLED flag and KPI config section.
"""

import sys
import os


def _safe_print(*args, **kwargs):
    """Unicode-safe print for Windows consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(
            sys.stdout.encoding or 'utf-8', errors='replace'), **kwargs)


class RAGConfig:
    """Configuration for GPU-accelerated local RAG system."""

    # ==================== MODEL CONFIG ====================

    EMBEDDING_MODEL = "BAAI/bge-base-en"
    LLM_MODEL = "gpt-oss:20b"
    IMAGE_MODEL = "Salesforce/blip2-opt-2.7b"

    # ==================== TEXT PROCESSING ====================

    CHUNK_SIZE    = 512
    CHUNK_OVERLAP = 128
    TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    # ==================== RETRIEVAL ====================

    N_RESULTS = 15

    SIMILARITY_THRESHOLD      = -0.15
    STRONG_CONTEXT_THRESHOLD  = -0.07

    USE_HYBRID_SEARCH = True
    SEMANTIC_WEIGHT   = 0.85

    ENABLE_RERANKING  = True

    ALLOW_GENERAL_KNOWLEDGE  = True
    STRICT_DOCUMENT_MODE     = False
    INDICATE_KNOWLEDGE_SOURCE = False

    # ==================== LLM GENERATION ====================

    MAX_NEW_TOKENS     = 1024
    TEMPERATURE        = 0.2
    TOP_P              = 0.85
    REPETITION_PENALTY = 1.1

    REWRITE_MAX_TOKENS   = 50
    REWRITE_TEMPERATURE  = 0.1
    REWRITE_MAX_HISTORY  = 4

    # ==================== MEMORY & CONTEXT ====================

    MAX_MEMORY_TOKEN_LIMIT = 6000
    MAX_HISTORY_TURNS      = 8
    MAX_CONTEXT_LENGTH     = 6000

    # ==================== MULTIMODAL ====================

    ENABLE_TABLE_EXTRACTION    = True
    TABLE_EXTRACTION_METHOD    = "pdfplumber"

    ENABLE_OCR                 = False
    OCR_DPI                    = 300
    OCR_LANG                   = "eng"

    ENABLE_IMAGE_DESCRIPTION   = False
    IMAGE_DESCRIPTION_MAX_TOKENS = 100

    # ==================== VECTOR DB ====================

    CHROMA_DB_PATH  = None
    COLLECTION_NAME = "docuvault_documents_enhanced"

    # ==================== PERFORMANCE ====================

    DEVICE               = None
    EMBEDDING_BATCH_SIZE = 64

    USE_8BIT_QUANTIZATION = False
    LLM_INT8_THRESHOLD    = 6.0

    # ==================== EXCEL ANALYTICS (NEW) ====================

    # Master switch: when True, the KPI analytics engine in excel_tool.py
    # is active and handles MTBF/MTTR/OEE/recurrence queries deterministically.
    EXCEL_ANALYTICS_ENABLED = True

    # Column name hints — the KPI engine tries to auto-detect, but if your
    # Excel files use non-standard names, override here.
    EXCEL_DOWNTIME_COL_HINTS = [
        'downtime', 'down time', 'total down time', 'duration',
        'repair time', 'repair duration', 'minutes', 'hours',
    ]
    EXCEL_EQUIPMENT_COL_HINTS = [
        'equipment', 'machine', 'asset', 'equipment name', 'machine name',
    ]
    EXCEL_TIMESTAMP_COL_HINTS = [
        'date', 'timestamp', 'start time', 'end time', 'failure date',
        'breakdown date', 'occurrence date',
    ]
    EXCEL_ISSUE_COL_HINTS = [
        'issue', 'failure', 'fault', 'problem', 'description',
        'issue category', 'failure mode', 'root cause',
    ]
    EXCEL_SHIFT_COL_HINTS = [
        'shift', 'shift name', 'shift id',
    ]

    # ==================== SYSTEM PROMPTS ====================

    SYSTEM_PROMPT = """You are a helpful AI assistant for DocuVault, a document management system.
You answer questions helpfully using document context when available, and general knowledge otherwise.

DOCUMENT CONTEXT RULES:
- When the Context section has detailed, relevant information: use it as your main answer.
- Do NOT say "According to the document", "the PDF says", "based on the context", or reference page numbers.
- Just state the facts naturally and confidently.

GENERAL KNOWLEDGE RULES:
- If the Context section is empty or contains no answer: answer from general knowledge immediately.
- If the Context mentions a concept only briefly (e.g., as a feature name in a list): explain it using general knowledge. Do NOT say "not mentioned in documents."
- If the Context is off-topic: ignore it and answer from general knowledge.
- NEVER refuse to answer or say "not in documents" when you know the answer from general knowledge.

CRITICAL DATA-GROUNDING RULES:
- When presenting numerical facts, statistics, rankings, or KPI values: ONLY use numbers
  that are explicitly present in the data or context provided. Do NOT compute, estimate,
  or round numbers yourself.
- If the data does not contain enough information to answer a question reliably,
  say: "The available data does not contain sufficient information to determine this."
- Do NOT make assumptions about operator behavior, training needs, maintenance quality,
  or root causes unless explicit evidence exists in the data.
- Do NOT inject generic industrial/maintenance knowledge as if it came from the data.

STYLE:
- Be concise, clear, and conversational.
- Use bullet points when listing multiple items.

LANGUAGE:
- Always respond in the exact same language the user used — Hindi, Hinglish, English, etc.
- Never switch languages."""

    STRICT_SYSTEM_PROMPT = """You are a helpful AI assistant that provides information strictly based on the provided documents.

HOW TO RESPOND:
- ONLY answer based on information in the Context section below
- Talk naturally and conversationally
- Present information directly without mentioning sources or documents
- Do NOT use phrases like "According to...", "The document shows...", etc.

STRICT RULES:
- If the Context section contains relevant information, use it to answer
- If the Context does NOT contain information to answer the question, clearly state:
  "I cannot find that information in the available documents"
- Do NOT use general knowledge or information outside the provided context
- Do NOT make assumptions or inferences beyond what's explicitly stated
- Do NOT compute metrics (MTBF, MTTR, OEE, etc.) yourself — only report precomputed values

PRESENTATION:
- Be clear and direct in your answers
- Use simple, natural language
- Never mention where the information comes from

Remember: You can ONLY use information from the Context section. If it's not there, say so."""

    INDICATED_SYSTEM_PROMPT = """You are a helpful AI assistant with access to specific documents and general knowledge.

HOW TO RESPOND:
- Answer questions naturally and conversationally
- Be clear, direct, and helpful
- Use simple language

USING DOCUMENT INFORMATION:
- When using information from the provided Context section, present it directly
- Do NOT mention documents, PDFs, or page numbers
- Simply state the information naturally

USING GENERAL KNOWLEDGE:
- When answering with general knowledge (not from documents), briefly indicate this
- Use phrases like:
  * "While this isn't covered in the specific documents, I can explain that..."
  * "Based on general knowledge, ..."
  * "The documents don't specifically address this, but..."
- Keep these indicators brief and natural

COMBINED ANSWERS:
- You can combine document information with general knowledge
- Make it clear which parts come from documents vs. general knowledge
- Keep the distinction subtle and conversational

IF YOU DON'T KNOW:
- Be honest about uncertainty
- Don't make up information

Remember: Help users understand when you're using specific document information versus general knowledge."""

    # ── NEW: Excel/spreadsheet-specific prompt for narrating precomputed results ──
    EXCEL_STRICT_SYSTEM_PROMPT = """You present precomputed data analysis results to the user.

ABSOLUTE RULES:
1. Use ONLY the numbers, labels, and values from the 'Computed result' section.
2. NEVER compute, derive, estimate, or infer any number yourself.
3. NEVER add industrial knowledge, maintenance theory, or domain expertise.
4. NEVER recommend actions (training, maintenance schedules, etc.) unless
   the computed result explicitly contains such recommendations.
5. NEVER say things like "this suggests..." or "this could indicate..." —
   only state what the data shows.
6. If the result says "Insufficient data", relay that honestly.
7. Be direct and concise — 1-3 sentences maximum.
8. Present the result in the same language the user asked in.

FORBIDDEN PHRASES (never use these):
- "This suggests that..."
- "This could indicate..."
- "It is likely that..."
- "Based on industry standards..."
- "Best practice would be..."
- "Operators should..."
- "Training is needed..."

You are a DATA NARRATOR, not an analyst. Narrate what the numbers say, nothing more."""

    REWRITE_SYSTEM_PROMPT = """Rewrite the follow-up question as a standalone question that includes necessary context from the conversation history.

Rules:
- Keep the rewritten question concise (1-2 sentences max)
- Include only essential context
- Maintain the original intent
- Output ONLY the rewritten question - no explanations
- If the question is already standalone, output it unchanged"""

    STOP_TOKEN_IDS = [151645]

    # ==================== TEMPLATES ====================

    NO_CONTEXT_TEMPLATE = """The user's uploaded documents do not contain information about this topic.

Answer the following question directly from your general knowledge. Give a clear, helpful, conversational answer. Do NOT say "I cannot find this in documents" or ask the user to upload anything — just answer the question.

Question: {question}"""

    WITH_CONTEXT_TEMPLATE = """Context from documents:
{context}

Question: {question}

Instructions:
- The context above is relevant — use it as your main answer source.
- Do not reference "the document" or "the context" in your answer.
- Do NOT compute metrics yourself — only report values present in the context.
- Respond in the same language the user used."""

    WEAK_CONTEXT_TEMPLATE = """The user's documents mention this topic only briefly or tangentially.

Context from documents (for reference only):
{context}

Question: {question}

Instructions:
- Answer the question using your general knowledge — give a clear, complete answer.
- If the document context contains something specifically relevant to the question, mention it naturally.
- Do NOT say "not mentioned in documents" or ask the user to upload anything.
- Respond in the same language the user used."""

    STRICT_NO_CONTEXT_RESPONSE = "I cannot find relevant information in the available documents to answer this question."

    # ==================== METHODS ====================

    @classmethod
    def set_chroma_path(cls, base_path: str):
        cls.CHROMA_DB_PATH = os.path.join(base_path, 'chroma_db_enhanced')
        os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)

    @classmethod
    def set_device(cls, device: str):
        cls.DEVICE = device

    @classmethod
    def get_active_system_prompt(cls) -> str:
        if cls.STRICT_DOCUMENT_MODE:
            return cls.STRICT_SYSTEM_PROMPT
        elif cls.INDICATE_KNOWLEDGE_SOURCE:
            return cls.INDICATED_SYSTEM_PROMPT
        else:
            return cls.SYSTEM_PROMPT

    @classmethod
    def get_config_summary(cls) -> dict:
        return {
            'embedding_model':        cls.EMBEDDING_MODEL,
            'llm_model':              cls.LLM_MODEL,
            'chunk_size':             cls.CHUNK_SIZE,
            'n_results':              cls.N_RESULTS,
            'hybrid_search':          cls.USE_HYBRID_SEARCH,
            'table_extraction':       cls.ENABLE_TABLE_EXTRACTION,
            'ocr_enabled':            cls.ENABLE_OCR,
            'image_description':      cls.ENABLE_IMAGE_DESCRIPTION,
            'device':                 cls.DEVICE or 'auto',
            'embedding_batch_size':   cls.EMBEDDING_BATCH_SIZE,
            'max_context_length':     cls.MAX_CONTEXT_LENGTH,
            'allow_general_knowledge': cls.ALLOW_GENERAL_KNOWLEDGE,
            'strict_document_mode':   cls.STRICT_DOCUMENT_MODE,
            'indicate_knowledge_source': cls.INDICATE_KNOWLEDGE_SOURCE,
            'excel_analytics_enabled': cls.EXCEL_ANALYTICS_ENABLED,
        }

    # ==================== MODES ====================

    @classmethod
    def enable_all_features(cls):
        cls.ENABLE_TABLE_EXTRACTION  = True
        cls.ENABLE_OCR               = True
        cls.ENABLE_IMAGE_DESCRIPTION = True
        cls.N_RESULTS                = 15
        cls.CHUNK_SIZE               = 512
        cls.ALLOW_GENERAL_KNOWLEDGE  = True
        cls.STRICT_DOCUMENT_MODE     = False
        cls.EXCEL_ANALYTICS_ENABLED  = True
        _safe_print("✅ All multi-modal features enabled")

    @classmethod
    def enable_lightweight(cls):
        cls.ENABLE_OCR               = False
        cls.ENABLE_IMAGE_DESCRIPTION = False
        cls.N_RESULTS                = 15
        cls.CHUNK_SIZE               = 512
        cls.EMBEDDING_BATCH_SIZE     = 64
        cls.ALLOW_GENERAL_KNOWLEDGE  = True
        _safe_print("✅ Lightweight mode enabled")

    @classmethod
    def enable_strict_mode(cls):
        cls.STRICT_DOCUMENT_MODE      = True
        cls.ALLOW_GENERAL_KNOWLEDGE   = False
        cls.INDICATE_KNOWLEDGE_SOURCE = False
        _safe_print("✅ Strict document-only mode enabled")

    @classmethod
    def enable_hybrid_mode(cls):
        cls.STRICT_DOCUMENT_MODE      = False
        cls.ALLOW_GENERAL_KNOWLEDGE   = True
        cls.INDICATE_KNOWLEDGE_SOURCE = False
        _safe_print("✅ Hybrid mode enabled")

    @classmethod
    def enable_indicated_mode(cls):
        cls.STRICT_DOCUMENT_MODE      = False
        cls.ALLOW_GENERAL_KNOWLEDGE   = True
        cls.INDICATE_KNOWLEDGE_SOURCE = True
        _safe_print("✅ Indicated mode enabled")

    @classmethod
    def set_similarity_threshold(cls, threshold: float):
        cls.SIMILARITY_THRESHOLD = threshold
        _safe_print(f"✅ Similarity threshold set to {threshold}")

    @classmethod
    def configure_for_use_case(cls, use_case: str):
        use_case = use_case.lower()

        if use_case == 'general_qa':
            cls.enable_hybrid_mode()
            cls.SIMILARITY_THRESHOLD = -0.15
            cls.N_RESULTS            = 15
            cls.TEMPERATURE          = 0.2
            _safe_print("📋 Configured for: General Q&A")

        elif use_case == 'strict_compliance':
            cls.enable_strict_mode()
            cls.SIMILARITY_THRESHOLD = -0.07
            cls.N_RESULTS            = 15
            cls.TEMPERATURE          = 0.1
            _safe_print("📋 Configured for: Strict Compliance")

        elif use_case == 'research':
            cls.enable_indicated_mode()
            cls.SIMILARITY_THRESHOLD = -0.20
            cls.N_RESULTS            = 15
            cls.TEMPERATURE          = 0.2
            cls.ENABLE_RERANKING     = True
            _safe_print("📋 Configured for: Research")

        elif use_case == 'customer_support':
            cls.enable_hybrid_mode()
            cls.SIMILARITY_THRESHOLD = -0.15
            cls.N_RESULTS            = 15
            cls.TEMPERATURE          = 0.3
            _safe_print("📋 Configured for: Customer Support")

        elif use_case == 'industrial_analytics':
            cls.enable_strict_mode()
            cls.SIMILARITY_THRESHOLD = -0.07
            cls.N_RESULTS            = 15
            cls.TEMPERATURE          = 0.1
            cls.EXCEL_ANALYTICS_ENABLED = True
            _safe_print("📋 Configured for: Industrial Analytics (strict, no hallucination)")

        else:
            _safe_print(f"❌ Unknown use case: {use_case}")
            _safe_print("   Available: 'general_qa', 'strict_compliance', 'research', "
                        "'customer_support', 'industrial_analytics'")
