"""
Unified Chatbot Router
Routes queries to the right engine based on file type:
  - .xlsx/.csv → ExcelAgent (PandasAI)
  - .pdf/.docx → RAGChatbot (existing system)

Drop this into your existing Django view in place of direct RAGChatbot calls.
"""

import os
from typing import List, Tuple, Dict, Optional

from .config import RAGConfig
from .conversation import RAGChatbot
from .excel_agent import ExcelAgent


class UnifiedChatbot:
    """
    Single entry point that routes to the right engine.
    Your Django view just calls .query() and doesn't care about file types.
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.rag_chatbot = RAGChatbot(config=self.config)
        self.excel_agent = ExcelAgent(config=self.config)
        self._rag_initialized = False

    def initialize(self, db_path: str = None, reset: bool = False):
        """Initialize the RAG side. ExcelAgent needs no init."""
        if not self._rag_initialized:
            self.rag_chatbot.initialize(db_path=db_path, reset=reset)
            self._rag_initialized = True

    def query(
        self,
        question: str,
        file_path: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, List[Dict]]:
        """
        Main query entry point.

        Args:
            question: User's question
            file_path: Path to the file being queried (optional).
                      If provided and it's tabular (.xlsx/.csv), routes to ExcelAgent.
                      Otherwise uses RAG over the indexed corpus.
            thread_id: Conversation thread ID (for RAG memory)

        Returns:
            (answer, sources)
        """

        # Route 1: Tabular file → PandasAI
        if file_path and ExcelAgent.is_tabular_file(file_path):
            answer, meta = self.excel_agent.query(file_path, question)

            # Normalize the response shape to match what your RAG returns
            sources = [{
                'source': meta['file'],
                'page': 1,
                'similarity': 1.0,
                'content_type': 'spreadsheet',
                'text_preview': f"{meta['rows']} rows analyzed via PandasAI",
            }]
            return answer, sources

        # Route 2: Everything else → existing RAG
        self.initialize()
        return self.rag_chatbot.query(
            question=question,
            thread_id=thread_id,
            **kwargs
        )
