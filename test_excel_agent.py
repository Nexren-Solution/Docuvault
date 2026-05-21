"""
Standalone test for ExcelAgent.

Usage:
    python test_excel_agent.py path/to/downtime_log.xlsx

Runs five representative queries and prints aggregated markdown answers.
No Django setup required.
"""

import os
import sys

# Ensure Docuvault project root is importable (handles running from other dirs)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from documents.rag.config import RAGConfig
from documents.rag.excel_agent import ExcelAgent


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        print(
            text.encode(sys.stdout.encoding or 'utf-8', errors='replace')
                .decode(sys.stdout.encoding or 'utf-8', errors='replace'),
            **kwargs,
        )


TEST_QUERIES = [
    "what are the top 10 repeated issues?",
    "summarize sub-assembly wise",
    "list machines",
    "BR 1&3 APPLICATOR down",
    "what is the action plan for B&T SHUTTLE MOVEMENT AXIS FAULT",
]


def main() -> None:
    if len(sys.argv) < 2:
        _safe_print("Usage: python test_excel_agent.py <path_to_excel_or_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        _safe_print(f"File not found: {file_path}")
        sys.exit(1)

    config = RAGConfig()
    agent = ExcelAgent(config=config)

    _safe_print("\n" + "=" * 70)
    _safe_print(f"Excel Agent Test  —  {os.path.basename(file_path)}")
    _safe_print("=" * 70)

    try:
        summary = agent.get_data_summary(file_path)
        col_preview = ', '.join(summary['columns'][:10])
        if len(summary['columns']) > 10:
            col_preview += f" ... (+{len(summary['columns']) - 10} more)"
        _safe_print(f"\nRows : {summary['rows']}")
        _safe_print(f"Cols : {len(summary['columns'])}")
        _safe_print(f"       {col_preview}")
    except Exception as exc:
        _safe_print(f"[summary error] {exc}")

    for idx, question in enumerate(TEST_QUERIES, 1):
        _safe_print(f"\n{'─' * 70}")
        _safe_print(f"[{idx}/{len(TEST_QUERIES)}]  {question}")
        _safe_print("─" * 70)
        try:
            answer, meta = agent.query(file_path, question)
            _safe_print(answer)
            _safe_print(f"\n[engine: {meta.get('engine')}  |  rows analysed: {meta.get('rows')}]")
        except Exception as exc:
            _safe_print(f"ERROR: {exc}")

    _safe_print(f"\n{'=' * 70}")
    _safe_print("Test complete.")
    _safe_print("=" * 70)


if __name__ == "__main__":
    main()
