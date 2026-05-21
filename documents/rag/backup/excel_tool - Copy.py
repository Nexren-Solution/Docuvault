"""
Excel/spreadsheet + PDF-table query tool for the RAG chatbot.

Uses a rule-based query engine (no LLM code generation) to answer aggregation,
filter, and group-by questions against uploaded Excel or PDF-table files.

Key design points
-----------------
- DataFrame cache: each file is loaded once per modification-time fingerprint.
  On the remote server with 40 PDFs this avoids re-extracting tables on every
  query (was taking ~60 s).
- Safe type coercion: uses errors='coerce' everywhere so single bad cells
  don't crash the whole column conversion.
- Reversed date ranges: always normalises start ≤ end, so "31-Mar to 1-Mar"
  returns the same result as "1-Mar to 31-Mar".

Public API
----------
query_excel(question, org_id, llm_manager) -> str | None
"""

import os
import re
import sys
import time
import logging
import threading
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(
            sys.stdout.encoding or 'utf-8', errors='replace'), **kwargs)


# ── DataFrame cache ────────────────────────────────────────────────────────────
# Keyed by (path, mtime).  Thread-safe via a simple lock.
# Holds: Excel DFs and lists-of-DFs for PDF tables.

_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_TTL  = 600  # seconds — re-read if file is older than 10 min in cache


def _cache_get(path: str):
    with _cache_lock:
        entry = _cache.get(path)
        if entry is None:
            return None
        cached_mtime, cached_ts, value = entry
        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            return None
        if current_mtime != cached_mtime or (time.time() - cached_ts) > _CACHE_TTL:
            del _cache[path]
            return None
        return value


def _cache_set(path: str, value):
    with _cache_lock:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        _cache[path] = (mtime, time.time(), value)


# ── Detection ──────────────────────────────────────────────────────────────────

_AGG_KEYWORDS = re.compile(
    r'\b(total|sum|count|average|avg|max|min|minimum|maximum|highest|lowest|'
    r'how many|list|breakdown|group\s*by|per facility|per shift|'
    r'downtime|down\s*time|by facility|by shift|by date|against facility|'
    r'from \d|between \d|march|january|february|april|may|june|'
    r'july|august|september|october|november|december)\b',
    re.IGNORECASE,
)


def is_spreadsheet_query(question: str) -> bool:
    return bool(_AGG_KEYWORDS.search(question))


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _org_filter(org_id: str):
    from django.db.models import Q
    if org_id and str(org_id).startswith('user_'):
        return Q(owner_id=int(org_id.split('_')[1]))
    try:
        return Q(owner__organization_id=int(org_id))
    except (TypeError, ValueError):
        return Q()


def _find_excel_files(org_id: str) -> list:
    try:
        from documents.models import Document
        flt = _org_filter(org_id)
        xls  = Document.objects.filter(flt, is_deleted=False, file__endswith='.xls')
        xlsx = Document.objects.filter(flt, is_deleted=False, file__endswith='.xlsx')
        result = []
        for doc in (xls | xlsx).distinct():
            try:
                result.append({'title': doc.title, 'path': doc.file.path})
            except Exception:
                pass
        return result
    except Exception as exc:
        logger.warning(f"[ExcelTool] Excel DB lookup failed: {exc}")
        return []


def _find_pdf_files(org_id: str) -> list:
    try:
        from documents.models import Document
        flt = _org_filter(org_id)
        result = []
        for doc in Document.objects.filter(flt, is_deleted=False, file__endswith='.pdf'):
            try:
                result.append({'title': doc.title, 'path': doc.file.path})
            except Exception:
                pass
        return result
    except Exception as exc:
        logger.warning(f"[ExcelTool] PDF DB lookup failed: {exc}")
        return []


# ── DataFrame loading (Excel) ──────────────────────────────────────────────────

def _coerce_df_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns and numeric columns safely."""
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce',
                                         format='mixed')
            except TypeError:
                try:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                except Exception:
                    pass
        else:
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # Only replace if at least half the non-null values converted
                if converted.notna().sum() > df[col].notna().sum() * 0.5:
                    df[col] = converted
            except Exception:
                pass
    return df


def _load_df(path: str) -> Optional[pd.DataFrame]:
    cached = _cache_get(path)
    if cached is not None:
        return cached
    try:
        engine = 'openpyxl' if path.lower().endswith('.xlsx') else 'xlrd'
        df = pd.read_excel(path, engine=engine)
        df.columns = [str(c).strip() for c in df.columns]
        df = _coerce_df_types(df)
        _cache_set(path, df)
        return df
    except Exception as exc:
        logger.warning(f"[ExcelTool] Failed to load Excel {path}: {exc}")
        return None


# ── PDF table extraction (cached) ──────────────────────────────────────────────

def _build_df_from_rows(headers: list, rows: list) -> Optional[pd.DataFrame]:
    """Build a typed DataFrame from a list-of-string rows."""
    if len(rows) < 2 or len(headers) < 2:
        return None
    try:
        n = len(headers)
        padded = [r[:n] + [''] * max(0, n - len(r)) for r in rows]
        df = pd.DataFrame(padded, columns=headers)
        df.replace('', pd.NA, inplace=True)
        df = _coerce_df_types(df)
        return df
    except Exception as exc:
        logger.warning(f"[ExcelTool] _build_df_from_rows failed: {exc}")
        return None


def _looks_like_data_row(headers: list) -> bool:
    """True when what pdfplumber thinks is a header is actually a data row."""
    if not headers:
        return False
    try:
        float(headers[0].strip())
        return True
    except (ValueError, AttributeError):
        return False


def _extract_pdf_tables(path: str) -> list:
    """
    Extract and merge multi-page tables from a PDF using pdfplumber.
    Results are cached by file mtime so 40-PDF scans only happen once.
    """
    cached = _cache_get(path)
    if cached is not None:
        return cached

    try:
        import pdfplumber
    except ImportError:
        logger.warning("[ExcelTool] pdfplumber not installed")
        return []

    raw_tables = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    page_tables = page.extract_tables() or []
                except Exception:
                    continue
                for raw in page_tables:
                    if not raw or len(raw) < 2:
                        continue
                    try:
                        headers = [str(c).strip() if c else f'Col{i}'
                                   for i, c in enumerate(raw[0])]
                        rows = []
                        for row in raw[1:]:
                            cells = [str(c).strip() if c is not None else ''
                                     for c in row]
                            if any(c for c in cells):
                                rows.append(cells)
                        if rows:
                            raw_tables.append((headers, rows))
                    except Exception:
                        continue
    except Exception as exc:
        logger.warning(f"[ExcelTool] pdfplumber failed on {path}: {exc}")
        _cache_set(path, [])
        return []

    # Merge continuation pages (page 2+ have data row as "header")
    groups: dict = {}      # tuple(headers) → {'headers', 'rows'}
    ungrouped = []

    for headers, rows in raw_tables:
        n = len(headers)
        if not _looks_like_data_row(headers):
            key = tuple(headers)
            if key not in groups:
                groups[key] = {'headers': headers, 'rows': []}
            groups[key]['rows'].extend(rows)
        else:
            matched = False
            for key, grp in groups.items():
                if len(key) == n:
                    grp['rows'].append(headers)   # reclaim the mis-labelled row
                    grp['rows'].extend(rows)
                    matched = True
                    break
            if not matched:
                ungrouped.append((headers, rows))

    result = []
    for grp in groups.values():
        df = _build_df_from_rows(grp['headers'], grp['rows'])
        if df is not None:
            result.append(df)
    for headers, rows in ungrouped:
        df = _build_df_from_rows(headers, rows)
        if df is not None:
            result.append(df)

    _cache_set(path, result)
    return result


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_df(df: pd.DataFrame, title: str, question: str) -> int:
    q = question.lower()
    score = sum(2 for col in df.columns if col.lower() in q)
    for col in df.columns:
        for kw in ['date', 'time', 'facility', 'shift', 'equipment', 'total', 'down']:
            if kw in col.lower():
                score += 1
    score += min(len(df) // 100, 5)
    if title.lower() in q:
        score += 3
    return score


# ── Rule-based query engine ────────────────────────────────────────────────────

_MONTHS = {
    'january': 1,  'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3,    'mar': 3,
    'april': 4,    'apr': 4,
    'may': 5,
    'june': 6,     'jun': 6,
    'july': 7,     'jul': 7,
    'august': 8,   'aug': 8,
    'september': 9,'sep': 9,  'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11,'nov': 11,
    'december': 12,'dec': 12,
}

_DATE_RANGE_RE = re.compile(
    r'(\d{1,2}[-/]\w+[-/]\d{4})\s*(?:to|-)\s*(\d{1,2}[-/]\w+[-/]\d{4})',
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    r'\b(' + '|'.join(_MONTHS) + r')\s+(\d{4})\b',
    re.IGNORECASE,
)


def _extract_date_filter(question: str, df: pd.DataFrame):
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is None or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return None
    col = df[date_col]

    m = _DATE_RANGE_RE.search(question)
    if m:
        try:
            d1 = pd.to_datetime(m.group(1), dayfirst=True)
            d2 = pd.to_datetime(m.group(2), dayfirst=True)
            start, end = min(d1, d2), max(d1, d2)
            return (col >= start) & (col <= end)
        except Exception:
            pass

    m = _MONTH_YEAR_RE.search(question)
    if m:
        month = _MONTHS[m.group(1).lower()]
        year  = int(m.group(2))
        return (col.dt.month == month) & (col.dt.year == year)

    for name, num in _MONTHS.items():
        if re.search(rf'\b{name}\b', question, re.IGNORECASE):
            years = col.dt.year.dropna().unique()
            for yr in sorted(years, reverse=True):
                mask = (col.dt.month == num) & (col.dt.year == yr)
                if mask.sum() > 0:
                    return mask

    return None


_GROUPBY_HINTS = [
    (re.compile(r'\b(facility|facilities)\b', re.I), ['Facility Name', 'Facility']),
    (re.compile(r'\bshift\b',                  re.I), ['Shift']),
    (re.compile(r'\bequipment\b',              re.I), ['Equipment Name', 'Equipment']),
    (re.compile(r'\b(nature|issue type)\b',    re.I), ['Nature of Issue', 'Nature']),
    (re.compile(r'\btype\b',                   re.I), ['Type']),
    (re.compile(r'\bdate\b',                   re.I), ['WO Date', 'Date']),
]

_TARGET_HINTS = [
    (re.compile(r'\bdown\s*time\b',            re.I), ['Total Down Time', 'Downtime', 'Down Time']),
    (re.compile(r'\b(incident|breakdown|count)\b', re.I), None),
]


def _find_col(candidates: list, columns: list) -> Optional[str]:
    col_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    return None


def _detect_groupby(question: str, columns: list) -> Optional[str]:
    for pattern, candidates in _GROUPBY_HINTS:
        if pattern.search(question):
            col = _find_col(candidates, columns)
            if col:
                return col
    return None


def _detect_target(question: str, columns: list) -> Optional[str]:
    for pattern, candidates in _TARGET_HINTS:
        if pattern.search(question):
            if candidates is None:
                return None
            col = _find_col(candidates, columns)
            if col:
                return col
    return _find_col(['Total Down Time', 'Downtime', 'Down Time'], columns)


def _detect_aggregation(question: str) -> str:
    q = question.lower()
    if re.search(r'\b(average|avg|mean)\b', q):  return 'mean'
    if re.search(r'\b(max|maximum|highest|most)\b', q): return 'max'
    if re.search(r'\b(min|minimum|lowest|least)\b', q): return 'min'
    if re.search(r'\b(count|how many|number of)\b', q): return 'count'
    if re.search(r'\b(list|show|display|detail)\b', q): return 'list'
    return 'sum'


def _execute_query(question: str, df: pd.DataFrame) -> tuple:
    mask      = _extract_date_filter(question, df)
    working   = df[mask] if mask is not None else df
    date_desc = f" ({mask.sum()} rows matched)" if mask is not None else f" (all {len(df)} rows)"

    agg        = _detect_aggregation(question)
    group_col  = _detect_groupby(question, df.columns.tolist())
    target_col = _detect_target(question, df.columns.tolist())

    _safe_print(f"[ExcelTool] agg={agg} group={group_col} target={target_col}{date_desc}")

    if agg == 'list' and not group_col:
        cap = min(len(working), 100)
        return working.head(cap), f"Showing {cap} of {len(working)} rows"

    if agg == 'count' or target_col is None:
        if group_col:
            return (working.groupby(group_col).size().sort_values(ascending=False),
                    f"Count by {group_col}{date_desc}")
        return len(working), f"Total count{date_desc}"

    if agg == 'list' and group_col:
        agg = 'sum'

    if agg in ('max', 'min') and group_col:
        grouped   = working.groupby(group_col)[target_col].sum().sort_values(ascending=(agg == 'min'))
        best_grp  = grouped.index[0]
        best_val  = grouped.iloc[0]
        label     = 'Highest' if agg == 'max' else 'Lowest'
        return (f"{best_grp} with {best_val:.0f} minutes",
                f"{label} total {target_col} by {group_col}{date_desc}")

    if group_col:
        return (working.groupby(group_col)[target_col].sum().sort_values(ascending=False),
                f"Sum of {target_col} by {group_col}{date_desc}")

    return working[target_col].agg(agg), f"{agg.title()} of {target_col}{date_desc}"


# ── Result formatting ──────────────────────────────────────────────────────────

def _format_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        return result.to_string(index=False) if not result.empty else "No matching records."
    if isinstance(result, pd.Series):
        return ('\n'.join(f"{idx}: {val}" for idx, val in result.items())
                if not result.empty else "No matching records.")
    return str(result)


# ── LLM narration ──────────────────────────────────────────────────────────────

_PRESENT_SYSTEM = (
    "You are a concise data analyst. The user asked a question about a spreadsheet "
    "and the exact answer has already been computed. Present it clearly in natural language. "
    "Do NOT recalculate or add information not in the result. Be direct and specific."
)


def _present_result(question: str, raw_result: str, description: str, llm_manager) -> str:
    messages = [
        {"role": "system", "content": _PRESENT_SYSTEM},
        {"role": "user", "content": (
            f"User question: {question}\n\n"
            f"What was computed: {description}\n\n"
            f"Computed result:\n{raw_result}\n\n"
            "Present this to the user clearly:"
        )},
    ]
    return llm_manager.generate(messages, max_new_tokens=600, temperature=0.2)


# ── Public entry point ─────────────────────────────────────────────────────────

def query_excel(question: str, org_id: str, llm_manager) -> Optional[str]:
    """
    Answer `question` by running a rule-based pandas query on the org's
    Excel (.xls/.xlsx) or PDF table files.
    Returns None if not a structured-data question or no matching data found.
    """
    if not is_spreadsheet_query(question):
        return None

    candidates = []   # {'title', 'df', 'score', 'source_type'}

    # ── 1. Excel files ─────────────────────────────────────────────────────────
    excel_files = _find_excel_files(org_id)
    if excel_files:
        _safe_print(f"[ExcelTool] {len(excel_files)} Excel file(s) for org {org_id}")
    for f in excel_files:
        df = _load_df(f['path'])
        if df is None:
            continue
        candidates.append({
            'title': f['title'], 'df': df, 'source_type': 'Excel',
            'score': _score_df(df, f['title'], question),
        })

    # ── 2. PDF tables ──────────────────────────────────────────────────────────
    pdf_files = _find_pdf_files(org_id)
    if pdf_files:
        _safe_print(f"[ExcelTool] {len(pdf_files)} PDF file(s) — extracting tables (cached)")
    for f in pdf_files:
        try:
            tables = _extract_pdf_tables(f['path'])
        except Exception as exc:
            logger.warning(f"[ExcelTool] Skipping PDF {f['title']}: {exc}")
            continue
        for i, df in enumerate(tables):
            if df is None or df.empty or len(df.columns) < 2:
                continue
            title = f"{f['title']} (table {i+1})"
            candidates.append({
                'title': title, 'df': df, 'source_type': 'PDF',
                'score': _score_df(df, f['title'], question),
            })

    if not candidates:
        return None

    chosen = max(candidates, key=lambda x: x['score'])
    df, title = chosen['df'], chosen['title']
    _safe_print(f"[ExcelTool] Querying '{title}' [{chosen['source_type']}] ({len(df):,} rows)")

    try:
        result, description = _execute_query(question, df)
        raw_result = _format_result(result)
        _safe_print(f"[ExcelTool] {description} → {str(raw_result)[:120]}")
    except Exception as exc:
        logger.error(f"[ExcelTool] Query failed: {exc}", exc_info=True)
        return None

    try:
        return _present_result(question, raw_result, description, llm_manager)
    except Exception as exc:
        logger.error(f"[ExcelTool] Narration failed: {exc}")
        return f"**{description}**\n\n{raw_result}"
