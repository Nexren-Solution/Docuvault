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
- Reversed date ranges: always normalises start <= end, so "31-Mar to 1-Mar"
  returns the same result as "1-Mar to 31-Mar".
- Excel preferred: when an Excel file exists for the org, PDFs are skipped
  to avoid the same data being aggregated from two sources with different
  precision (this caused the 6232 vs 1625 contradiction in production).
- Multi-dimensional groupby: questions like "downtime by facility and type"
  group by both columns instead of picking just the first match.
- Deterministic narration: tables/series are returned as markdown directly;
  only scalar answers are paraphrased by the LLM, at temperature 0.

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
from typing import Optional, List, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(
            sys.stdout.encoding or 'utf-8', errors='replace'), **kwargs)


# -- DataFrame cache -----------------------------------------------------------
# Keyed by (path, mtime).  Thread-safe via a simple lock.

_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 600  # seconds
_CACHE_VERSION = 2 


def _cache_get(path: str):
    with _cache_lock:
        entry = _cache.get(path)
        if entry is None:
            return None
        cached_mtime, cached_ts, cached_ver, value = entry
        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            return None
        if current_mtime != cached_mtime or (time.time() - cached_ts) > _CACHE_TTL or cached_ver != _CACHE_VERSION:
            del _cache[path]
            return None
        return value


def _cache_set(path: str, value):
    with _cache_lock:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        _cache[path] = (mtime, time.time(), _CACHE_VERSION, value)


# -- Detection -----------------------------------------------------------------
# Greatly expanded vs the original. Fires on virtually any phrasing the user
# might use about tabular/quantitative breakdown data, including bare
# follow-ups like "with facility wise" or "list the same".

_AGG_KEYWORDS = re.compile(
    r'\b('
    r'total|sum|count|average|avg|mean|max|min|minimum|maximum|'
    r'highest|lowest|most|more|least|top|bottom|'
    r'how\s+many|number\s+of|list|show|display|detail|'
    r'breakdown|breakdowns|occurrence|occurrences|occ|incident|incidents|'
    r'group\s*by|per\s+\w+|by\s+\w+|wise|facility|facilities|shift|equipment|'
    r'against|categori[sz]e|distribution|repeated|repeat|phenomenon|phenomena|'
    r'downtime|down\s*time|breakdown\s*time|breakdown\s*duration|duration|'
    r'electrical|mechanical|instrumentation|nature|issue|type|'
    r'from\s+\d|between\s+\d|'
    r'jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|'
    r'jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?'
    r')\b',
    re.IGNORECASE,
)


def is_spreadsheet_query(question: str) -> bool:
    return bool(_AGG_KEYWORDS.search(question or ''))


# -- DB helpers ----------------------------------------------------------------

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
        xls = Document.objects.filter(flt, is_deleted=False, file__endswith='.xls')
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


# -- DataFrame loading (Excel) -------------------------------------------------

def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace inside string cells.
    Real-world data has '\\n' embedded inside cell values (e.g.
    'CC package stick with\\ndrum' shows up as a distinct phenomenon from
    'CC package stick with drum'). Collapse all internal whitespace so
    groupby/value_counts treat them as the same value."""
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                    .replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA})
                )
            except Exception:
                pass
    return df


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
            # Don't coerce known categorical/text columns to numeric. The
            # 'Facility Name' column in the breakdown sheet is numeric-looking
            # (220, 410...) but is semantically a label, and we want to keep
            # any future text values like "FAC-A" intact.
            if col.lower() in {'facility name', 'facility', 'shift', 'type',
                               'nature of issue', 'equipment name', 'equipment',
                               'issue summary', 'issue details', 'final action'}:
                continue
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
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
        df = _normalize_text_columns(df)
        df = _coerce_df_types(df)
        _cache_set(path, df)
        return df
    except Exception as exc:
        logger.warning(f"[ExcelTool] Failed to load Excel {path}: {exc}")
        return None


# -- PDF table extraction (cached) --------------------------------------------

def _build_df_from_rows(headers: list, rows: list) -> Optional[pd.DataFrame]:
    if len(rows) < 2 or len(headers) < 2:
        return None
    try:
        n = len(headers)
        padded = [r[:n] + [''] * max(0, n - len(r)) for r in rows]
        df = pd.DataFrame(padded, columns=headers)
        df.replace('', pd.NA, inplace=True)
        df = _normalize_text_columns(df)
        df = _coerce_df_types(df)
        return df
    except Exception as exc:
        logger.warning(f"[ExcelTool] _build_df_from_rows failed: {exc}")
        return None


def _looks_like_data_row(headers: list) -> bool:
    if not headers:
        return False
    try:
        float(headers[0].strip())
        return True
    except (ValueError, AttributeError):
        return False


def _extract_pdf_tables(path: str) -> list:
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

    groups: dict = {}
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
                    grp['rows'].append(headers)
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


# -- Scoring -------------------------------------------------------------------

def _score_df(df: pd.DataFrame, title: str, question: str) -> int:
    q = (question or '').lower()
    score = sum(2 for col in df.columns if str(col).lower() in q)
    for col in df.columns:
        for kw in ['date', 'time', 'facility', 'shift', 'equipment', 'total',
                  'down', 'nature', 'type', 'issue']:
            if kw in str(col).lower():
                score += 1
    score += min(len(df) // 100, 5)
    if title and title.lower() in q:
        score += 3
    return score


# -- Date filtering ------------------------------------------------------------

_MONTHS = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
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
    date_col = next((c for c in df.columns if 'date' in str(c).lower()), None)
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
        year = int(m.group(2))
        return (col.dt.month == month) & (col.dt.year == year)

    for name, num in _MONTHS.items():
        if re.search(rf'\b{name}\b', question, re.IGNORECASE):
            years = col.dt.year.dropna().unique()
            for yr in sorted(years, reverse=True):
                mask = (col.dt.month == num) & (col.dt.year == yr)
                if mask.sum() > 0:
                    return mask
    return None


# -- Column / dimension detection ---------------------------------------------
# (pattern, candidate column names) — first match in question wins,
# but _detect_groupby returns ALL matches (multi-dim groupby).

_GROUPBY_HINTS = [
    (re.compile(r'\b(facility|facilities|facility\s*wise|facility\s*name)\b', re.I),
        ['Facility Name', 'Facility']),
    (re.compile(r'\b(shift|shift\s*wise)\b', re.I),
        ['Shift']),
    (re.compile(r'\b(equipment|machine|equipment\s*name)\b', re.I),
        ['Equipment Name', 'Equipment', 'Machine']),
    (re.compile(r'\b(nature(\s+of\s+issue)?|electrical|mechanical|instrumentation)\b', re.I),
        ['Nature of Issue', 'Nature']),
    (re.compile(r'\b(issue\s*summary|phenomenon|phenomena|repeated)\b', re.I),
        ['Issue Summary', 'Phenomenon', 'Issue']),
    (re.compile(r'\btype\b', re.I),
        ['Type']),
    (re.compile(r'\bdate\b', re.I),
        ['WO Date', 'Date']),
]

# Order in which to prefer columns when the user says "duration"-style words.
_DURATION_HINT = re.compile(
    r'\b(down\s*time|breakdown\s*time|breakdown\s*duration|duration|downtime|minutes?|mins?)\b',
    re.I,
)
_COUNT_HINT = re.compile(
    r'\b(occurrence|occurrences|occ|incidents?|count|how\s+many|number\s+of|frequency)\b',
    re.I,
)

_DURATION_COLS = ['Total Down Time', 'Downtime', 'Down Time', 'Duration',
                  'Total Breakdown Time', 'Breakdown Time']


def _find_col(candidates: List[str], columns: list) -> Optional[str]:
    col_map = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    return None


def _detect_groupby(question: str, columns: list) -> List[str]:
    """Return ALL matching group-by columns (multi-dim groupby supported).
    Order is preserved so 'list equipment downtime with facility name and type'
    groups by Equipment, Facility, Type."""
    matches: List[str] = []
    seen = set()
    for pattern, candidates in _GROUPBY_HINTS:
        if pattern.search(question or ''):
            col = _find_col(candidates, columns)
            if col and col not in seen:
                matches.append(col)
                seen.add(col)
    return matches


def _detect_target(question: str, columns: list) -> Tuple[Optional[str], str]:
    """Return (target_column, intent) where intent is 'duration' or 'count'.
    'count' means user wants a row count, not the sum of any column."""
    q = question or ''
    has_duration = bool(_DURATION_HINT.search(q))
    has_count = bool(_COUNT_HINT.search(q))

    # Both keywords present: prefer duration unless count is far more dominant
    if has_duration:
        col = _find_col(_DURATION_COLS, columns)
        if col:
            return col, 'duration'
    if has_count:
        return None, 'count'
    # Default for breakdown data: assume duration when a duration column exists
    col = _find_col(_DURATION_COLS, columns)
    if col:
        return col, 'duration'
    return None, 'count'


def _detect_aggregation(question: str) -> str:
    q = (question or '').lower()
    if re.search(r'\b(average|avg|mean)\b', q): return 'mean'
    if re.search(r'\b(max|maximum|highest|most|more|top)\b', q): return 'max'
    if re.search(r'\b(min|minimum|lowest|least|bottom)\b', q): return 'min'
    if re.search(r'\b(count|how\s+many|number\s+of|occurrence|occurrences|occ)\b', q):
        return 'count'
    if re.search(r'\b(list|show|display|detail|categori[sz]e|distribution|breakdown\s+by)\b', q):
        return 'list'
    return 'sum'


# -- Query execution ----------------------------------------------------------

def _execute_query(question: str, df: pd.DataFrame) -> Tuple:
    mask = _extract_date_filter(question, df)
    working = df[mask] if mask is not None else df
    date_desc = (f" ({mask.sum()} rows matched)" if mask is not None
                 else f" (all {len(df):,} rows)")

    cols = df.columns.tolist()
    agg = _detect_aggregation(question)
    group_cols = _detect_groupby(question, cols)
    target_col, intent = _detect_target(question, cols)

    _safe_print(f"[ExcelTool] agg={agg} group={group_cols} target={target_col} "
                f"intent={intent}{date_desc}")

    # ---- Special: "repeated phenomenon" / top issue summaries ----
    q_lower = question.lower()
    if re.search(r'\b(repeated|repeat|phenomenon|phenomena|recurring|frequent)\b', q_lower) \
            and not group_cols:
        phenom_col = _find_col(['Issue Summary', 'Phenomenon', 'Issue'], cols)
        dur_col = _find_col(_DURATION_COLS, cols)
        if phenom_col:
            counts = working[phenom_col].value_counts(dropna=True)
            if dur_col is not None:
                durations = working.groupby(phenom_col)[dur_col].sum()
                dur_col_name = dur_col if dur_col.lower().startswith('total') \
                            else f'Total {dur_col}'
                out = pd.DataFrame({
                    dur_col_name: durations.astype(int),
                    'No. of Occurrences': counts,
                }).dropna().sort_values(dur_col_name, ascending=False).head(20)
                out[dur_col_name] = out[dur_col_name].astype(int)
                out['No. of Occurrences'] = out['No. of Occurrences'].astype(int)
                out['Avg Down Time / Breakdown'] = (
                    out[dur_col_name] / out['No. of Occurrences']
                ).round(2)
                out.index.name = phenom_col
                return out, (f"Top {len(out)} repeated phenomena by total downtime "
                            f"with occurrence count{date_desc}")
            top = counts.head(20)
            return top, f"Top {len(top)} repeated phenomena by occurrence{date_desc}"

    # ---- Categorize: list (or pivot) by Nature of Issue ----
    if re.search(r'\bcategori[sz]e|distribution\b', q_lower):
        nature_col = _find_col(['Nature of Issue', 'Nature'], cols)
        if nature_col:
            primary = group_cols[0] if group_cols else _find_col(
                ['Facility Name', 'Facility'], cols)
            dur_col = _find_col(_DURATION_COLS, cols)
            if primary and dur_col:
                pivot = working.pivot_table(
                    index=primary, columns=nature_col, values=dur_col,
                    aggfunc='sum', fill_value=0,
                )
                pivot['Total'] = pivot.sum(axis=1)
                pivot = pivot.sort_values('Total', ascending=False)
                pivot = pivot.astype(int)
                return pivot, (f"{dur_col} categorized by {nature_col}, "
                               f"per {primary}{date_desc}")

    # ---- Pure list of rows ----
    if agg == 'list' and not group_cols:
        cap = min(len(working), 100)
        return working.head(cap), f"Showing {cap:,} of {len(working):,} rows"

    # ---- Count branch ----
    if agg == 'count' or intent == 'count':
        if group_cols:
            counts = working.groupby(group_cols).size().sort_values(ascending=False)
            counts.name = 'Occurrences'
            return counts, f"Occurrences by {' & '.join(group_cols)}{date_desc}"
        return len(working), f"Total occurrences{date_desc}"

    # ---- Duration aggregation with multi-dim groupby ----
    if agg == 'list' and group_cols:
        agg = 'sum'

    if not target_col:
        # Nothing to sum; degrade gracefully to count
        if group_cols:
            counts = working.groupby(group_cols).size().sort_values(ascending=False)
            counts.name = 'Occurrences'
            return counts, f"Occurrences by {' & '.join(group_cols)}{date_desc}"
        return len(working), f"Total rows{date_desc}"

    if agg in ('max', 'min') and group_cols:
        grouped = (working.groupby(group_cols)[target_col].sum()
                   .sort_values(ascending=(agg == 'min')))
        if grouped.empty:
            return "No data", f"No matching rows{date_desc}"
        best_grp = grouped.index[0]
        best_val = grouped.iloc[0]
        label = 'Highest' if agg == 'max' else 'Lowest'
        # If user asked "which X has more/most..." they probably want the
        # answer plus a leaderboard for context.
        leaderboard = grouped.head(10).astype(int)
        leaderboard.name = target_col
        if isinstance(best_grp, tuple):
            best_grp = ' / '.join(str(x) for x in best_grp)
        # Avoid "highest total total down time" if target already starts with "Total"
        target_phrase = target_col.lower()
        if target_phrase.startswith('total '):
            target_phrase = target_phrase[6:]
        scalar_msg = (f"{best_grp} has the highest total {target_phrase} "
                      f"({int(best_val):,})." if agg == 'max'
                      else f"{best_grp} has the lowest total {target_phrase} "
                           f"({int(best_val):,}).")
        return (leaderboard,
                f"{label} {target_col} by {' & '.join(group_cols)}{date_desc}. "
                f"{scalar_msg}")

    if group_cols:
        # Sum + count side-by-side: this is what users almost always actually
        # want when they ask for "downtime by facility" -- they then ask for
        # the count next anyway.
        # Avoid double "Total Total Down Time" by stripping a leading "Total".
        sum_col_name = target_col
        if not sum_col_name.lower().startswith('total'):
            sum_col_name = f'Total {target_col}'
        grouped_sum = working.groupby(group_cols)[target_col].sum()
        grouped_cnt = working.groupby(group_cols).size()
        out = pd.DataFrame({
            sum_col_name: grouped_sum.astype(int),
            'No. of Occurrences': grouped_cnt.astype(int),
        }).sort_values(sum_col_name, ascending=False)
        out['Avg Down Time / Breakdown'] = (
            out[sum_col_name] / out['No. of Occurrences']
        ).round(2)
        return out, f"{target_col} by {' & '.join(group_cols)}{date_desc}"

    return working[target_col].agg(agg), f"{agg.title()} of {target_col}{date_desc}"


# -- Result formatting --------------------------------------------------------

def _format_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No matching records."
        return result.to_markdown() if hasattr(result, 'to_markdown') else result.to_string()
    if isinstance(result, pd.Series):
        if result.empty:
            return "No matching records."
        return result.to_markdown() if hasattr(result, 'to_markdown') else \
               '\n'.join(f"{idx}: {val}" for idx, val in result.items())
    return str(result)


def _is_structured(result) -> bool:
    if isinstance(result, pd.DataFrame):
        return True
    if isinstance(result, pd.Series):
        return len(result) > 1
    return False


# -- LLM narration (scalars only) ---------------------------------------------

_PRESENT_SYSTEM = (
    "You present a precomputed data result to the user.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY numbers, names, and labels from the 'Computed result' section.\n"
    "2. Never compute, infer, round, or change any number.\n"
    "3. Never invent facility names, equipment names, or any label not in the result.\n"
    "4. Be direct and concise — one or two sentences."
)


def _present_scalar(question: str, raw_result: str, description: str,
                    llm_manager) -> str:
    messages = [
        {"role": "system", "content": _PRESENT_SYSTEM},
        {"role": "user", "content": (
            f"User question: {question}\n\n"
            f"What was computed: {description}\n\n"
            f"Computed result (DO NOT MODIFY):\n{raw_result}\n\n"
            "Present this in one or two sentences:"
        )},
    ]
    return llm_manager.generate(messages, max_new_tokens=200, temperature=0.0)


# -- Public entry point -------------------------------------------------------

def query_excel(question: str, org_id: str, llm_manager) -> Optional[str]:
    """Answer `question` by running a rule-based pandas query on the org's
    Excel (.xls/.xlsx) or PDF table files.

    Returns None when the question doesn't match a tabular pattern or no
    matching data is found, in which case the caller should fall back to RAG.
    """
    if not is_spreadsheet_query(question):
        return None

    candidates = []

    # -- 1. Excel files (PREFERRED) -------------------------------------------
    # When an Excel file exists for the org we never fall through to PDF
    # tables. Mixing the two caused the 6232-vs-1625 contradiction in
    # production: same data, two slightly different aggregations from
    # different sources.
    excel_files = _find_excel_files(org_id)
    if excel_files:
        _safe_print(f"[ExcelTool] {len(excel_files)} Excel file(s) for org "
                    f"{org_id} -- skipping PDF tables")
        for f in excel_files:
            df = _load_df(f['path'])
            if df is None:
                continue
            candidates.append({
                'title': f['title'], 'df': df, 'source_type': 'Excel',
                'score': _score_df(df, f['title'], question),
            })
    else:
        # -- 2. PDF tables (only if no Excel) ---------------------------------
        pdf_files = _find_pdf_files(org_id)
        if pdf_files:
            _safe_print(f"[ExcelTool] {len(pdf_files)} PDF file(s) "
                        f"-- extracting tables (cached)")
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

    # Tiebreaker: prefer larger DataFrames (more rows = more complete).
    chosen = max(candidates, key=lambda x: (x['score'], len(x['df'])))
    df, title = chosen['df'], chosen['title']
    _safe_print(f"[ExcelTool] Querying '{title}' [{chosen['source_type']}] "
                f"({len(df):,} rows, {len(df.columns)} cols)")

    try:
        result, description = _execute_query(question, df)
        raw_result = _format_result(result)
        _safe_print(f"[ExcelTool] {description}")
    except Exception as exc:
        logger.error(f"[ExcelTool] Query failed: {exc}", exc_info=True)
        return None

    # For tables/series: skip the LLM and return markdown deterministically.
    # The LLM has been observed to invent facility numbers and corrupt
    # totals during paraphrasing, even at low temperature.
    if _is_structured(result):
        return f"**{description}**\n\n{raw_result}"

    # For scalar answers only: paraphrase via LLM at temperature 0.
    try:
        return _present_scalar(question, raw_result, description, llm_manager)
    except Exception as exc:
        logger.error(f"[ExcelTool] Narration failed: {exc}")
        return f"**{description}**\n\n{raw_result}"