"""
Schema-agnostic Excel/PDF table query tool.

DESIGN
------
This module turns natural-language questions into pandas operations on
spreadsheet/PDF tables. It is designed to generalize across arbitrary
schemas (manufacturing logs, sales data, HR records, inventory, etc.)
without hardcoded domain vocabulary.

Architecture: hybrid heuristic + LLM.

  1. FAST PATH — pure heuristics for trivially obvious queries
     (e.g. "how many rows", "total <only-numeric-column>",
     "list everything", "list X and count by Y"). Returns in <100ms.
     No LLM call.

  2. LLM PLANNER — for everything else, the LLM receives the table
     schema (column names, roles, sample values, distinct counts) plus
     the question, and returns a JSON execution plan. The plan is
     executed deterministically by pandas. ~1-2s per query.

  3. LLM PRESENTER — a small final call narrates scalar results in
     natural language. Skipped for tabular results (rendered directly).

The tool has NO hardcoded:
  - measure keyword list ("amount", "total", "qty", ...)
  - synonym groups (manufacturing/HR/finance vocabulary)
  - action-column tokens ("corrective", "action", "remark")
  - count-cue words ("incidents", "breakdowns", "tickets")

Column roles are inferred purely from data shape (cardinality, dtype,
parsability as date, ratio of unique values to row count). The LLM does
the semantic mapping from question words to columns.

CONTRACT
--------
- query_excel(question, org_id, llm_manager) -> Optional[str]
- Returns None when the question is not a spreadsheet query OR no tables
  found OR an unrecoverable error occurs (lets caller fall back to RAG).
- Returns "CLARIFY: ..." when the question is ambiguous and the user
  should be re-prompted.
- llm_manager.generate(messages, max_new_tokens, temperature) is required.

CACHING
-------
Loaded DataFrames are cached per-path with mtime + TTL invalidation.
The cache key is bumped via _CACHE_VERSION whenever the loader changes.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


# ─── small utilities ─────────────────────────────────────────────────────────

def _safe_print(*args, **kwargs):
    """print() that won't crash on Windows consoles with non-UTF-8 encodings."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        enc = sys.stdout.encoding or 'utf-8'
        print(text.encode(enc, errors='replace').decode(enc, errors='replace'),
              **kwargs)


_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _tokenize(name: str) -> List[str]:
    """Split a column name or string into lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(str(name).lower())


# ─── cache ───────────────────────────────────────────────────────────────────

_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 600  # seconds
_CACHE_VERSION = 8  # bumped due to planner/executor changes


def _cache_get(path: str):
    """Return cached value for path if mtime/TTL/version are still valid."""
    with _cache_lock:
        entry = _cache.get(path)
        if entry is None:
            return None
        cached_mtime, cached_ts, cached_ver, value = entry
        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            return None
        if (current_mtime != cached_mtime
                or (time.time() - cached_ts) > _CACHE_TTL
                or cached_ver != _CACHE_VERSION):
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


# ─── trigger detection ───────────────────────────────────────────────────────

_TRIGGER_RE = re.compile(
    r'\b('
    # aggregation words
    r'total|sum|count|average|avg|mean|max|min|minimum|maximum|'
    r'highest|lowest|most|more|least|top|bottom|biggest|smallest|'
    r'how\s+many|number\s+of|frequency|frequent|recurring|repeated|'
    r'common(est)?|distribution|categori[sz]e|occurrences?|'
    # grouping
    r'group\s*by|per\s+\w+|by\s+\w+|wise|for\s+each\s+\w+|against\s+\w+|'
    # comparison
    r'compare|comparison|versus|vs\.?|'
    # listing
    r'list|show|display|details?|'
    # generic data-question shape
    r'which\s+\w+|what\s+\w+|what\s+is\s+the|what\s+are\s+the|'
    # action/issue lookup phrasings
    r'used\s+for|taken\s+for|done\s+for|'
    # date hints
    r'jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|'
    r'jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?|'
    r'q[1-4]|fy\d{2,4}|between|since\s+\d|'
    r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
    r')\b',
    re.IGNORECASE,
)

_BINARY_TRIGGER_RE = re.compile(
    r'^\s*\w+\s+or\s+\w+\s*\??\s*$',
    re.IGNORECASE,
)


def is_spreadsheet_query(question: str) -> bool:
    """Cheap pre-filter so non-data questions skip the whole pipeline."""
    q = question or ''
    return bool(_TRIGGER_RE.search(q) or _BINARY_TRIGGER_RE.match(q))


# ─── DB lookup helpers ───────────────────────────────────────────────────────

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
        for doc in Document.objects.filter(flt, is_deleted=False,
                                           file__endswith='.pdf'):
            try:
                result.append({'title': doc.title, 'path': doc.file.path})
            except Exception:
                pass
        return result
    except Exception as exc:
        logger.warning(f"[ExcelTool] PDF DB lookup failed: {exc}")
        return []


# ─── data cleaning ───────────────────────────────────────────────────────────

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        series = df[col]

        if (pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)):
            try:
                df[col] = (series.astype(str)
                           .str.replace(r'[\r\n\t]+', ' ', regex=True)
                           .str.replace(r'  +', ' ', regex=True)
                           .str.strip()
                           .replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA}))
            except Exception:
                pass
            continue

        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            unique_count = non_null.nunique()
            ratio = unique_count / max(len(non_null), 1)
            if unique_count <= 200 and ratio < 0.05:
                try:
                    df[col] = (series.astype('Int64').astype(str)
                               .replace({'<NA>': pd.NA}))
                except Exception:
                    pass
    return df


# ─── schema inference ────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    role: str
    dtype: str
    unique_count: int
    null_ratio: float
    sample_values: list = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)


@dataclass
class TableSchema:
    columns: Dict[str, ColumnProfile]
    date_cols: List[str]
    measure_cols: List[str]
    dimension_cols: List[str]
    text_cols: List[str]
    id_cols: List[str]
    row_count: int


def _looks_like_id(name: str, unique_count: int, row_count: int) -> bool:
    tokens = _tokenize(name)
    if not tokens:
        return False
    last = tokens[-1]
    name_l = str(name).lower()
    looks_id_named = (last in {'id', 'no', 'code', 'number', 'num', 'ref', 'sno'}
                      or last.endswith('id')
                      or 's.no' in name_l or 'sl.no' in name_l)
    if looks_id_named and row_count > 0 and unique_count / row_count > 0.5:
        return True
    return False


def _try_parse_dates(series: pd.Series) -> Optional[pd.Series]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    sample = non_null.head(50)
    try:
        parsed = pd.to_datetime(sample, errors='coerce', dayfirst=True,
                                format='mixed')
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(sample, errors='coerce', dayfirst=True)
        except Exception:
            return None
    except Exception:
        return None
    if parsed.notna().sum() / len(sample) < 0.7:
        return None
    try:
        full = pd.to_datetime(series, errors='coerce', dayfirst=True,
                              format='mixed')
    except (TypeError, ValueError):
        try:
            full = pd.to_datetime(series, errors='coerce', dayfirst=True)
        except Exception:
            return None
    except Exception:
        return None
    return full


def _classify_column(name: str, series: pd.Series, row_count: int) -> ColumnProfile:
    tokens = _tokenize(name)
    non_null = series.dropna()
    null_ratio = 1 - (len(non_null) / row_count) if row_count else 1.0
    unique_count = non_null.nunique() if len(non_null) else 0
    samples = list(non_null.head(3).astype(str))

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnProfile(name, 'date', str(series.dtype), unique_count,
                             null_ratio, samples, tokens)

    is_textlike = (pd.api.types.is_object_dtype(series)
                   or pd.api.types.is_string_dtype(series))

    if is_textlike and len(non_null) > 0:
        parsed = _try_parse_dates(series)
        if parsed is not None:
            return ColumnProfile(name, 'date', str(series.dtype),
                                 unique_count, null_ratio, samples, tokens)

    if _looks_like_id(name, unique_count, row_count):
        return ColumnProfile(name, 'id', str(series.dtype), unique_count,
                             null_ratio, samples, tokens)

    if pd.api.types.is_numeric_dtype(series):
        try:
            non_null_num = series.dropna()
            if len(non_null_num) > 0:
                rng = float(non_null_num.max()) - float(non_null_num.min())
                wide_spread = (unique_count > 0
                               and rng / max(unique_count, 1) >= 1.0
                               and rng >= 20)
            else:
                wide_spread = False
        except (TypeError, ValueError):
            wide_spread = False

        if (not wide_spread
                and row_count > 0
                and unique_count <= max(20, int(row_count * 0.02))):
            return ColumnProfile(name, 'dimension', str(series.dtype),
                                 unique_count, null_ratio, samples, tokens)
        return ColumnProfile(name, 'measure', str(series.dtype),
                             unique_count, null_ratio, samples, tokens)

    if is_textlike and len(non_null) > 0:
        coerced = pd.to_numeric(series, errors='coerce')
        if coerced.notna().sum() / len(non_null) > 0.8:
            unique_after = coerced.dropna().nunique()
            try:
                non_null_num = coerced.dropna()
                if len(non_null_num) > 0:
                    rng = float(non_null_num.max()) - float(non_null_num.min())
                    wide_spread = (unique_after > 0
                                   and rng / max(unique_after, 1) >= 1.0
                                   and rng >= 20)
                    all_ints = bool((non_null_num % 1 == 0).all())
                    looks_like_code = (
                        all_ints
                        and unique_after <= 50
                        and row_count > 0
                        and unique_after / row_count < 0.05
                    )
                    if looks_like_code:
                        wide_spread = False
                else:
                    wide_spread = False
            except (TypeError, ValueError):
                wide_spread = False

            if (not wide_spread
                    and row_count > 0
                    and unique_after <= max(20, int(row_count * 0.02))):
                return ColumnProfile(name, 'dimension', 'numeric (coerced)',
                                     unique_after, null_ratio, samples, tokens)
            return ColumnProfile(name, 'measure', 'numeric (coerced)',
                                 unique_after, null_ratio, samples, tokens)

    if row_count > 0 and unique_count > 0:
        ratio = unique_count / row_count
        role = 'dimension' if ratio <= 0.5 and unique_count <= 200 else 'text'
    else:
        role = 'text'
    return ColumnProfile(name, role, str(series.dtype), unique_count,
                         null_ratio, samples, tokens)


def _profile_dataframe(df: pd.DataFrame) -> TableSchema:
    profiles: Dict[str, ColumnProfile] = {}
    n = len(df)
    for col in df.columns:
        profiles[col] = _classify_column(col, df[col], n)
    return TableSchema(
        columns=profiles,
        date_cols=[c for c, p in profiles.items() if p.role == 'date'],
        measure_cols=[c for c, p in profiles.items() if p.role == 'measure'],
        dimension_cols=[c for c, p in profiles.items() if p.role == 'dimension'],
        text_cols=[c for c, p in profiles.items() if p.role == 'text'],
        id_cols=[c for c, p in profiles.items() if p.role == 'id'],
        row_count=n,
    )


def _apply_schema(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    for col, prof in schema.columns.items():
        if col not in df.columns:
            continue
        is_textlike = (pd.api.types.is_object_dtype(df[col])
                       or pd.api.types.is_string_dtype(df[col]))
        if (prof.role == 'date'
                and not pd.api.types.is_datetime64_any_dtype(df[col])):
            parsed = _try_parse_dates(df[col])
            if parsed is not None:
                df[col] = parsed
        elif prof.role == 'measure' and is_textlike:
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.notna().sum() > df[col].notna().sum() * 0.5:
                df[col] = coerced
    return df


# ─── loaders ─────────────────────────────────────────────────────────────────

@dataclass
class TableHandle:
    title: str
    df: pd.DataFrame
    schema: TableSchema
    source_type: str


def _load_excel(path: str, file_title: str) -> List[TableHandle]:
    cached = _cache_get(path)
    if cached is not None:
        return cached
    handles: List[TableHandle] = []
    try:
        engine = 'openpyxl' if path.lower().endswith('.xlsx') else 'xlrd'
        xl = pd.ExcelFile(path, engine=engine)
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
            except Exception as exc:
                logger.warning(f"[ExcelTool] Sheet {sheet} of {path}: {exc}")
                continue
            if df is None or df.empty or len(df.columns) < 2:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            df = _clean_dataframe(df)
            schema = _profile_dataframe(df)
            df = _apply_schema(df, schema)
            schema = _profile_dataframe(df)
            label = (file_title if len(xl.sheet_names) == 1
                     else f"{file_title} :: {sheet}")
            handles.append(TableHandle(label, df, schema, 'Excel'))
    except Exception as exc:
        logger.warning(f"[ExcelTool] Failed to load Excel {path}: {exc}")
    _cache_set(path, handles)
    return handles


def _build_df_from_rows(headers: list, rows: list) -> Optional[pd.DataFrame]:
    if len(rows) < 1 or len(headers) < 2:
        return None
    try:
        n = len(headers)
        padded = [r[:n] + [''] * max(0, n - len(r)) for r in rows]
        df = pd.DataFrame(padded, columns=headers)
        df.replace('', pd.NA, inplace=True)
        return df
    except Exception as exc:
        logger.warning(f"[ExcelTool] _build_df_from_rows failed: {exc}")
        return None


def _looks_like_data_row(headers: list) -> bool:
    if not headers:
        return False
    try:
        float(str(headers[0]).strip())
        return True
    except (ValueError, AttributeError):
        return False


def _load_pdf_tables(path: str, file_title: str) -> List[TableHandle]:
    cached = _cache_get(path)
    if cached is not None:
        return cached
    try:
        import pdfplumber
    except ImportError:
        logger.warning("[ExcelTool] pdfplumber not installed; skipping PDFs")
        _cache_set(path, [])
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

    handles: List[TableHandle] = []
    idx = 0
    for grp in groups.values():
        df = _build_df_from_rows(grp['headers'], grp['rows'])
        if df is not None:
            df = _clean_dataframe(df)
            schema = _profile_dataframe(df)
            df = _apply_schema(df, schema)
            schema = _profile_dataframe(df)
            idx += 1
            label = (file_title if (len(groups) + len(ungrouped)) == 1
                     else f"{file_title} :: table {idx}")
            handles.append(TableHandle(label, df, schema, 'PDF'))
    for headers, rows in ungrouped:
        df = _build_df_from_rows(headers, rows)
        if df is not None:
            df = _clean_dataframe(df)
            schema = _profile_dataframe(df)
            df = _apply_schema(df, schema)
            schema = _profile_dataframe(df)
            idx += 1
            label = (file_title if (len(groups) + len(ungrouped)) == 1
                     else f"{file_title} :: table {idx}")
            handles.append(TableHandle(label, df, schema, 'PDF'))
    _cache_set(path, handles)
    return handles


# ─── table picker ────────────────────────────────────────────────────────────

_PICK_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'at',
    'to', 'from', 'for', 'and', 'or', 'with', 'by', 'per', 'show', 'list',
    'tell', 'me', 'give', 'what', 'which', 'how', 'many', 'much', 'has',
    'have', 'had', 'do', 'does', 'did', 'this', 'that', 'these', 'those',
    'i', 'we', 'you', 'they', 'it', 'most', 'least', 'top', 'bottom',
    'against', 'display', 'count', 'no',
}


def _pick_tokens(question: str) -> set:
    return {t for t in _tokenize(question)
            if t not in _PICK_STOPWORDS and len(t) > 1}


def _score_table(table: TableHandle, question: str) -> int:
    schema = table.schema
    q_tokens = _pick_tokens(question)
    if not q_tokens:
        return 0
    score = 0
    for col, prof in schema.columns.items():
        score += 3 * len(q_tokens & set(prof.tokens))
    score += 2 * len(q_tokens & set(_tokenize(table.title)))
    try:
        for col in schema.dimension_cols:
            uniques = table.df[col].dropna().astype(str).unique()
            if len(uniques) > 500:
                continue
            qlow = question.lower()
            for v in uniques:
                vs = str(v).lower().strip()
                if len(vs) >= 4 and re.search(rf'\b{re.escape(vs)}\b', qlow):
                    score += 4
                    break
    except Exception:
        pass
    if schema.measure_cols:
        score += 1
    score += min(schema.row_count // 200, 3)
    return score


# ─── execution plan ──────────────────────────────────────────────────────────

@dataclass
class Plan:
    """
    Deterministic execution plan produced by either the fast-path or the LLM.

    SINGLE-METRIC: primary {agg, measure} drives the result.
    MULTI-METRIC:  extra_aggs adds more columns to the grouped output.
                   Each entry: {"agg": "count"|"sum"|"mean"|"min"|"max",
                                "measure": "<col>"|null,
                                "label": "<display name>"}

    LIMIT SEMANTICS:
      limit = 0  → return all groups (with internal safety cap)
      limit > 0  → return top `limit` groups
    """
    intent: str
    agg: str
    measure: Optional[str] = None
    target_col: Optional[str] = None
    group_cols: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    or_filter_groups: List[Dict[str, Any]] = field(default_factory=list)
    limit: int = 0
    rationale: str = ''
    extra_aggs: List[Dict[str, Any]] = field(default_factory=list)


# Safety caps applied AFTER plan.limit is consulted
_MAX_GROUPED_ROWS = 5000
_MAX_LIST_ROWS = 200


# ─── fast-path heuristics ────────────────────────────────────────────────────

_GLOBAL_COUNT_RE = re.compile(
    r'^\s*(how\s+many\s+(rows?|records?|entries?)(\s+are\s+there)?'
    r'|total\s+(rows?|records?|count)'
    r'|count(\s+all)?(\s+rows?)?'
    r'|number\s+of\s+(rows?|records?|entries?))\s*\??\s*$',
    re.IGNORECASE,
)

_LIST_ALL_RE = re.compile(
    r'^\s*(list|show|display)(\s+all)?(\s+(rows?|records?|entries?|data))?\s*\??\s*$',
    re.IGNORECASE,
)

# Multi-metric fast-path patterns. Matches sentences like:
#   "list the downtime and count against facility name"
#   "show downtime and occurrences by facility"
#   "list downtime and display the count against facility name"
#   "downtime and number of occurrences per facility"
#   "list X, count by Y"
_MULTI_METRIC_RE = re.compile(
    r'^\s*'
    r'(?:list|show|display|give|tell)?\s*'
    r'(?:the\s+|me\s+)?'
    r'(?P<measure>[\w\s]+?)\s+'                       # measure phrase
    r'(?:and|,)\s+'                                   # connector
    r'(?:display\s+(?:the\s+)?|show\s+(?:the\s+)?|the\s+)?'
    r'(?:no\.?\s*of\s+|number\s+of\s+)?'
    r'(?P<count_word>count|counts|occurrences?|incidents?|records?|entries?|frequency)\s+'
    r'(?:against|by|per|for\s+each|wise)\s+'          # group connector
    r'(?P<group>[\w\s]+?)'
    r'\s*\??\s*$',
    re.IGNORECASE,
)

# "top N <measure> and count by <group>" variant
_TOP_N_MULTI_RE = re.compile(
    r'^\s*top\s+(?P<n>\d+)\s+'
    r'(?P<group>[\w\s]+?)\s+by\s+'
    r'(?P<measure>[\w\s]+?)\s+'
    r'(?:and\s+(?:the\s+)?'
    r'(?:no\.?\s*of\s+|number\s+of\s+)?'
    r'(?P<count_word>count|counts|occurrences?))?\s*\??\s*$',
    re.IGNORECASE,
)

# Single-metric groupby: "list X against/by Y" / "show X per Y" / "X by Y"
# Tolerates typos in "against" like "aganist", "agianst", "againt" via fuzzy
# pattern (any word starting with 'ag' and ending in 't' counts as "against").
_SINGLE_METRIC_GROUPBY_RE = re.compile(
    r'^\s*'
    r'(?:list|show|display|give|tell)?\s*'
    r'(?:the\s+|me\s+)?'
    r'(?P<measure>[\w\s]+?)\s+'
    r'(?:ag\w*t|by|per|for\s+each|wise)\s+'   # against (or typo), by, per, etc.
    r'(?P<group>[\w\s]+?)'
    r'\s*\??\s*$',
    re.IGNORECASE,
)


def _resolve_column(phrase: str, candidates: List[str],
                    schema: TableSchema) -> Optional[str]:
    """Find the column whose tokens best match the given phrase."""
    phrase_tokens = set(_tokenize(phrase)) - _PICK_STOPWORDS
    if not phrase_tokens:
        return None
    best_col = None
    best_overlap = 0
    for col in candidates:
        col_tokens = set(schema.columns[col].tokens)
        overlap = len(phrase_tokens & col_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col
    return best_col if best_overlap > 0 else None


def _fast_path(question: str, table: TableHandle) -> Optional[Plan]:
    """Return a Plan for trivially obvious queries, else None."""
    q = (question or '').strip()
    schema = table.schema

    if _GLOBAL_COUNT_RE.match(q):
        return Plan(intent='count_rows', agg='count',
                    rationale='trivial global row count')

    if _LIST_ALL_RE.match(q):
        return Plan(intent='list', agg='list', limit=_MAX_LIST_ROWS,
                    rationale='trivial list-all')

    # Multi-metric: "list <measure> and count against/by <group>"
    m = _MULTI_METRIC_RE.match(q)
    if m:
        measure_phrase = m.group('measure').strip()
        group_phrase = m.group('group').strip()

        # Filter out leading verbs the regex may have absorbed into "measure"
        measure_tokens = [t for t in _tokenize(measure_phrase)
                          if t not in _PICK_STOPWORDS]
        measure_phrase = ' '.join(measure_tokens)

        measure_col = _resolve_column(measure_phrase,
                                      schema.measure_cols, schema)
        group_col = _resolve_column(group_phrase,
                                    schema.dimension_cols + schema.id_cols,
                                    schema)

        if measure_col and group_col:
            return Plan(
                intent='multi_metric_groupby',
                agg='sum',
                measure=measure_col,
                group_cols=[group_col],
                extra_aggs=[{
                    'agg': 'count', 'measure': None, 'label': 'Occurrences',
                }],
                limit=0,
                rationale=f'fast-path multi-metric: sum({measure_col}) + count'
                          f' by {group_col}',
            )

    # Single-metric groupby: "list/show/display X against/by Y"
    # (handles typos like "aganist" via fuzzy connector match)
    m = _SINGLE_METRIC_GROUPBY_RE.match(q)
    if m:
        measure_phrase = m.group('measure').strip()
        group_phrase = m.group('group').strip()

        measure_tokens = [t for t in _tokenize(measure_phrase)
                          if t not in _PICK_STOPWORDS]
        measure_phrase = ' '.join(measure_tokens)

        measure_col = _resolve_column(measure_phrase,
                                      schema.measure_cols, schema)
        group_col = _resolve_column(group_phrase,
                                    schema.dimension_cols + schema.id_cols,
                                    schema)

        if measure_col and group_col:
            return Plan(
                intent='single_metric_groupby',
                agg='sum',
                measure=measure_col,
                group_cols=[group_col],
                limit=0,
                rationale=f'fast-path: sum({measure_col}) by {group_col}',
            )

    # Top N variant
    m = _TOP_N_MULTI_RE.match(q)
    if m:
        try:
            n = int(m.group('n'))
        except (TypeError, ValueError):
            n = 10
        measure_phrase = m.group('measure').strip()
        group_phrase = m.group('group').strip()
        measure_col = _resolve_column(measure_phrase,
                                      schema.measure_cols, schema)
        group_col = _resolve_column(group_phrase,
                                    schema.dimension_cols + schema.id_cols,
                                    schema)
        if measure_col and group_col:
            extras = []
            if m.group('count_word'):
                extras.append({'agg': 'count', 'measure': None,
                               'label': 'Occurrences'})
            return Plan(
                intent='top_n_multi_metric',
                agg='sum',
                measure=measure_col,
                group_cols=[group_col],
                extra_aggs=extras,
                limit=n,
                rationale=f'fast-path top-{n}',
            )

    # Single-measure global aggregation
    if len(schema.measure_cols) == 1:
        m = re.match(
            r'^\s*(total|sum|average|avg|mean|max|maximum|highest|'
            r'min|minimum|lowest)\s+(\w+(?:\s+\w+){0,3})\s*\??\s*$',
            q, re.IGNORECASE,
        )
        if m:
            agg_word = m.group(1).lower()
            measure_phrase = m.group(2).lower()
            measure_col = schema.measure_cols[0]
            measure_tokens = set(schema.columns[measure_col].tokens)
            phrase_tokens = set(_tokenize(measure_phrase))
            if phrase_tokens & measure_tokens:
                for dim in schema.dimension_cols + schema.text_cols:
                    if phrase_tokens & set(schema.columns[dim].tokens):
                        return None
                agg_map = {
                    'total': 'sum', 'sum': 'sum',
                    'average': 'mean', 'avg': 'mean', 'mean': 'mean',
                    'max': 'max', 'maximum': 'max', 'highest': 'max',
                    'min': 'min', 'minimum': 'min', 'lowest': 'min',
                }
                return Plan(
                    intent='global_agg',
                    agg=agg_map[agg_word],
                    measure=measure_col,
                    rationale=f'global {agg_map[agg_word]} of single measure',
                )

    return None


# ─── LLM planner ─────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You translate a user question about tabular data into a JSON execution plan.

OUTPUT FORMAT — return ONLY valid JSON, no commentary, no markdown fences:

{
  "agg": "count" | "sum" | "mean" | "min" | "max" | "value_counts" | "list",
  "measure": "<column name or null>",
  "target_col": "<column name or null>",
  "group_cols": ["<column name>", ...],
  "extra_aggs": [
    {"agg": "count"|"sum"|"mean"|"min"|"max", "measure": "<col>"|null, "label": "<name>"}
  ],
  "filters": [
    {"col": "<column name>", "op": "eq",       "value":  "<value>"},
    {"col": "<column name>", "op": "ne",       "value":  "<value>"},
    {"col": "<column name>", "op": "in",       "values": ["<value>", ...]},
    {"col": "<column name>", "op": "not_in",   "values": ["<value>", ...]},
    {"col": "<column name>", "op": "contains", "pattern": "<substring>"},
    {"col": "<column name>", "op": "contains_any", "patterns": ["<substring>", ...]},
    {"col": "<column name>", "op": "gt"|"gte"|"lt"|"lte", "value": <number>},
    {"col": "<column name>", "op": "between", "low": <number>, "high": <number>},
    {"col": "<column name>", "op": "year",   "value": <int>},
    {"col": "<column name>", "op": "month",  "year": <int>, "month": <int>},
    {"col": "<column name>", "op": "between_dates", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
  ],
  "or_filter_groups": [
    {"any_of": [ {filter}, {filter} ]}
  ],
  "limit": <int>,
  "rationale": "<one sentence>"
}

How filters combine: top-level "filters" are AND'd. Each "or_filter_groups"
entry is a group whose inner filters are OR'd, then AND'd with everything else.

CRITICAL RULES:

1. Use ONLY column names from the provided schema. Match exactly (case-sensitive).

2. "agg" — primary aggregation:
   - "count" — count rows. Use when the user asks "how many", "frequency",
     or names a row-counting concept (incidents, occurrences, breakdowns,
     tickets, complaints — IF the data is a log of those). measure MUST be null.
   - "sum"/"mean"/"min"/"max" — aggregate a NUMERIC measure column.
     measure MUST be a column with role "measure".
   - "value_counts" — frequency of distinct values in one column.
     Set target_col. measure and group_cols MUST be empty/null.
   - "list" — return matching rows. Use sparingly.

3. ⚠️ MULTI-METRIC REQUESTS — IMPORTANT ⚠️
   When the user asks for MULTIPLE metrics in one query, put the primary
   metric in "agg"/"measure" and additional metrics in "extra_aggs".

   Phrases that signal multi-metric:
     - "X and count" / "X and occurrences" / "X and no. of Y"
     - "X and Y" where both are aggregations or one is a counting word
     - "list X and display count" / "X along with count"
     - "sum and average" / "min, max, and average"
     - "total X and number of Y"

   Common counting words: count, occurrences, incidents, records, entries,
   frequency, number of rows.

   EXAMPLE — "List the downtime and No. of Occurrences against facility name":
   {
     "agg": "sum",
     "measure": "Total Down Time",
     "group_cols": ["Facility Name"],
     "extra_aggs": [
       {"agg": "count", "measure": null, "label": "Occurrences"}
     ],
     "limit": 0,
     "rationale": "Total downtime + row count per facility"
   }

   EXAMPLE — "Show me total and average sales by region":
   {
     "agg": "sum",
     "measure": "Sales",
     "group_cols": ["Region"],
     "extra_aggs": [
       {"agg": "mean", "measure": "Sales", "label": "Average Sales"}
     ],
     "limit": 0
   }

   EXAMPLE — "Min, max and average response time by team":
   {
     "agg": "min",
     "measure": "Response Time",
     "group_cols": ["Team"],
     "extra_aggs": [
       {"agg": "max",  "measure": "Response Time", "label": "Max Response Time"},
       {"agg": "mean", "measure": "Response Time", "label": "Avg Response Time"}
     ],
     "limit": 0
   }

   When BOTH metrics are counts of different things, the primary is count
   and extras are skipped — you only ever have ONE row-count.

4. "limit":
   - 0 = return ALL groups (use this when grouping by a low-cardinality
     dimension and the user did NOT say "top N" or "first N").
   - N > 0 = return top N (use when user says "top 5", "first 10", etc.).
   - For "value_counts" without a "top N", default to 20.
   - For "list" without explicit count, default to 50.

5. "group_cols":
   - Empty for global aggregates ("total downtime").
   - One column for "by X", "per X", "against X", "for each X".
   - Two only if the user explicitly asks for a cross-tab ("by X and Y").

6. "filters" — translate every constraint:
   - "in March 2024" → {"op":"month","year":2024,"month":3} on date column
   - "for shift A" → {"op":"eq","value":"SHIFTA"} (use casing from samples)
   - "other than X", "excluding X" → {"op":"ne","value":"X"} or "not_in"

7. ⚠️ ISSUE / TOPIC KEYWORD FILTERING ⚠️
   When filtering on a topic that could appear in any free-text column
   (vacuum, calibration, jam, scanner, leak, etc.), search ALL relevant
   free-text columns using or_filter_groups.

   Free-text columns: role "text", or names containing Issue, Detail,
   Description, Problem, Summary, Nature, Comment, Remark, Note.

   EXAMPLE — "Which equipment had calibration issues?":
     "or_filter_groups": [{
       "any_of": [
         {"col": "Issue Summary",  "op": "contains", "pattern": "calibration"},
         {"col": "Issue Details",  "op": "contains", "pattern": "calibration"}
       ]
     }],
     "group_cols": ["Equipment Name"],
     "agg": "count"

8. AMBIGUITY-BREAKING:
   a) "Most/highest <thing>" where <thing> names a row type (incidents,
      breakdowns, complaints) AND data is a log of those → agg="count".
   b) "Most common <X>" / "frequent <X>" → agg="value_counts", target_col=X.
   c) "Action/fix taken for <issue>" — filter issue keyword on issue columns,
      then value_counts on action column.
   d) "Which <dimension>" → exactly ONE group_col matching that dimension.
   e) If a dimension has nearly-uniform values (one value ≥95% of rows),
      do NOT filter on that dimension even if the question mentions the value.

9. ⚠️ BINARY "X OR Y?" QUESTIONS ⚠️
   "X or Y?" / "is it X or Y?" / "X versus Y?" → use:
     "agg": "value_counts", "target_col": "<dimension column>"
     "filters": [{"col": "<column>", "op": "in", "values": ["X","Y"]}]
   Let the data decide which is higher.

10. If the question is too ambiguous, return:
    {"clarify": "<what you need from the user>"}

Return ONLY the JSON object. No prose before or after.
"""


def _build_schema_for_llm(table: TableHandle, max_samples: int = 5) -> dict:
    cols = []
    for name, prof in table.schema.columns.items():
        try:
            uniques = table.df[name].dropna().astype(str).unique()
            samples = [str(v) for v in uniques[:max_samples]]
        except Exception:
            samples = prof.sample_values[:max_samples]
        col_desc = {
            'name': name,
            'role': prof.role,
            'distinct_values': prof.unique_count,
            'samples': samples,
        }
        if prof.role == 'dimension' and prof.unique_count <= 30:
            try:
                all_vals = table.df[name].dropna().astype(str).unique().tolist()
                col_desc['all_values'] = all_vals[:30]
            except Exception:
                pass
        cols.append(col_desc)

    return {
        'table': table.title,
        'row_count': table.schema.row_count,
        'columns': cols,
    }


def _strip_json_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    return s.strip()


def _llm_plan(question: str, table: TableHandle, llm_manager
              ) -> Tuple[Optional[Plan], Optional[str]]:
    if llm_manager is None:
        return None, None

    schema_payload = _build_schema_for_llm(table)
    user_msg = (
        f"Question: {question}\n\n"
        f"Schema:\n{json.dumps(schema_payload, default=str, indent=2)}\n\n"
        "Return the JSON plan."
    )

    try:
        raw = llm_manager.generate(
            [{"role": "system", "content": _PLANNER_SYSTEM},
             {"role": "user", "content": user_msg}],
            max_new_tokens=800,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning(f"[ExcelTool] LLM planner call failed: {exc}")
        return None, None

    raw = _strip_json_fences(raw or '')
    try:
        plan_dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(f"[ExcelTool] LLM returned invalid JSON: {exc}\n{raw[:300]}")
        return None, None

    if isinstance(plan_dict, dict) and isinstance(plan_dict.get('clarify'), str):
        msg = plan_dict['clarify'].strip()
        return None, (msg or "Could you rephrase your question?")

    plan = _validate_plan(plan_dict, table)
    if plan is None:
        return None, None
    return plan, None


_VALID_OPS = {
    'eq', 'ne', 'in', 'not_in', 'contains', 'contains_any',
    'gt', 'gte', 'lt', 'lte', 'between',
    'year', 'month', 'between_dates',
}


def _validate_filter_list(raw_filters: Any, valid_cols: set) -> List[Dict[str, Any]]:
    if not isinstance(raw_filters, list):
        return []
    out: List[Dict[str, Any]] = []
    for f in raw_filters:
        if not isinstance(f, dict):
            continue
        if f.get('col') in valid_cols and f.get('op') in _VALID_OPS:
            out.append(f)
    return out


def _default_limit_for_plan(plan_dict: dict, group_cols: List[str],
                            agg: str, schema: TableSchema) -> int:
    """Pick a sensible default limit when the LLM didn't specify."""
    if 'limit' in plan_dict and plan_dict['limit'] is not None:
        try:
            return max(0, int(plan_dict['limit']))
        except (TypeError, ValueError):
            pass
    if agg == 'list':
        return _MAX_LIST_ROWS
    if agg == 'value_counts':
        return 20
    if group_cols:
        # For grouped aggregations, default to "all" if the group has few values
        try:
            max_card = max(schema.columns[g].unique_count for g in group_cols
                           if g in schema.columns)
            if max_card <= 100:
                return 0
            return 50
        except (ValueError, KeyError):
            return 0
    return 0


def _validate_plan(plan_dict: dict, table: TableHandle) -> Optional[Plan]:
    if not isinstance(plan_dict, dict):
        return None
    valid_cols = set(table.schema.columns.keys())
    schema = table.schema

    agg = plan_dict.get('agg', 'count')
    if agg not in {'count', 'sum', 'mean', 'min', 'max', 'value_counts', 'list'}:
        agg = 'count'

    measure = plan_dict.get('measure')
    if measure and measure not in valid_cols:
        measure = None
    if agg in {'sum', 'mean', 'min', 'max'} and measure:
        if schema.columns[measure].role != 'measure':
            try:
                if not pd.api.types.is_numeric_dtype(table.df[measure]):
                    measure = None
            except Exception:
                measure = None

    target_col = plan_dict.get('target_col')
    if target_col and target_col not in valid_cols:
        target_col = None

    group_cols_raw = plan_dict.get('group_cols') or []
    if not isinstance(group_cols_raw, list):
        group_cols_raw = []
    group_cols = [c for c in group_cols_raw if c in valid_cols][:3]

    filters = _validate_filter_list(plan_dict.get('filters'), valid_cols)

    raw_or_groups = plan_dict.get('or_filter_groups') or []
    if not isinstance(raw_or_groups, list):
        raw_or_groups = []
    or_filter_groups: List[Dict[str, Any]] = []
    for grp in raw_or_groups:
        if not isinstance(grp, dict):
            continue
        any_of = _validate_filter_list(grp.get('any_of'), valid_cols)
        if any_of:
            or_filter_groups.append({'any_of': any_of})

    # Validate extra_aggs
    raw_extras = plan_dict.get('extra_aggs') or []
    if not isinstance(raw_extras, list):
        raw_extras = []
    extra_aggs: List[Dict[str, Any]] = []
    seen_keys = set()
    # Avoid duplicating the primary metric
    primary_key = (agg, measure)
    seen_keys.add(primary_key)

    for e in raw_extras:
        if not isinstance(e, dict):
            continue
        e_agg = e.get('agg')
        if e_agg not in {'count', 'sum', 'mean', 'min', 'max'}:
            continue
        e_measure = e.get('measure')
        if e_measure is not None and e_measure not in valid_cols:
            continue
        if e_agg in {'sum', 'mean', 'min', 'max'}:
            if not e_measure:
                continue
            # measure must be numeric-coercible
            try:
                if (schema.columns[e_measure].role != 'measure'
                        and not pd.api.types.is_numeric_dtype(table.df[e_measure])):
                    continue
            except Exception:
                continue
        key = (e_agg, e_measure)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if e_agg == 'count':
            default_label = 'Count'
        elif e_measure:
            default_label = f"{e_agg.title()} {e_measure}"
        else:
            default_label = e_agg.title()
        e_label = str(e.get('label') or default_label).strip() or default_label
        extra_aggs.append({
            'agg': e_agg,
            'measure': e_measure,
            'label': e_label,
        })

    limit = _default_limit_for_plan(plan_dict, group_cols, agg, schema)
    # Internal upper bound (independent of user-facing 0=all semantics)
    if limit > _MAX_GROUPED_ROWS:
        limit = _MAX_GROUPED_ROWS

    return Plan(
        intent=plan_dict.get('intent') or agg,
        agg=agg,
        measure=measure,
        target_col=target_col,
        group_cols=group_cols,
        filters=filters,
        or_filter_groups=or_filter_groups,
        limit=limit,
        rationale=str(plan_dict.get('rationale', '')),
        extra_aggs=extra_aggs,
    )


# ─── execution ───────────────────────────────────────────────────────────────

def _eval_filter(df: pd.DataFrame, f: Dict[str, Any]
                 ) -> Tuple[Optional[pd.Series], Optional[str]]:
    col = f.get('col')
    op = f.get('op')
    if col is None or col not in df.columns:
        return None, None
    try:
        series = df[col]
        if op == 'eq':
            v = str(f['value']).lower()
            mask = series.astype(str).str.lower() == v
            note = f"{col} = {f['value']}"
        elif op == 'ne':
            v = str(f['value']).lower()
            mask = series.astype(str).str.lower() != v
            note = f"{col} ≠ {f['value']}"
        elif op == 'in':
            vals = [str(v).lower() for v in f.get('values', [])]
            mask = series.astype(str).str.lower().isin(vals)
            note = f"{col} in {f['values']}"
        elif op == 'not_in':
            vals = [str(v).lower() for v in f.get('values', [])]
            mask = ~series.astype(str).str.lower().isin(vals)
            note = f"{col} not in {f['values']}"
        elif op == 'contains':
            pat = str(f['pattern'])
            mask = series.astype(str).str.contains(
                pat, case=False, na=False, regex=False)
            note = f"{col} contains '{pat}'"
        elif op == 'contains_any':
            patterns = [str(p) for p in f.get('patterns', [])]
            if not patterns:
                return None, None
            mask = pd.Series(False, index=df.index)
            for p in patterns:
                mask = mask | series.astype(str).str.contains(
                    p, case=False, na=False, regex=False)
            note = f"{col} contains any of {patterns}"
        elif op in {'gt', 'gte', 'lt', 'lte'}:
            v = pd.to_numeric(f['value'], errors='coerce')
            num = pd.to_numeric(series, errors='coerce')
            mask = {
                'gt': num > v, 'gte': num >= v,
                'lt': num < v, 'lte': num <= v,
            }[op]
            op_sym = {'gt': '>', 'gte': '≥', 'lt': '<', 'lte': '≤'}[op]
            note = f"{col} {op_sym} {f['value']}"
        elif op == 'between':
            lo = pd.to_numeric(f['low'], errors='coerce')
            hi = pd.to_numeric(f['high'], errors='coerce')
            num = pd.to_numeric(series, errors='coerce')
            mask = (num >= lo) & (num <= hi)
            note = f"{col} ∈ [{f['low']},{f['high']}]"
        elif op == 'year':
            if not pd.api.types.is_datetime64_any_dtype(series):
                return None, None
            mask = series.dt.year == int(f['value'])
            note = f"{col} year={f['value']}"
        elif op == 'month':
            if not pd.api.types.is_datetime64_any_dtype(series):
                return None, None
            mask = ((series.dt.year == int(f['year']))
                    & (series.dt.month == int(f['month'])))
            note = f"{col} {f['year']}-{int(f['month']):02d}"
        elif op == 'between_dates':
            if not pd.api.types.is_datetime64_any_dtype(series):
                return None, None
            start = pd.to_datetime(f['start'])
            end = pd.to_datetime(f['end'])
            mask = (series >= start) & (series <= end)
            note = f"{col} {f['start']}…{f['end']}"
        else:
            return None, None
        return mask.fillna(False), note
    except Exception as exc:
        logger.warning(f"[ExcelTool] filter {f} failed: {exc}")
        return None, None


def _apply_filters(df: pd.DataFrame,
                   filters: List[Dict[str, Any]],
                   or_filter_groups: Optional[List[Dict[str, Any]]] = None
                   ) -> Tuple[pd.DataFrame, List[str]]:
    working = df
    notes: List[str] = []

    for f in filters:
        mask, note = _eval_filter(working, f)
        if mask is None:
            continue
        working = working[mask]
        if note:
            notes.append(note)

    or_filter_groups = or_filter_groups or []
    for grp in or_filter_groups:
        any_of = grp.get('any_of') or []
        if not any_of:
            continue
        combined_mask: Optional[pd.Series] = None
        inner_notes: List[str] = []
        for f in any_of:
            mask, note = _eval_filter(working, f)
            if mask is None:
                continue
            combined_mask = mask if combined_mask is None else (combined_mask | mask)
            if note:
                inner_notes.append(note)
        if combined_mask is not None and inner_notes:
            working = working[combined_mask]
            notes.append("(" + " OR ".join(inner_notes) + ")")

    return working, notes


def _format_int_or_round(s: pd.Series) -> pd.Series:
    try:
        if (s.dropna() % 1 == 0).all():
            return s.astype('Int64')
    except (TypeError, ValueError):
        pass
    try:
        return s.round(2)
    except Exception:
        return s


def _apply_limit(obj, limit: int):
    """Apply limit to a DataFrame or Series. limit=0 means no truncation
    (subject to safety cap _MAX_GROUPED_ROWS)."""
    if limit <= 0:
        # safety cap
        if hasattr(obj, '__len__') and len(obj) > _MAX_GROUPED_ROWS:
            return obj.head(_MAX_GROUPED_ROWS)
        return obj
    return obj.head(limit)


def _action_label(agg: str) -> str:
    return {'sum': 'Total', 'mean': 'Average',
            'min': 'Minimum', 'max': 'Maximum',
            'count': 'Count'}.get(agg, agg.title())


def _display_metric_name(agg: str, measure: Optional[str]) -> str:
    """Build a friendly column name like 'Total Down Time'.
    Avoid double-prefix when the measure already starts with the action word."""
    action = _action_label(agg)
    if not measure:
        return action
    if measure.lower().startswith(action.lower()):
        return measure
    return f"{action} {measure}"


def _compute_grouped_agg(working: pd.DataFrame, group_cols: List[str],
                         agg: str, measure: Optional[str]) -> pd.Series:
    """Compute one grouped aggregation. Returns a Series indexed by group."""
    if agg == 'count':
        return working.groupby(group_cols, dropna=False).size()
    measure_series = pd.to_numeric(working[measure], errors='coerce')
    grouped = working.assign(__m=measure_series).groupby(
        group_cols, dropna=False)['__m']
    return {'sum': grouped.sum, 'mean': grouped.mean,
            'min': grouped.min, 'max': grouped.max}[agg]()


def _execute(plan: Plan, table: TableHandle) -> Tuple[Any, str]:
    df = table.df
    working, filter_notes = _apply_filters(df, plan.filters, plan.or_filter_groups)

    scope = (f"all {len(df):,} rows" if not filter_notes
             else f"{len(working):,} rows where {' AND '.join(filter_notes)}")

    if len(working) == 0:
        return "No matching rows.", f"No rows matched: {scope}."

    # ── value_counts ──
    if plan.agg == 'value_counts':
        target = plan.target_col or (
            plan.group_cols[0] if plan.group_cols else None)
        if not target or target not in working.columns:
            return None, "Cannot determine which column to count values in."
        counts = working[target].value_counts(dropna=True)
        counts = _apply_limit(counts, plan.limit)
        counts.name = 'Count'
        return counts, f"Most frequent values of '{target}' ({scope})."

    # ── list rows ──
    if plan.agg == 'list':
        cap = plan.limit if plan.limit > 0 else _MAX_LIST_ROWS
        cap = min(len(working), cap)
        return working.head(cap), f"Showing {cap:,} of {len(working):,} rows ({scope})."

    # ── grouped path (count or numeric agg) ──
    if plan.group_cols:
        primary_label = _display_metric_name(plan.agg, plan.measure)
        primary_series = _compute_grouped_agg(
            working, plan.group_cols, plan.agg, plan.measure)

        # Determine sort key
        ascending = plan.agg == 'min'

        # If extras exist OR primary is count and no extras, we build a DataFrame
        # for consistency. Single-metric without extras stays as a Series for
        # backwards compat with the binary "X or Y?" detector.
        if plan.extra_aggs:
            cols_dict = {primary_label: primary_series}
            for extra in plan.extra_aggs:
                try:
                    series = _compute_grouped_agg(
                        working, plan.group_cols,
                        extra['agg'], extra.get('measure'))
                    cols_dict[extra['label']] = series
                except Exception as exc:
                    logger.warning(f"[ExcelTool] extra_agg {extra} failed: {exc}")
                    continue

            result_df = pd.DataFrame(cols_dict)
            result_df = result_df.sort_values(primary_label, ascending=ascending)
            result_df = _apply_limit(result_df, plan.limit)
            for col in result_df.columns:
                result_df[col] = _format_int_or_round(result_df[col])
            result_df = result_df.reset_index()

            metric_names = [primary_label] + [e['label'] for e in plan.extra_aggs]
            metrics_str = (', '.join(metric_names[:-1])
                           + (' and ' + metric_names[-1]
                              if len(metric_names) > 1 else ''))
            shown = len(result_df)
            total_groups = primary_series.shape[0]
            scope_suffix = (f"showing all {shown:,} groups"
                            if shown == total_groups
                            else f"showing top {shown:,} of {total_groups:,} groups")
            desc = (f"{metrics_str} by {' & '.join(plan.group_cols)} "
                    f"({scope}; {scope_suffix}).")
            if not result_df.empty:
                top_row = result_df.iloc[0]
                grp_label = ' / '.join(
                    str(top_row[g]) for g in plan.group_cols
                    if g in result_df.columns)
                top_val = top_row[primary_label]
                try:
                    tv = f"{int(top_val):,}"
                except (ValueError, TypeError):
                    tv = str(top_val)
                superlative = ('highest' if plan.agg in ('sum', 'max', 'count')
                               else 'lowest' if plan.agg == 'min'
                               else 'top')
                desc += (f" {grp_label} has the {superlative} "
                         f"{primary_label.lower()} ({tv}).")
            return result_df, desc

        # Single-metric grouped path
        if plan.agg == 'count':
            result = primary_series.sort_values(ascending=False)
            result = _apply_limit(result, plan.limit)
            result.name = 'Count'
            shown = len(result)
            total_groups = primary_series.shape[0]
            scope_suffix = (f"showing all {shown:,} groups"
                            if shown == total_groups
                            else f"showing top {shown:,} of {total_groups:,} groups")
            return result, (f"Count by {' & '.join(plan.group_cols)} "
                            f"({scope}; {scope_suffix}).")

        # Single-metric grouped numeric agg
        if not plan.measure:
            # Fallback to count
            result = primary_series.sort_values(ascending=False)
            result = _apply_limit(result, plan.limit)
            result.name = 'Count'
            return result, f"Count by {' & '.join(plan.group_cols)} ({scope})."

        result = primary_series.sort_values(ascending=ascending)
        result = _apply_limit(result, plan.limit)
        result = _format_int_or_round(result)
        result.name = plan.measure

        shown = len(result)
        total_groups = primary_series.shape[0]
        scope_suffix = (f"showing all {shown:,} groups"
                        if shown == total_groups
                        else f"showing top {shown:,} of {total_groups:,} groups")
        desc = (f"{primary_label} by {' & '.join(plan.group_cols)} "
                f"({scope}; {scope_suffix}).")
        if not result.empty:
            best_idx = result.index[0]
            best_val = result.iloc[0]
            if isinstance(best_idx, tuple):
                best_idx = ' / '.join(str(x) for x in best_idx)
            try:
                bv = f"{int(best_val):,}"
            except (ValueError, TypeError):
                bv = str(best_val)
            superlative = ('highest' if plan.agg in ('sum', 'max')
                           else 'lowest' if plan.agg == 'min' else 'top')
            desc += (f" {best_idx} has the {superlative} "
                     f"{primary_label.lower()} ({bv}).")
        return result, desc

    # ── global path (no grouping) ──
    if plan.agg == 'count':
        if plan.extra_aggs:
            # Global count + global numeric aggs → single-row DataFrame
            cols_dict = {'Count': len(working)}
            for extra in plan.extra_aggs:
                try:
                    if extra['agg'] == 'count':
                        cols_dict[extra['label']] = len(working)
                    else:
                        m_series = pd.to_numeric(
                            working[extra['measure']], errors='coerce')
                        val = {'sum': m_series.sum, 'mean': m_series.mean,
                               'min': m_series.min,
                               'max': m_series.max}[extra['agg']]()
                        if pd.notna(val) and float(val) == int(val):
                            val = int(val)
                        elif pd.notna(val):
                            val = round(float(val), 2)
                        cols_dict[extra['label']] = val
                except Exception:
                    continue
            return pd.DataFrame([cols_dict]), f"Global summary ({scope})."
        return len(working), f"Row count ({scope})."

    measure = plan.measure
    if not measure:
        return len(working), f"Row count ({scope})."

    if measure not in working.columns:
        return None, f"Column '{measure}' not found."

    measure_series = pd.to_numeric(working[measure], errors='coerce')
    if measure_series.notna().sum() == 0:
        return None, f"Column '{measure}' has no numeric values after filtering."

    primary_label = _display_metric_name(plan.agg, measure)

    if plan.extra_aggs:
        # Global multi-metric → single-row DataFrame
        cols_dict = {}
        primary_val = {'sum': measure_series.sum, 'mean': measure_series.mean,
                       'min': measure_series.min,
                       'max': measure_series.max}[plan.agg]()
        try:
            if pd.notna(primary_val) and float(primary_val) == int(primary_val):
                primary_val = int(primary_val)
            elif pd.notna(primary_val):
                primary_val = round(float(primary_val), 2)
        except (TypeError, ValueError):
            pass
        cols_dict[primary_label] = primary_val

        for extra in plan.extra_aggs:
            try:
                if extra['agg'] == 'count':
                    cols_dict[extra['label']] = len(working)
                else:
                    m_series = pd.to_numeric(
                        working[extra['measure']], errors='coerce')
                    val = {'sum': m_series.sum, 'mean': m_series.mean,
                           'min': m_series.min,
                           'max': m_series.max}[extra['agg']]()
                    if pd.notna(val) and float(val) == int(val):
                        val = int(val)
                    elif pd.notna(val):
                        val = round(float(val), 2)
                    cols_dict[extra['label']] = val
            except Exception:
                continue
        return pd.DataFrame([cols_dict]), f"Global summary ({scope})."

    # Single global scalar
    val = {'sum': measure_series.sum, 'mean': measure_series.mean,
           'min': measure_series.min, 'max': measure_series.max}[plan.agg]()
    try:
        if val == int(val):
            val = int(val)
        else:
            val = round(float(val), 2)
    except (ValueError, TypeError):
        pass
    return val, f"{primary_label} ({scope}): {val}."


# ─── result formatting ───────────────────────────────────────────────────────

def _format_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No matching records."
        return (result.to_markdown(index=False)
                if hasattr(result, 'to_markdown') else result.to_string(index=False))
    if isinstance(result, pd.Series):
        if result.empty:
            return "No matching records."
        if hasattr(result, 'to_markdown'):
            return result.to_markdown()
        return '\n'.join(f"{idx}: {val}" for idx, val in result.items())
    return str(result)


def _is_structured(result) -> bool:
    if isinstance(result, pd.DataFrame):
        return True
    if isinstance(result, pd.Series):
        return len(result) > 1
    return False


_BINARY_QUESTION_RE = re.compile(
    r'\b(\w+)\s+or\s+(\w+)\s*\??\s*$',
    re.IGNORECASE,
)


def _try_answer_binary(question: str, result, description: str) -> Optional[str]:
    m = _BINARY_QUESTION_RE.search(question.strip())
    if not m:
        return None
    opt_a, opt_b = m.group(1).lower(), m.group(2).lower()
    if opt_a in {'is', 'are', 'was', 'were'} or opt_b in {'is', 'are'}:
        return None
    # Only fire for single-metric Series (not multi-metric DataFrames)
    if not isinstance(result, pd.Series) or result.empty:
        return None

    val_a = None
    val_b = None
    label_a = None
    label_b = None
    for idx, val in result.items():
        idx_l = str(idx).lower()
        if val_a is None and (opt_a in idx_l or idx_l in opt_a):
            val_a = val
            label_a = str(idx)
        if val_b is None and (opt_b in idx_l or idx_l in opt_b):
            val_b = val
            label_b = str(idx)

    if val_a is None or val_b is None:
        return None
    try:
        a_num = float(val_a)
        b_num = float(val_b)
    except (ValueError, TypeError):
        return None

    if a_num > b_num:
        winner, w_val, loser, l_val = label_a, int(a_num), label_b, int(b_num)
    elif b_num > a_num:
        winner, w_val, loser, l_val = label_b, int(b_num), label_a, int(a_num)
    else:
        return f"{label_a} and {label_b} are tied at {int(a_num):,} each."
    return (f"**{winner}** ({w_val:,}) is more common than "
            f"{loser} ({l_val:,}).")


_PRESENTER_SYSTEM = (
    "You present a precomputed data result in one or two sentences.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY numbers, names, and labels from the 'Computed result'.\n"
    "2. Never compute, infer, round, or modify any number.\n"
    "3. Never invent labels not in the result.\n"
    "4. Be direct and concise.\n"
    "5. For 'X or Y?' questions: the answer is whichever has the LARGER number. "
    "State both numbers for transparency."
)


def _present_scalar(question: str, raw_result: str, description: str,
                    llm_manager) -> str:
    if llm_manager is None:
        return f"{description}\n\nResult: {raw_result}"
    try:
        msgs = [
            {"role": "system", "content": _PRESENTER_SYSTEM},
            {"role": "user", "content": (
                f"User question: {question}\n\n"
                f"What was computed: {description}\n\n"
                f"Computed result (DO NOT MODIFY):\n{raw_result}\n\n"
                "Present this in one or two sentences:"
            )},
        ]
        return llm_manager.generate(msgs, max_new_tokens=150, temperature=0.0)
    except Exception as exc:
        logger.warning(f"[ExcelTool] presenter failed: {exc}")
        return f"{description}\n\nResult: {raw_result}"


# ─── public entry point ──────────────────────────────────────────────────────

def query_excel(question: str, org_id: str, llm_manager) -> Optional[str]:
    if not is_spreadsheet_query(question):
        return None

    candidates: List[TableHandle] = []
    for f in _find_excel_files(org_id):
        try:
            candidates.extend(_load_excel(f['path'], f['title']))
        except Exception as exc:
            logger.warning(f"[ExcelTool] Excel load failed {f['title']}: {exc}")
    for f in _find_pdf_files(org_id):
        try:
            candidates.extend(_load_pdf_tables(f['path'], f['title']))
        except Exception as exc:
            logger.warning(f"[ExcelTool] PDF load failed {f['title']}: {exc}")

    if not candidates:
        return None

    scored = [(t, _score_table(t, question)) for t in candidates]
    scored.sort(key=lambda x: -x[1])
    best, best_score = scored[0]
    if best_score == 0 and len(candidates) > 1:
        candidates.sort(key=lambda t: -t.schema.row_count)
        best = candidates[0]

    _safe_print(f"[ExcelTool] Picked '{best.title}' [{best.source_type}] "
                f"score={best_score} rows={best.schema.row_count}")
    _safe_print(f"[ExcelTool] dates={best.schema.date_cols} "
                f"measures={best.schema.measure_cols} "
                f"dims={best.schema.dimension_cols} "
                f"text={best.schema.text_cols}")

    plan = _fast_path(question, best)
    plan_source = 'fast-path'

    if plan is None:
        plan, clarify = _llm_plan(question, best, llm_manager)
        if clarify:
            return f"CLARIFY: {clarify}"
        if plan is None:
            return ("CLARIFY: I couldn't confidently interpret that question "
                    "against this data. Could you rephrase, mentioning a "
                    f"specific column? Available columns: "
                    f"{list(best.schema.columns.keys())}")
        plan_source = 'llm-plan'

    _safe_print(f"[ExcelTool] {plan_source}: agg={plan.agg} measure={plan.measure} "
                f"target={plan.target_col} groups={plan.group_cols} "
                f"extras={plan.extra_aggs} filters={plan.filters} limit={plan.limit}")
    if plan.rationale:
        _safe_print(f"[ExcelTool] rationale: {plan.rationale}")

    try:
        result, description = _execute(plan, best)
    except Exception as exc:
        logger.error(f"[ExcelTool] execution failed: {exc}", exc_info=True)
        return None

    if result is None:
        return f"CLARIFY: {description}"

    raw = _format_result(result)
    _safe_print(f"[ExcelTool] {description}")

    binary_answer = _try_answer_binary(question, result, description)
    if binary_answer is not None:
        return binary_answer

    if _is_structured(result):
        return f"**{description}**\n\n{raw}"

    try:
        return _present_scalar(question, raw, description, llm_manager)
    except Exception as exc:
        logger.error(f"[ExcelTool] presenter failed: {exc}")
        return f"**{description}**\n\nResult: {raw}"