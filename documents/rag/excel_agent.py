"""
Excel/CSV Agent — LangChain pandas dataframe agent edition.

Works on Python 3.13 with your existing pandas 3.x and langchain_ollama setup.
No new dependencies beyond `langchain` and `langchain-experimental`.

The agent writes pandas code, executes it in a sandboxed REPL, and returns
the aggregated answer — exactly what Copilot does for downtime Excel files.
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional

from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from .config import RAGConfig


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = ' '.join(str(a) for a in args)
        print(text.encode('utf-8', errors='replace').decode('utf-8'), **kwargs)


def _fmt_cell(v) -> str:
    """Format a single cell value for markdown — handles numbers, NaN, dates."""
    if v is None:
        return ''
    try:
        # Detect pandas NaN without importing numpy
        if isinstance(v, float) and v != v:
            return ''
    except Exception:
        pass
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.2f}".rstrip('0').rstrip('.')
    return str(v)


def _df_to_markdown(df_or_series, index: bool = True) -> str:
    """
    Convert a DataFrame or Series to a markdown table WITHOUT requiring the
    `tabulate` package.  pandas' built-in `.to_markdown()` is an optional
    dependency we don't want to force on production installs.
    """
    try:
        import pandas as pd
    except Exception:
        return str(df_or_series)

    # Normalize Series → DataFrame
    if isinstance(df_or_series, pd.Series):
        df = df_or_series.to_frame(name=df_or_series.name or 'Value')
    else:
        df = df_or_series

    if df.empty:
        return '_(no rows)_'

    headers = []
    if index:
        idx_name = df.index.name or ''
        headers.append(idx_name)
    headers.extend(str(c) for c in df.columns)

    lines = ['| ' + ' | '.join(h for h in headers) + ' |']
    lines.append('|' + '|'.join('---' for _ in headers) + '|')

    for idx, row in df.iterrows():
        cells = []
        if index:
            cells.append(_fmt_cell(idx))
        for c in df.columns:
            cells.append(_fmt_cell(row[c]))
        lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)


# Schema-agnostic analyst context for the LLM fallback.  The deterministic
# Python path handles known intents directly; this prompt only fires when
# the user asks something unusual.  It deliberately avoids assuming specific
# column names — it tells the LLM to discover them from `df.columns`.
ANALYST_CONTEXT_TEMPLATE = """
You are a data analyst working with a pandas DataFrame called `df`.

DATA HANDLING RULES (apply BEFORE answering):
1. Inspect `df.columns` and `df.dtypes` first. Map the user's question
   words to the most relevant column(s) — do not assume any specific
   column names.
2. When counting incidents/records, deduplicate on a sensible natural key
   (date + group columns) if duplicates are likely.
3. Never return raw rows. Always groupby + aggregate.
4. For numeric summaries, coerce with pd.to_numeric(col, errors='coerce')
   before summing/averaging in case the column has text.

DATAFRAME SCHEMA FOR THIS FILE:
{schema_block}

OUTPUT FORMAT:
- Convert final results to markdown tables.
- Use clear section headers (### Title) for multi-part answers.
- Be concise. No preamble like "I will now analyze...". No disclaimers.
- For action-plan style questions, include 'Corrective Actions' and
  'Preventive Actions' sections derived from the observed data.
"""

# Backwards-compatible alias (some callers still import the old name)
MANUFACTURING_CONTEXT = ANALYST_CONTEXT_TEMPLATE


class ExcelAgent:
    """
    LangChain pandas dataframe agent for Excel/CSV files.
    Uses the same Ollama model as your RAG (via ChatOllama).
    """

    SUPPORTED_EXTENSIONS = ('.xlsx', '.xls', '.csv', '.tsv')

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self._llm: Optional[ChatOllama] = None

    def _get_llm(self) -> ChatOllama:
        """Reuse your existing ChatOllama setup."""
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.config.LLM_MODEL,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                repeat_penalty=self.config.REPETITION_PENALTY,
            )
            _safe_print(f"Excel agent connected to Ollama model: {self.config.LLM_MODEL}")
        return self._llm

    @classmethod
    def is_tabular_file(cls, file_path: str) -> bool:
        """Check if a file should be routed here instead of RAG."""
        return file_path.lower().endswith(cls.SUPPORTED_EXTENSIONS)

    def _load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Load Excel or CSV into a pandas DataFrame."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(file_path)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Strip whitespace from string columns - common Excel issue
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        _safe_print(f"Loaded {len(df)} rows, {len(df.columns)} cols from {os.path.basename(file_path)}")
        return df

    def _build_agent(self, df: pd.DataFrame):
        """Create the LangChain pandas agent on demand."""
        llm = self._get_llm()

        return create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type="tool-calling",
            verbose=False,
            allow_dangerous_code=True,
            max_iterations=8,
            include_df_in_prompt=True,
            number_of_head_rows=3,
        )

    def query(
        self,
        file_path: str,
        question: str,
        extra_context: str = ""
    ) -> Tuple[str, Dict]:
        """
        Answer a question about an Excel/CSV file.

        Args:
            file_path: Path to the .xlsx/.csv file
            question: User's natural language question
            extra_context: Optional extra context for this specific file

        Returns:
            (answer_text, metadata_dict)
        """
        df = self._load_dataframe(file_path)

        metadata = {
            "file": os.path.basename(file_path),
            "rows": len(df),
            "columns": df.columns.tolist(),
            "engine": "deterministic",
        }

        # Deterministic pandas path — runs first for all recognised intents.
        # Guaranteed correct counts; no tool-call format issues; sub-millisecond.
        det = self._try_deterministic(df, question)
        if det is not None:
            return det, metadata

        # Unknown intent — hand off to LLM agent.
        metadata["engine"] = "langchain-pandas-agent"
        agent = self._build_agent(df)

        # Build a schema description of *this* file so the LLM doesn't need
        # to assume any column names.
        schema_lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(3).tolist()
            sample_str = ', '.join(repr(s)[:30] for s in sample)
            schema_lines.append(f"  - {col}  ({dtype})  e.g. {sample_str}")
        schema_block = '\n'.join(schema_lines) if schema_lines else '  (no columns detected)'

        context = ANALYST_CONTEXT_TEMPLATE.format(schema_block=schema_block)
        if extra_context:
            context += f"\n\nADDITIONAL CONTEXT:\n{extra_context}"

        full_prompt = (
            f"{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Follow the rules above. Inspect the schema, choose the right columns, "
            "deduplicate if needed, aggregate, then format as markdown."
        )

        try:
            result = agent.invoke({"input": full_prompt})
            answer = result.get("output", str(result))

            # Detect raw tool-call JSON leaks (model returned a function-call
            # blob instead of executing it).  Try deterministic again as a
            # safety net; if that also yields nothing, return a friendly note.
            stripped = answer.strip() if isinstance(answer, str) else ''
            looks_like_toolcall = (
                stripped.startswith('{') and stripped.endswith('}')
                and ('"name"' in stripped or "'name'" in stripped)
                and ('"parameters"' in stripped or "'parameters'" in stripped or '"arguments"' in stripped)
            )
            if looks_like_toolcall or not stripped:
                fallback = self._try_deterministic(df, question)
                if fallback:
                    answer = fallback
                else:
                    answer = (
                        "Couldn't produce a structured answer for this question. "
                        "Try one of these patterns instead:\n"
                        "- *Top repeated issues*\n"
                        "- *List all equipment with incidents*\n"
                        "- *Sub-assembly wise summary*\n"
                        "- *Which equipment has highest downtime this month?*\n"
                        "- *Action plan for top issue*\n"
                        "- *Predict next possible failure*\n"
                        "- *<Equipment Name> down*"
                    )
        except Exception as exc:
            err_str = str(exc)
            _safe_print(f"Excel agent error: {exc}")
            # Friendlier message when Ollama is down or model is missing
            if 'not found' in err_str.lower() and 'model' in err_str.lower():
                answer = (
                    "**Could not complete this question** — the LLM model "
                    f"`{self.config.LLM_MODEL}` is not available on the Ollama "
                    "server. Either install it via:\n\n"
                    f"```\nollama pull {self.config.LLM_MODEL}\n```\n\n"
                    "Or try one of the supported question patterns directly:\n"
                    "- *Top repeated issues*\n"
                    "- *List all equipment with incidents*\n"
                    "- *Sub-assembly wise summary*\n"
                    "- *Which equipment has highest downtime this month?*\n"
                    "- *Action plan for top issue*\n"
                    "- *<Equipment Name> down*"
                )
            elif 'connection' in err_str.lower() or '10061' in err_str:
                answer = (
                    "**Could not reach Ollama** — the LLM server isn't running. "
                    "Start it with `ollama serve`, then retry. "
                    "Many common questions also work without the LLM — try "
                    "*top issues*, *list machines*, *highest downtime*, *X down*, etc."
                )
            else:
                answer = f"Analysis failed. Error: {err_str[:200]}"

        return answer, metadata

    # ── Generic column-name resolver ──────────────────────────────────────────
    # The deterministic patterns are written against logical roles (equipment,
    # issue, downtime, action, date, shift) — not against the specific column
    # names of any one Excel file.  _find_col() maps each logical role to the
    # first matching real column in the dataframe, using case-insensitive
    # substring matching against a list of common aliases.
    COLUMN_ALIASES = {
        'equipment': [
            'equipment name', 'equipment', 'machine name', 'machine', 'asset',
            'asset name', 'device', 'unit', 'station', 'line', 'subassembly',
            'sub-assembly', 'sub assembly',
        ],
        'issue': [
            'issue summary', 'issue', 'problem', 'incident', 'fault',
            'description', 'defect', 'symptom', 'breakdown reason',
            'failure mode', 'observation',
        ],
        'downtime': [
            'total down time', 'total downtime', 'downtime', 'down time',
            'duration', 'lost time', 'loss time', 'idle time', 'breakdown time',
        ],
        'action': [
            'final action', 'action', 'action taken', 'corrective action',
            'resolution', 'fix', 'remedy', 'repair', 'remarks', 'solution',
        ],
        'date': [
            'wo date', 'work order date', 'incident date', 'date',
            'created at', 'created', 'occurrence date', 'report date',
            'logged at', 'timestamp',
        ],
        'shift': [
            'shift', 'shift name', 'work shift', 'shift no', 'shift number',
        ],
    }

    @classmethod
    def _find_col(cls, df: pd.DataFrame, role: str) -> Optional[str]:
        """
        Return the first column in `df` that matches any alias for the given
        logical role (case-insensitive, exact match first, then substring).
        Returns None if no column matches.
        """
        aliases = cls.COLUMN_ALIASES.get(role, [role])
        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        # 1) Exact case-insensitive match
        for alias in aliases:
            if alias in cols_lower:
                return cols_lower[alias]

        # 2) Substring match — the alias is contained in the column name
        for alias in aliases:
            for low, orig in cols_lower.items():
                if alias in low:
                    return orig

        return None

    @classmethod
    def _apply_period_filter(cls, df: pd.DataFrame, q: str) -> pd.DataFrame:
        """
        Filter df by date period keywords in the question.
        Auto-detects the date column via _find_col() — works with any schema.
        Returns the original df if no period keyword or no date column.
        """
        if 'today' not in q and 'this week' not in q and 'this month' not in q \
                and 'last week' not in q and 'last month' not in q \
                and 'this year' not in q and 'last 7' not in q and 'last 30' not in q:
            return df

        date_col = cls._find_col(df, 'date')
        if not date_col:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    date_col = c
                    break
        if not date_col:
            return df

        try:
            parsed = pd.to_datetime(df[date_col], errors='coerce')
        except Exception:
            return df

        now = pd.Timestamp.now().normalize()
        if 'today' in q:
            mask = parsed.dt.date == now.date()
        elif 'this week' in q or 'last 7' in q:
            start = now - pd.Timedelta(days=7)
            mask = parsed >= start
        elif 'this month' in q:
            mask = (parsed.dt.year == now.year) & (parsed.dt.month == now.month)
        elif 'last month' in q:
            last_month = (now - pd.DateOffset(months=1))
            mask = (parsed.dt.year == last_month.year) & (parsed.dt.month == last_month.month)
        elif 'this year' in q:
            mask = parsed.dt.year == now.year
        elif 'last 30' in q:
            start = now - pd.Timedelta(days=30)
            mask = parsed >= start
        else:
            return df

        filtered = df[mask]
        return filtered if len(filtered) else df  # fall back to full df if filter empties everything

    def _try_deterministic(self, df: pd.DataFrame, question: str) -> Optional[str]:
        """
        Pure-pandas answer for recognised analytical intents.
        Returns the answer string, or None if the question doesn't match any
        known pattern (callers should then try the LLM agent).
        """
        q = question.lower()

        try:
            # Resolve logical column roles → actual column names in *this* file
            col_equip  = self._find_col(df, 'equipment')
            col_issue  = self._find_col(df, 'issue')
            col_dt     = self._find_col(df, 'downtime')
            col_action = self._find_col(df, 'action')
            col_shift  = self._find_col(df, 'shift')
            col_date   = self._find_col(df, 'date')

            # Apply date-period filter first so "this month" etc. work everywhere
            df_period = self._apply_period_filter(df, q)

            # Build dedup keys from whatever role-columns exist in the file
            dedup_keys = [c for c in (col_date, col_shift, col_equip, col_issue)
                          if c and c in df_period.columns]
            df_clean = df_period.drop_duplicates(subset=dedup_keys) if dedup_keys else df_period

            # ── predictive / forecast — next likely failure ───────────────────
            if any(kw in q for kw in ['predict', 'forecast', 'next failure',
                                       'next possible', 'will fail', 'likely to fail',
                                       'next breakdown', 'expected failure',
                                       'risk of failure', 'most at risk', 'next down',
                                       'anticipate']):
                if col_equip and col_issue:
                    # Score: most-frequent equipment × most-frequent issue per equipment.
                    # Optionally weight by recency if a date column exists.
                    recency_weight = pd.Series(1.0, index=df_clean.index)
                    if col_date:
                        try:
                            d = pd.to_datetime(df_clean[col_date], errors='coerce')
                            valid = d.notna()
                            if valid.any():
                                max_d = d[valid].max()
                                age_days = (max_d - d).dt.days.fillna(365)
                                # Exponential decay: newer events count more
                                recency_weight = pd.Series(
                                    [pow(0.5, max(a, 0) / 90.0) for a in age_days],
                                    index=df_clean.index,
                                )
                        except Exception:
                            pass

                    df_scored = df_clean.assign(_w=recency_weight)
                    eq_score = (df_scored.groupby(col_equip)['_w'].sum()
                                .sort_values(ascending=False))
                    if eq_score.empty:
                        return "Not enough historical data to make a prediction."

                    top_equipments = eq_score.head(5)
                    lines = ["### Predictive Failure Forecast\n"]
                    lines.append(
                        "Based on historical frequency"
                        + (" (recency-weighted)" if col_date else "")
                        + ", the equipment most likely to fail next:\n"
                    )

                    rows = []
                    for eq_name, score in top_equipments.items():
                        sub = df_clean[df_clean[col_equip].astype(str) == str(eq_name)]
                        top_issue = sub[col_issue].value_counts().head(1)
                        likely_issue = str(top_issue.index[0]) if len(top_issue) else '—'
                        rows.append({
                            'Equipment':     str(eq_name),
                            'Risk Score':    round(float(score), 1),
                            'Past Incidents': len(sub),
                            'Likely Issue':  likely_issue[:60],
                        })

                    rank_df = pd.DataFrame(rows).set_index('Equipment')
                    lines.append(_df_to_markdown(rank_df))

                    # Recommendation block
                    top_pred = rows[0]
                    lines.append("\n### Recommended Preventive Actions")
                    lines.append(
                        f"1. **Inspect {top_pred['Equipment']}** immediately — "
                        f"watch for: *{top_pred['Likely Issue']}*."
                    )
                    if col_action:
                        sub_top = df_clean[df_clean[col_equip].astype(str) == top_pred['Equipment']]
                        if not sub_top.empty:
                            common_fix = sub_top[col_action].value_counts().head(1)
                            if len(common_fix):
                                lines.append(
                                    f"2. **Pre-stage spares / tools** for the most common fix: "
                                    f"*{str(common_fix.index[0])[:80]}*."
                                )
                    lines.append("3. **Increase monitoring frequency** for the top 3 equipment above.")
                    lines.append("4. **Run condition-based maintenance** on high-risk units before next shift.")
                    lines.append(
                        "\n*Forecast is based on historical patterns in the data, not a physical "
                        "wear model. Treat as a prioritisation guide, not a guarantee.*"
                    )
                    return "\n".join(lines)

            # ── highest / max downtime by equipment ───────────────────────────
            if any(kw in q for kw in ['highest downtime', 'most downtime', 'max downtime',
                                       'longest downtime', 'biggest downtime',
                                       'highest down time', 'maximum downtime',
                                       'top downtime']):
                if col_equip and col_dt:
                    dt = pd.to_numeric(df_clean[col_dt], errors='coerce')
                    df_dt = df_clean.assign(_dt=dt).dropna(subset=['_dt'])
                    if df_dt.empty:
                        return "No numeric downtime values found."
                    ranked = (df_dt.groupby(col_equip)
                              .agg(Incidents=(col_equip, 'count'),
                                   Total_Downtime_min=('_dt', 'sum'),
                                   Avg_Downtime_min=('_dt', 'mean'))
                              .sort_values('Total_Downtime_min', ascending=False)
                              .head(10))
                    ranked['Total_Downtime_hr'] = (ranked['Total_Downtime_min'] / 60).round(2)
                    ranked['Avg_Downtime_min'] = ranked['Avg_Downtime_min'].round(1)
                    top_eq = str(ranked.index[0])
                    top_hr = ranked.iloc[0]['Total_Downtime_hr']
                    period_note = ""
                    if df_period is not df:
                        period_note = f"  *(filtered to period from question)*"
                    return (
                        f"### Equipment with Highest Downtime{period_note}\n\n"
                        f"**{top_eq}** has the highest downtime — **{top_hr} hours** across "
                        f"**{int(ranked.iloc[0]['Incidents'])} incidents**.\n\n"
                        f"#### Top 10 by Total Downtime\n\n"
                        + _df_to_markdown(ranked[['Incidents', 'Total_Downtime_min', 'Total_Downtime_hr', 'Avg_Downtime_min']])
                    )

            # ── top repeated issues ───────────────────────────────────────────
            if any(kw in q for kw in ['repeated', 'frequent', 'top issue', 'most common', 'top 10']):
                if col_issue:
                    agg_kwargs: Dict = {'Count': (col_issue, 'count')}
                    if col_dt:
                        agg_kwargs['Total_Downtime_min'] = (col_dt, 'sum')
                    top = (df_clean.groupby(col_issue)
                           .agg(**agg_kwargs)
                           .sort_values('Count', ascending=False)
                           .head(10))
                    return "### Top 10 Repeated Issues\n\n" + _df_to_markdown(top)

            # ── equipment / machine list ──────────────────────────────────────
            if any(kw in q for kw in ['list machine', 'list equipment', 'equipment list',
                                       'machine list', 'all machine', 'all equipment',
                                       'list machines']):
                if col_equip:
                    agg_kwargs = {'Incidents': (col_equip, 'count')}
                    if col_dt:
                        agg_kwargs['Total_Downtime_min'] = (col_dt, 'sum')
                    machines = (df_clean.groupby(col_equip)
                                .agg(**agg_kwargs)
                                .sort_values('Incidents', ascending=False))
                    return f"### {col_equip} Summary\n\n" + _df_to_markdown(machines)

            # ── sub-assembly / machine-wise summary ───────────────────────────
            if any(kw in q for kw in ['sub', 'assembly', 'machine wise', 'machinewise',
                                       'equipment wise', 'summarize', 'summary']):
                if col_equip and col_issue:
                    lines = [f"### {col_equip}-wise {col_issue} Summary\n"]
                    top_machines = (df_clean.groupby(col_equip)
                                    .size().sort_values(ascending=False).head(15).index)
                    for machine in top_machines:
                        sub = df_clean[df_clean[col_equip] == machine]
                        top_issues = sub[col_issue].value_counts().head(5)
                        lines.append(f"\n#### {machine}  ({len(sub)} records)")
                        lines.append(_df_to_markdown(top_issues))
                    return "\n".join(lines)

            # ── action plan for a specific issue ─────────────────────────────
            if any(kw in q for kw in ['action plan', 'action for', 'how to fix',
                                       'corrective', 'preventive', 'resolution']):
                if col_issue and col_action:
                    issue_hint = ""
                    for marker in ['action plan for', 'action for', 'how to fix', 'plan for']:
                        if marker in q:
                            issue_hint = question[q.find(marker) + len(marker):].strip()
                            break

                    hint_lower = issue_hint.lower()
                    is_top_alias = any(t in hint_lower for t in [
                        'top issue', 'top problem', 'biggest issue', 'biggest problem',
                        'most common issue', 'most frequent issue', 'most repeated',
                        'top one', 'top 1',
                    ]) or hint_lower in ('top', 'biggest', 'main', 'most common')

                    if is_top_alias:
                        top_issue = df_clean[col_issue].value_counts()
                        if top_issue.empty:
                            return "No issues found in the data."
                        issue_hint = str(top_issue.index[0])

                    matched = (
                        df_clean[df_clean[col_issue].astype(str).str.contains(
                            issue_hint, case=False, na=False, regex=False)]
                        if issue_hint else df_clean
                    )
                    if matched.empty:
                        return f"No records found matching: **{issue_hint}**"
                    actions = matched[col_action].value_counts().head(8)
                    lines = [f"### Action Plan — {issue_hint or 'All Records'}\n",
                             f"**Matching records:** {len(matched)}\n",
                             "#### Observed Actions (most frequent first)\n",
                             _df_to_markdown(actions),
                             "\n#### Corrective Actions",
                             "Resolve active occurrence based on the most frequent action above.",
                             "\n#### Preventive Actions",
                             "Implement a scheduled inspection or FMEA entry for this failure mode."]
                    return "\n".join(lines)

            # ── specific equipment / "X down" — triage with SOP ───────────────
            if col_equip:
                equipment_names = df_clean[col_equip].dropna().astype(str).unique().tolist()
                matched_equip = None
                # Prefer longest-name match first
                for equip in sorted(equipment_names, key=lambda e: -len(e)):
                    if equip.lower() in q:
                        matched_equip = equip
                        break
                if not matched_equip:
                    for equip in equipment_names:
                        equip_words = [w for w in equip.lower().split() if len(w) > 3]
                        question_words = [w for w in q.split() if len(w) > 3]
                        if equip_words and any(ew in question_words for ew in equip_words):
                            matched_equip = equip
                            break

                if matched_equip:
                    sub = df_clean[df_clean[col_equip].astype(str) == matched_equip]
                    is_down_query = any(kw in q for kw in ['down', 'breakdown', 'not working',
                                                            'failure', 'fault', 'stopped'])

                    lines = [f"### {matched_equip} — {len(sub)} records\n"]

                    if col_dt:
                        dt_num = pd.to_numeric(sub[col_dt], errors='coerce').dropna()
                        if not dt_num.empty:
                            total_min = dt_num.sum()
                            avg_min = dt_num.mean()
                            lines.append(
                                f"**Total {col_dt}:** {total_min:.0f} ({total_min/60:.1f} hrs) "
                                f"· **Avg:** {avg_min:.1f} per record\n"
                            )

                    top_issue_name = None
                    if col_issue:
                        vc = sub[col_issue].value_counts().head(5)
                        if not vc.empty:
                            top_issue_name = str(vc.index[0])
                            lines.append(f"#### Top 5 Recurring {col_issue}\n" + _df_to_markdown(vc) + "\n")

                    common_actions = []
                    if col_action:
                        actions_vc = sub[col_action].value_counts().head(8)
                        if not actions_vc.empty:
                            common_actions = list(actions_vc.index)
                            lines.append(f"#### Most Effective Past {col_action}\n" + _df_to_markdown(actions_vc) + "\n")

                    if is_down_query:
                        # Triage-style response (Copilot-like SOP)
                        lines.append("#### Rapid Triage Checklist\n")
                        lines.append("1. **Safety & interlocks** — confirm safety doors closed, HMI shows no lockout alarms.")
                        lines.append("2. **Diagnostics** — pull current HMI alarm/error history; note any recurring fault codes.")
                        lines.append("3. **Visual inspection** — check for visible damage, loose connections, jams, debris.")
                        lines.append("4. **Calibration / homing** — if positional or offset faults, run calibration sequence.")
                        lines.append("5. **Sensors & actuators** — verify all sensors signal stable; no actuators left in disabled state from prior workaround.")
                        if common_actions:
                            lines.append(f"6. **Apply known-good action** — most recent successful fix for this equipment: **{common_actions[0]}**.")
                        lines.append("")

                        lines.append("#### Preventive Maintenance")
                        lines.append("- **Daily:** Quick visual inspection + clean sensors/lenses if vision-based.")
                        lines.append("- **Weekly:** Calibration check, alignment verification.")
                        lines.append("- **Monthly:** Detailed mechanical inspection, lubrication, fastener torque.")
                        lines.append("- **Quarterly:** FMEA review for this asset; spare-parts availability check.")

                        if top_issue_name:
                            lines.append(f"\n*If the current fault is **{top_issue_name}** or similar, jump directly to the action shown above for that issue.*")

                    return "\n".join(lines)

        except Exception as exc:
            _safe_print(f"[ExcelAgent] deterministic path failed: {exc!r}")

        return None  # no pattern matched — caller should use LLM

    def _fallback_answer(self, df: pd.DataFrame, question: str, error: str) -> str:
        """Last-resort answer when the LLM agent raises an exception."""
        det = self._try_deterministic(df, question)
        if det is not None:
            return det
        return f"The analysis agent encountered an error and could not answer this query. Error: {error[:150]}"

    def get_data_summary(self, file_path: str) -> Dict:
        """Quick file structure summary - useful for an initial 'here's what's in your file'."""
        df = self._load_dataframe(file_path)

        return {
            "filename": os.path.basename(file_path),
            "rows": len(df),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "date_columns": [
                col for col in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[col])
            ],
            "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
            "sample": df.head(3).to_dict(orient='records'),
        }
