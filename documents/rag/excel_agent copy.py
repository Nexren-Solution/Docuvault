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


# Manufacturing-specific system context — embedded in every prompt.
# This is what gives you Copilot-level answers instead of raw row dumps.
MANUFACTURING_CONTEXT = """
You are a manufacturing downtime data analyst working with a pandas DataFrame called `df`.

DATA HANDLING RULES (apply BEFORE answering):
1. ALWAYS deduplicate before counting. The natural key is typically:
   ['WO Date', 'Shift', 'Equipment Name', 'Issue Summary']
   Use: df_clean = df.drop_duplicates(subset=[...natural_key...])
2. Never return raw rows. Always groupby + aggregate.
3. For downtime totals, use 'Total Down Time' from one row per dedup group.

INTENT -> PANDAS PATTERN:
- "repeated issues" / "frequent down" / "top issues"
  -> df_clean.groupby('Issue Summary').agg(Count=('Issue Summary','count'),
      Total_Downtime=('Total Down Time','sum')).sort_values('Count', ascending=False).head(10)
- "sub assembly wise" / "by equipment"
  -> For each Equipment Name, take top 5 Issue Summaries by count.
- "action plan for X" / "how to fix X"
  -> matched = df_clean[df_clean['Issue Summary'].str.contains('X', case=False, na=False)]
  -> actions = matched['Final Action'].value_counts().head(5)
  -> Then synthesize: Corrective Actions (from observed) + Preventive Actions.
- "list machines" / "equipment list"
  -> df_clean.groupby('Equipment Name').agg(Incidents=('Equipment Name','count'),
      Downtime=('Total Down Time','sum')).sort_values('Incidents', ascending=False)
- "X down" / "X breakdown" (X is equipment name)
  -> matched = df_clean[df_clean['Equipment Name'].str.contains('X', case=False, na=False)]
  -> Show: total incidents, total downtime, top 5 Issue Summaries, most common Final Action.

OUTPUT FORMAT:
- Convert final results to markdown tables using df.to_markdown().
- Two sections for action plans: '### Corrective Actions' and '### Preventive Actions'.
- Be concise. No preamble like "I will now analyze...". No disclaimers.
"""


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

        context = MANUFACTURING_CONTEXT
        if extra_context:
            context += f"\n\nADDITIONAL CONTEXT:\n{extra_context}"

        full_prompt = (
            f"{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Follow the rules above. Deduplicate first, aggregate, then format as markdown."
        )

        try:
            result = agent.invoke({"input": full_prompt})
            answer = result.get("output", str(result))
        except Exception as exc:
            _safe_print(f"Excel agent error: {exc}")
            answer = f"Analysis failed. Error: {str(exc)[:200]}"

        return answer, metadata

    def _try_deterministic(self, df: pd.DataFrame, question: str) -> Optional[str]:
        """
        Pure-pandas answer for recognised analytical intents.
        Returns the answer string, or None if the question doesn't match any
        known pattern (callers should then try the LLM agent).
        """
        q = question.lower()

        try:
            dedup_keys = [c for c in ['WO Date', 'Shift', 'Equipment Name', 'Issue Summary']
                          if c in df.columns]
            df_clean = df.drop_duplicates(subset=dedup_keys) if dedup_keys else df

            # ── top repeated issues ───────────────────────────────────────────
            if any(kw in q for kw in ['repeated', 'frequent', 'top issue', 'most common', 'top 10']):
                if 'Issue Summary' in df_clean.columns:
                    agg_kwargs: Dict = {'Count': ('Issue Summary', 'count')}
                    if 'Total Down Time' in df_clean.columns:
                        agg_kwargs['Total_Downtime_min'] = ('Total Down Time', 'sum')
                    top = (df_clean.groupby('Issue Summary')
                           .agg(**agg_kwargs)
                           .sort_values('Count', ascending=False)
                           .head(10))
                    return "### Top 10 Repeated Issues\n\n" + top.to_markdown()

            # ── equipment / machine list ──────────────────────────────────────
            if any(kw in q for kw in ['list machine', 'list equipment', 'equipment list',
                                       'machine list', 'all machine', 'all equipment',
                                       'list machines']):
                if 'Equipment Name' in df_clean.columns:
                    agg_kwargs = {'Incidents': ('Equipment Name', 'count')}
                    if 'Total Down Time' in df_clean.columns:
                        agg_kwargs['Total_Downtime_min'] = ('Total Down Time', 'sum')
                    machines = (df_clean.groupby('Equipment Name')
                                .agg(**agg_kwargs)
                                .sort_values('Incidents', ascending=False))
                    return "### Equipment Summary\n\n" + machines.to_markdown()

            # ── sub-assembly / machine-wise summary ───────────────────────────
            if any(kw in q for kw in ['sub', 'assembly', 'machine wise', 'machinewise',
                                       'equipment wise', 'summarize', 'summary']):
                if 'Equipment Name' in df_clean.columns and 'Issue Summary' in df_clean.columns:
                    lines = ["### Sub-Assembly / Equipment-Wise Issue Summary\n"]
                    top_machines = (df_clean.groupby('Equipment Name')
                                    .size().sort_values(ascending=False).head(15).index)
                    for machine in top_machines:
                        sub = df_clean[df_clean['Equipment Name'] == machine]
                        top_issues = sub['Issue Summary'].value_counts().head(5)
                        lines.append(f"\n#### {machine}  ({len(sub)} incidents)")
                        lines.append(top_issues.to_markdown())
                    return "\n".join(lines)

            # ── action plan for a specific issue ─────────────────────────────
            if any(kw in q for kw in ['action plan', 'action for', 'how to fix',
                                       'corrective', 'preventive', 'resolution']):
                if 'Issue Summary' in df_clean.columns and 'Final Action' in df_clean.columns:
                    issue_hint = ""
                    for marker in ['action plan for', 'action for', 'how to fix', 'plan for']:
                        if marker in q:
                            issue_hint = question[q.find(marker) + len(marker):].strip()
                            break
                    matched = (
                        df_clean[df_clean['Issue Summary'].str.contains(
                            issue_hint, case=False, na=False, regex=False)]
                        if issue_hint else df_clean
                    )
                    if matched.empty:
                        return f"No records found matching: **{issue_hint}**"
                    actions = matched['Final Action'].value_counts().head(8)
                    lines = [f"### Action Plan — {issue_hint or 'All Issues'}\n",
                             f"**Matching incidents:** {len(matched)}\n",
                             "#### Observed Actions (most frequent first)\n",
                             actions.to_markdown(),
                             "\n#### Corrective Actions",
                             "Resolve active occurrence based on the most frequent action above.",
                             "\n#### Preventive Actions",
                             "Implement a scheduled inspection or FMEA entry for this failure mode."]
                    return "\n".join(lines)

            # ── specific equipment / "X down" ─────────────────────────────────
            if 'Equipment Name' in df_clean.columns:
                equipment_names = df_clean['Equipment Name'].dropna().unique().tolist()
                matched_equip = None
                for equip in equipment_names:
                    if equip.lower() in q or any(
                        word in equip.lower() for word in q.split() if len(word) > 3
                    ):
                        matched_equip = equip
                        break
                if matched_equip:
                    sub = df_clean[df_clean['Equipment Name'] == matched_equip]
                    lines = [f"### {matched_equip}  —  {len(sub)} incidents\n"]
                    if 'Total Down Time' in sub.columns:
                        lines.append(f"**Total Downtime:** {sub['Total Down Time'].sum():.1f} min\n")
                    if 'Issue Summary' in sub.columns:
                        lines.append("**Top Issues:**\n" + sub['Issue Summary'].value_counts().head(5).to_markdown())
                    if 'Final Action' in sub.columns:
                        lines.append(f"\n**Most Common Action:** {sub['Final Action'].value_counts().idxmax()}")
                    return "\n".join(lines)

        except Exception:
            pass

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
