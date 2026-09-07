from __future__ import annotations
import os, re
import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

ALLOWED_TABLE = "applications"
BLOCKED = {"insert", "update", "delete", "drop", "alter", "create", "copy", "attach", "pragma", "call"}

EXAMPLES = {
    "overall default rate": "SELECT COUNT(*) AS applications, ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct FROM applications",
    "default rate by gender": "SELECT CODE_GENDER, COUNT(*) AS applications, ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct FROM applications GROUP BY CODE_GENDER ORDER BY applications DESC",
    "average income by risk": "SELECT TARGET, ROUND(AVG(AMT_INCOME_TOTAL), 2) AS average_income FROM applications GROUP BY TARGET ORDER BY TARGET",
    "top income types": "SELECT NAME_INCOME_TYPE, COUNT(*) AS applications, ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct FROM applications GROUP BY NAME_INCOME_TYPE ORDER BY applications DESC LIMIT 10",
    "credit amount by education": "SELECT NAME_EDUCATION_TYPE, ROUND(AVG(AMT_CREDIT), 2) AS average_credit, COUNT(*) AS applications FROM applications GROUP BY NAME_EDUCATION_TYPE ORDER BY average_credit DESC",
}


def validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if len(cleaned) > 1500 or any(re.search(rf"\b{k}\b", cleaned, re.I) for k in BLOCKED):
        raise ValueError("Only safe, read-only analytical queries are allowed.")
    trees = sqlglot.parse(cleaned, read="duckdb")
    if len(trees) != 1 or not isinstance(trees[0], exp.Select):
        raise ValueError("The generated query must be one SELECT statement.")
    tables = {t.name.lower() for t in trees[0].find_all(exp.Table)}
    if tables - {ALLOWED_TABLE}:
        raise ValueError("The query may only access the applications table.")
    if not trees[0].args.get("limit"):
        trees[0].set("limit", exp.Limit(expression=exp.Literal.number(100)))
    return trees[0].sql(dialect="duckdb")


def fallback_sql(question: str) -> str:
    q = question.lower()
    best = max(EXAMPLES, key=lambda k: len(set(k.split()) & set(q.split())))
    overlap = set(best.split()) & set(q.split())
    if not overlap:
        raise ValueError("Try one of the five example questions shown in the app, or configure GEMINI_API_KEY.")
    return EXAMPLES[best]


def llm_sql(question: str, columns: list[str], history: list[dict]) -> str:
    if not os.getenv("GEMINI_API_KEY"):
        return fallback_sql(question)
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""You convert banking questions into DuckDB SQL.
Return SQL only. Use table applications. Only SELECT is allowed. Always aggregate or LIMIT 100.
Columns: {', '.join(columns)}
TARGET: 1 means payment difficulty/default, 0 means no default.
Recent context: {history[-3:]}
Question: {question}"""
    response = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt)
    return response.text.replace("```sql", "").replace("```", "").strip()


def ask(df: pd.DataFrame, question: str, history=None):
    history = history or []
    proposed = llm_sql(question, df.columns.tolist(), history)
    sql = validate_sql(proposed)
    con = duckdb.connect(":memory:")
    con.register(ALLOWED_TABLE, df)
    result = con.execute(sql).df()
    con.close()
    return sql, result

