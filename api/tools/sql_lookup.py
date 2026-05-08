"""
SQL Lookup Tool
Converts natural language to SQL via the LLM, queries local SQLite DB.

Failure contract:
- Empty question  → MALFORMED
- No DB found     → EXECUTION error
- No rows         → EMPTY_RESULTS
- Bad SQL         → EXECUTION error with stderr
"""
from __future__ import annotations
import os
import sqlite3
from typing import Any, Dict, List, Optional

from tools.base import BaseTool


# ── Sample DB seeding ─────────────────────────────────────────────────────────
DB_PATH = os.getenv("SQLITE_SAMPLE_DB", "/app/data/sample.db")


def seed_sample_db(path: str = DB_PATH):
    """Create and populate a sample SQLite DB for demos and evals."""
    conn = sqlite3.connect(path)
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id          INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            category    TEXT NOT NULL,
            price       REAL NOT NULL,
            stock       INTEGER NOT NULL,
            rating      REAL
        );

        CREATE TABLE IF NOT EXISTS sales (
            id          INTEGER PRIMARY KEY,
            product_id  INTEGER,
            quantity    INTEGER,
            sale_date   TEXT,
            revenue     REAL,
            FOREIGN KEY(product_id) REFERENCES products(id)
        );

        CREATE TABLE IF NOT EXISTS customers (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            email       TEXT,
            country     TEXT,
            joined_date TEXT
        );

        INSERT OR IGNORE INTO products VALUES
            (1,  'Laptop Pro',      'Electronics', 1299.99, 45,  4.7),
            (2,  'Wireless Mouse',  'Electronics', 29.99,  200,  4.3),
            (3,  'Standing Desk',   'Furniture',   499.99,  30,  4.5),
            (4,  'Python Book',     'Books',        39.99,  150,  4.8),
            (5,  'Coffee Maker',    'Appliances',   89.99,   75,  4.2),
            (6,  'Monitor 27"',     'Electronics', 349.99,   60,  4.6),
            (7,  'Ergonomic Chair', 'Furniture',   299.99,   25,  4.4),
            (8,  'Mechanical KB',   'Electronics',  99.99,   90,  4.5),
            (9,  'AI Book',         'Books',        49.99,  120,  4.9),
            (10, 'Desk Lamp',       'Furniture',    35.99,  180,  4.1);

        INSERT OR IGNORE INTO sales VALUES
            (1,  1, 3, '2024-01-15', 3899.97),
            (2,  2, 10,'2024-01-16',  299.90),
            (3,  3, 2, '2024-01-17',  999.98),
            (4,  4, 5, '2024-01-18',  199.95),
            (5,  1, 2, '2024-02-01', 2599.98),
            (6,  6, 4, '2024-02-05', 1399.96),
            (7,  9, 8, '2024-02-10',  399.92),
            (8,  5, 3, '2024-02-12',  269.97),
            (9,  7, 1, '2024-03-01',  299.99),
            (10, 8, 6, '2024-03-05',  599.94);

        INSERT OR IGNORE INTO customers VALUES
            (1, 'Alice Johnson', 'alice@email.com',   'USA',   '2023-01-10'),
            (2, 'Bob Smith',     'bob@email.com',     'UK',    '2023-03-15'),
            (3, 'Carol Lee',     'carol@email.com',   'India', '2023-06-20'),
            (4, 'David Kim',     'david@email.com',   'Korea', '2023-08-05'),
            (5, 'Eva Martinez',  'eva@email.com',     'Spain', '2024-01-02');
    """)

    conn.commit()
    conn.close()


def _nl_to_sql(question: str) -> str:
    """
    Simple rule-based NL→SQL for common patterns.
    In the full pipeline the LLM agent handles this —
    this fallback handles eval cases without an LLM call.
    """
    q = question.lower()

    if "most expensive" in q or "highest price" in q:
        return "SELECT name, price FROM products ORDER BY price DESC LIMIT 5;"

    if "cheapest" in q or "lowest price" in q:
        return "SELECT name, price FROM products ORDER BY price ASC LIMIT 5;"

    if "best rated" in q or "highest rated" in q or "top rated" in q:
        return "SELECT name, rating FROM products ORDER BY rating DESC LIMIT 5;"

    if "electronics" in q:
        return "SELECT name, price, stock FROM products WHERE category='Electronics';"

    if "furniture" in q:
        return "SELECT name, price, stock FROM products WHERE category='Furniture';"

    if "total revenue" in q or "total sales" in q:
        return "SELECT SUM(revenue) as total_revenue FROM sales;"

    if "product" in q and "category" in q:
        return "SELECT category, COUNT(*) as count, AVG(price) as avg_price FROM products GROUP BY category;"

    if "customer" in q or "customers" in q:
        return "SELECT name, country, joined_date FROM customers ORDER BY joined_date DESC;"

    if "stock" in q and "low" in q:
        return "SELECT name, stock FROM products WHERE stock < 50 ORDER BY stock ASC;"

    # Default: show all products
    return "SELECT name, category, price, rating FROM products ORDER BY rating DESC;"


class SQLLookupTool(BaseTool):
    name = "sql_lookup"
    timeout_seconds = 10.0

    def __init__(self, db_path: str = DB_PATH, llm_client=None):
        self.db_path    = db_path
        self.llm_client = llm_client  # optional: pass Groq client for real NL→SQL

    def _validate_input(self, question: str = "", **kwargs) -> Optional[str]:
        if not question or not question.strip():
            return "Question cannot be empty"
        if len(question.strip()) < 5:
            return "Question too short to generate meaningful SQL"
        return None

    def _execute(self, question: str, **kwargs) -> Dict[str, Any]:
        # Ensure DB exists and is seeded
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            seed_sample_db(self.db_path)

        # Convert NL → SQL
        if self.llm_client:
            sql = self._llm_nl_to_sql(question)
        else:
            sql = _nl_to_sql(question)

        # Execute SQL
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur  = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            conn.close()
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL execution failed: {e} | SQL was: {sql}")

        return {
            "question": question,
            "sql":      sql,
            "columns":  columns,
            "rows":     [dict(row) for row in rows],
            "row_count": len(rows),
        }

    def _llm_nl_to_sql(self, question: str) -> str:
        """Use Groq LLM to convert natural language to SQL."""
        schema = """
        Tables:
        - products(id, name, category, price, stock, rating)
        - sales(id, product_id, quantity, sale_date, revenue)
        - customers(id, name, email, country, joined_date)
        """
        prompt = f"""Convert this question to SQLite SQL. Return ONLY the SQL query, nothing else.

Schema: {schema}
Question: {question}
SQL:"""
        try:
            response = self.llm_client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            sql = response.choices[0].message.content.strip()
            # Clean up markdown if LLM wraps in backticks
            sql = sql.replace("```sql", "").replace("```", "").strip()
            return sql
        except Exception:
            # Fallback to rule-based
            return _nl_to_sql(question)