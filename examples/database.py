import sqlite3

def initialize_database():
    """Initialize the SQLite database and create the results table if it doesn't exist."""

    conn = sqlite3.connect("anneal.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY,
        temp REAL,
        top_k INTEGER,
        top_p REAL,
        prompt TEXT,
        response TEXT
    );
    """)

    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS results_fts USING fts5(response, content='results', content_rowid='id', tokenize='porter');
    """)

    conn.commit()

    return conn, cursor

def insert_result(conn, cursor, temp, top_k, top_p, prompt, response):
    # Insert a result row into the SQLite database.
    cursor.execute("""
    INSERT INTO results (temp, top_k, top_p, prompt, response)
    VALUES (?, ?, ?, ?, ?)
    """, (temp, top_k, top_p, prompt, response))

    # Insert the same response into the results_fts virtual table
    cursor.execute("""
    INSERT INTO results_fts (rowid, response) VALUES (last_insert_rowid(), ?)
    """, (response,))

    conn.commit()


def search_response(cursor, query):
    """Search for responses in the database containing the given query."""
    cursor.execute("SELECT rowid, response FROM results_fts WHERE response MATCH ?", (query,))
    return cursor.fetchall()


if __name__ == "__main__":
    # Initialize or connect to the database
    conn, cursor = initialize_database()

    # Close the connection to the database
    conn.close()

