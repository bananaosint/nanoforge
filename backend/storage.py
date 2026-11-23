"""SQLite is the source of truth for feedback, run history, and timeline snapshots."""
import json
import sqlite3
import threading
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, config TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS metrics(id INTEGER PRIMARY KEY, run_id TEXT, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS timeline(id INTEGER PRIMARY KEY, run_id TEXT, step INTEGER, prompt TEXT, output TEXT);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        self.db.commit()

    def execute(self, query, args=()):
        with self.lock:
            cursor = self.db.execute(query, args)
            self.db.commit()
            return cursor.lastrowid

    def rows(self, query, args=()):
        with self.lock:
            return [dict(row) for row in self.db.execute(query, args).fetchall()]

    def feedback(self, kind=None):
        rows = self.rows('SELECT * FROM feedback' + (' WHERE kind=?' if kind else '') + ' ORDER BY id DESC', (kind,) if kind else ())
        return [{**json.loads(row['payload']), 'id': row['id'], 'kind': row['kind'], 'created_at': row['created_at']} for row in rows]

    def save_feedback(self, kind, payload, item_id=None):
        if item_id is None:
            return self.execute('INSERT INTO feedback(kind,payload,created_at) VALUES(?,?,?)', (kind, json.dumps(payload), now()))
        self.execute('UPDATE feedback SET kind=?,payload=? WHERE id=?', (kind, json.dumps(payload), item_id))
        return item_id

    def setting(self, key, default=None):
        rows = self.rows('SELECT value FROM settings WHERE key=?', (key,))
        return json.loads(rows[0]['value']) if rows else default

    def set_setting(self, key, value):
        self.execute('INSERT OR REPLACE INTO settings VALUES(?,?)', (key, json.dumps(value)))

    def close(self):
        self.db.close()
