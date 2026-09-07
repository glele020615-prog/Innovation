"""
账号体系：本地 SQLite 存储 + pbkdf2 密码哈希。
首次运行自动建表，并写入种子账号（医生 / 患者各一）。
"""
import os
import sys
import sqlite3
import hashlib
import secrets
from datetime import datetime


def _resolve_data_dir() -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(appdata, "ZhyingShiWei", "Data")
    else:
        # 可用环境变量 ZYSW_DATA_DIR 临时指向别的账号库目录（如在部署机上添加账号）
        d = os.environ.get("ZYSW_DATA_DIR") or r"F:\Innovation\Data"
    os.makedirs(d, exist_ok=True)
    return d


DB_PATH = os.path.join(_resolve_data_dir(), "accounts.db")

_ITERATIONS = 120000
_SALT_BYTES = 16
_ALGO = "sha256"


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(_ALGO, password.encode("utf-8"), salt, _ITERATIONS)


def make_password_hash(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _hash_password(password, salt)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False
    got = _hash_password(password, salt)
    return secrets.compare_digest(got, expected)


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT UNIQUE NOT NULL,
                password     TEXT NOT NULL,
                role         TEXT NOT NULL CHECK(role IN ('doctor','patient')),
                display_name TEXT,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT NOT NULL,
                title        TEXT,
                path         TEXT NOT NULL,
                uploaded_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_logs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT NOT NULL,
                game         TEXT NOT NULL,
                difficulty   TEXT,
                total        INTEGER,
                correct      INTEGER,
                accuracy     REAL,
                avg_rt_ms    INTEGER,
                duration_s   INTEGER,
                played_at    TEXT NOT NULL
            );
            """
        )
        cur = conn.execute("SELECT COUNT(*) FROM accounts")
        if cur.fetchone()[0] == 0:
            now = datetime.now().isoformat(timespec="seconds")
            seeds = [
                ("doctor",  make_password_hash("Doctor@2026"),  "doctor",  "示例医生"),
                ("patient", make_password_hash("Patient@2026"), "patient", "示例患者"),
            ]
            conn.executemany(
                "INSERT INTO accounts(username,password,role,display_name,created_at) VALUES(?,?,?,?,?)",
                [(u, p, r, n, now) for u, p, r, n in seeds],
            )


def add_user(username: str, password: str, role: str, display_name: str = "") -> tuple:
    """管理员添加账号（无注册界面，账号由管理员离线创建）。返回 (ok, 提示信息)。"""
    if not username or not username.strip():
        return False, "用户名不能为空"
    if role not in ("doctor", "patient"):
        return False, "角色必须是 doctor 或 patient"
    if not password or len(password) < 6:
        return False, "密码至少 6 位"
    with _connect() as conn:
        dup = conn.execute(
            "SELECT 1 FROM accounts WHERE username=?", (username,)
        ).fetchone()
        if dup:
            return False, f"用户名已存在：{username}"
        conn.execute(
            "INSERT INTO accounts(username,password,role,display_name,created_at) VALUES(?,?,?,?,?)",
            (
                username,
                make_password_hash(password),
                role,
                display_name or username,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    return True, f"账号已创建：{username}（{role}）"


def verify(username: str, password: str):
    with _connect() as conn:
        row = conn.execute(
            "SELECT password, role, display_name FROM accounts WHERE username=?",
            (username,),
        ).fetchone()
    if not row:
        return False, None, None, "账号不存在"
    if not verify_password(password, row["password"]):
        return False, None, None, "密码错误"
    return True, row["role"], row["display_name"] or username, None


def get_account(username: str):
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, role, display_name, created_at FROM accounts WHERE username=?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def change_password(username: str, old_pwd: str, new_pwd: str) -> tuple:
    if not new_pwd or len(new_pwd) < 6:
        return False, "新密码至少 6 位"
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM accounts WHERE username=?", (username,)
        ).fetchone()
        if not row or not verify_password(old_pwd, row["password"]):
            return False, "原密码错误"
        conn.execute(
            "UPDATE accounts SET password=? WHERE username=?",
            (make_password_hash(new_pwd), username),
        )
    return True, "密码已更新"


init_db()