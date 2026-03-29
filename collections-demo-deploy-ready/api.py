from __future__ import annotations

import json
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import requests

app = FastAPI(title="Collections Intelligence API")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
WEB_DIR = Path(os.getenv("WEB_DIR", str(BASE_DIR / "web"))).resolve()

AGING_FILE = DATA_DIR / "aging_snapshot.csv"
CUSTOMERS_FILE = DATA_DIR / "customers.csv"
PAYMENTS_FILE = DATA_DIR / "payments.csv"
DISPUTES_FILE = DATA_DIR / "disputes.csv"
COMM_FILE = DATA_DIR / "communication_log.csv"

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "20"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, Any] = {"data": None, "loaded_at": None}


def safe_num(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (int, str, bool)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def safe_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: safe_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_obj(v) for v in obj]
    return safe_num(obj)


def safe_json(data: Any) -> JSONResponse:
    return JSONResponse(content=safe_obj(data))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def derive_confidence_and_action(df_aging: pd.DataFrame) -> pd.DataFrame:
    df = df_aging.copy()
    dispute_col = "is_disputed_open" if "is_disputed_open" in df.columns else None

    def conf_row(r) -> float:
        dpd = float(r.get("days_past_due") or 0)
        open_amt = float(r.get("open_amount") or 0)

        base = 0.55
        if dpd >= 90:
            base += 0.22
        elif dpd >= 61:
            base += 0.16
        elif dpd >= 31:
            base += 0.10
        elif dpd >= 1:
            base += 0.06

        if open_amt >= 50000:
            base += 0.08
        elif open_amt >= 10000:
            base += 0.05
        elif open_amt >= 1000:
            base += 0.02

        if dispute_col:
            try:
                if bool(r.get(dispute_col)):
                    base -= 0.18
            except Exception:
                pass

        return max(0.05, min(0.99, base))

    def action_row(r) -> str:
        dpd = float(r.get("days_past_due") or 0)
        disputed = False
        if dispute_col:
            try:
                disputed = bool(r.get(dispute_col))
            except Exception:
                disputed = False

        if disputed:
            return "Resolve Dispute / Validate Backup"
        if dpd >= 90:
            return "Escalate: Pre-Collections / Legal Review"
        if dpd >= 61:
            return "Escalate: Manager Review + AM Notify"
        if dpd >= 31:
            return "Call + Email Sequence"
        if dpd >= 1:
            return "Reminder + Statement"
        return "Monitor"

    df["confidence"] = df.apply(conf_row, axis=1)
    df["recommended_action"] = df.apply(action_row, axis=1)
    return df


def priority_score_row(open_amount: float, days_past_due: float, confidence: float, disputed: bool) -> float:
    oa = max(0.0, float(open_amount or 0))
    dpd = max(0.0, float(days_past_due or 0))
    c = max(0.0, min(1.0, float(confidence or 0.0)))

    time_factor = 1.0 + min(3.0, dpd / 45.0)
    dispute_factor = 0.75 if disputed else 1.0
    return oa * time_factor * (0.5 + 0.5 * c) * dispute_factor


def load_data(force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if _cache["data"] is not None and not force:
        return _cache["data"]

    df_aging = read_csv(AGING_FILE)
    df_customers = read_csv(CUSTOMERS_FILE)
    df_payments = read_csv(PAYMENTS_FILE)
    df_disputes = read_csv(DISPUTES_FILE)
    df_comm = read_csv(COMM_FILE)

    required_aging = ["invoice_id", "customer_id", "open_amount", "days_past_due", "aging_bucket"]
    missing_aging = [c for c in required_aging if c not in df_aging.columns]
    if missing_aging:
        raise KeyError(f"aging_snapshot.csv missing columns: {missing_aging}. Found: {list(df_aging.columns)}")

    if "customer_id" not in df_customers.columns:
        raise KeyError(f"customers.csv missing 'customer_id'. Found: {list(df_customers.columns)}")

    df_aging["open_amount"] = pd.to_numeric(df_aging["open_amount"], errors="coerce")
    df_aging["days_past_due"] = pd.to_numeric(df_aging["days_past_due"], errors="coerce")
    df_aging = derive_confidence_and_action(df_aging)

    _cache["data"] = (df_aging, df_customers, df_payments, df_disputes, df_comm)
    _cache["loaded_at"] = now_iso()
    return _cache["data"]


def ollama_health() -> bool:
    if not OLLAMA_ENABLED:
        return False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def ollama_chat(prompt: str, system: str = "") -> str:
    if not OLLAMA_ENABLED:
        raise RuntimeError("Ollama disabled for demo")
    body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=body, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise RuntimeError("Unexpected Ollama response")
        return msg
    except requests.Timeout as e:
        raise RuntimeError("timed out") from e
    except requests.RequestException as e:
        raise RuntimeError(str(e)) from e


def build_queue(df_aging: pd.DataFrame, df_customers: pd.DataFrame) -> pd.DataFrame:
    cust_cols = ["customer_id"]
    if "customer_name" in df_customers.columns:
        cust_cols.append("customer_name")
    df_cust = df_customers[cust_cols].drop_duplicates("customer_id")

    df = df_aging.merge(df_cust, on="customer_id", how="left")
    if "customer_name" not in df.columns:
        df["customer_name"] = df["customer_id"]

    disputed = df["is_disputed_open"].fillna(False).astype(bool) if "is_disputed_open" in df.columns else pd.Series([False] * len(df))
    df["priority_score"] = [
        priority_score_row(oa, dpd, conf, bool(d))
        for oa, dpd, conf, d in zip(
            df["open_amount"].fillna(0),
            df["days_past_due"].fillna(0),
            df["confidence"].fillna(0),
            disputed,
        )
    ]
    return df


def why_payload(customer_id: str, invoice_id: str) -> Dict[str, Any]:
    df_aging, df_customers, df_payments, df_disputes, df_comm = load_data()
    df = build_queue(df_aging, df_customers)

    row = df[(df["customer_id"] == customer_id) & (df["invoice_id"] == invoice_id)]
    if row.empty:
        return {"error": "Not found", "customer_id": customer_id, "invoice_id": invoice_id}

    r = row.iloc[0].to_dict()

    cust_name = r.get("customer_name") or customer_id
    dpd = float(r.get("days_past_due") or 0)
    oa = float(r.get("open_amount") or 0)
    disputed = bool(r.get("is_disputed_open")) if "is_disputed_open" in r else False

    signals = []
    if dpd >= 90:
        signals.append("Invoice is 90+ days past due.")
    elif dpd >= 61:
        signals.append("Invoice is 61–90 days past due.")
    elif dpd >= 31:
        signals.append("Invoice is 31–60 days past due.")
    elif dpd >= 1:
        signals.append("Invoice is past due.")
    else:
        signals.append("Invoice is current.")

    if oa >= 50000:
        signals.append("Large open balance with meaningful cash impact.")
    elif oa >= 10000:
        signals.append("Meaningful open balance.")

    if disputed:
        signals.append("Dispute is open, which changes the right next step.")

    risks = []
    if dpd >= 61:
        risks.append("Higher probability of delayed cash receipt.")
    if dpd >= 90:
        risks.append("Escalation risk increases as aging deepens.")
    if disputed:
        risks.append("Collections velocity slows until dispute resolution is clear.")

    next_actions = [str(r.get("recommended_action") or "Review account")]
    if disputed:
        next_actions.append("Confirm dispute owner and required backup.")
    else:
        next_actions.append("Confirm last contact and target payment date.")

    base = {
        "customer_id": customer_id,
        "customer_name": cust_name,
        "invoice_id": invoice_id,
        "open_amount": oa,
        "days_past_due": dpd,
        "aging_bucket": r.get("aging_bucket"),
        "confidence": float(r.get("confidence") or 0),
        "recommended_action": r.get("recommended_action"),
        "why_summary": f"{cust_name} / {invoice_id} is {int(dpd)} DPD with ${oa:,.2f} open. Recommended action is {r.get('recommended_action')}.",
        "top_signals": signals,
        "risks": risks,
        "next_best_actions": next_actions,
    }

    if DEMO_MODE or not ollama_health():
        return base

    prompt = (
        "Rewrite the following into concise JSON with keys: "
        "why_summary, top_signals, risks, next_best_actions. "
        "Do not invent facts.\n\n"
        f"{json.dumps(base, indent=2)}"
    )
    try:
        raw = ollama_chat(prompt, system="You are a senior collections strategist. Respond with valid JSON only.")
        parsed = json.loads(raw)
        return {**base, **parsed}
    except Exception:
        return base


@app.get("/")
def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return safe_json({
        "status": "ok",
        "loaded_at": _cache.get("loaded_at"),
        "demo_mode": DEMO_MODE,
        "ollama_enabled": OLLAMA_ENABLED,
    })


@app.get("/api/kpis")
def kpis():
    df_aging, df_customers, df_payments, df_disputes, df_comm = load_data()

    total_open = float(df_aging["open_amount"].fillna(0).sum())
    invoice_count = int(df_aging["invoice_id"].nunique())
    avg_dpd = float(df_aging["days_past_due"].fillna(0).mean())
    disputes_open = 0
    if "is_disputed_open" in df_aging.columns:
        disputes_open = int(df_aging["is_disputed_open"].fillna(False).astype(bool).sum())

    return safe_json({
        "total_open": total_open,
        "invoice_count": invoice_count,
        "avg_dpd": avg_dpd,
        "disputes_open": disputes_open,
    })


@app.get("/api/queue")
def queue(
    bucket: str = Query("All"),
    min_conf: float = Query(0.70),
    top_n: int = Query(25),
    search: str = Query(""),
):
    df_aging, df_customers, df_payments, df_disputes, df_comm = load_data()
    df = build_queue(df_aging, df_customers)

    if bucket and bucket != "All":
        df = df[df["aging_bucket"].astype(str) == str(bucket)]

    df = df[df["confidence"].fillna(0) >= float(min_conf)]

    s = (search or "").strip().lower()
    if s:
        df = df[
            df["customer_name"].astype(str).str.lower().str.contains(s, na=False)
            | df["invoice_id"].astype(str).str.lower().str.contains(s, na=False)
            | df["customer_id"].astype(str).str.lower().str.contains(s, na=False)
        ]

    df = df.sort_values(["priority_score", "days_past_due", "open_amount"], ascending=[False, False, False]).head(int(top_n))

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "customer_id": r.get("customer_id"),
            "customer_name": r.get("customer_name") or r.get("customer_id"),
            "invoice_id": r.get("invoice_id"),
            "open_amount": float(r.get("open_amount") or 0),
            "days_past_due": float(r.get("days_past_due") or 0),
            "aging_bucket": r.get("aging_bucket"),
            "confidence": float(r.get("confidence") or 0),
            "recommended_action": r.get("recommended_action"),
            "priority_score": float(r.get("priority_score") or 0),
        })

    return safe_json({"rows": rows})


@app.get("/api/detail")
def detail(customer_id: str, invoice_id: str):
    df_aging, df_customers, df_payments, df_disputes, df_comm = load_data()
    df = build_queue(df_aging, df_customers)

    row = df[(df["customer_id"] == customer_id) & (df["invoice_id"] == invoice_id)]
    if row.empty:
        return safe_json({"error": "Not found", "customer_id": customer_id, "invoice_id": invoice_id})

    r = row.iloc[0].to_dict()
    return safe_json({
        "customer_id": customer_id,
        "customer_name": r.get("customer_name") or customer_id,
        "invoice_id": invoice_id,
        "open_amount": float(r.get("open_amount") or 0),
        "days_past_due": float(r.get("days_past_due") or 0),
        "aging_bucket": r.get("aging_bucket"),
        "recommended_action": r.get("recommended_action"),
        "confidence": float(r.get("confidence") or 0),
        "is_disputed_open": bool(r.get("is_disputed_open")) if "is_disputed_open" in r else False,
        "currency": r.get("currency") if "currency" in r else "USD",
        "due_date": r.get("due_date") if "due_date" in r else None,
        "as_of_date": r.get("as_of_date") if "as_of_date" in r else None,
    })


@app.get("/api/learning")
def learning():
    return safe_json({
        "status": "ok",
        "note": "Learning endpoint active. Public demo uses deterministic mode.",
    })


@app.get("/api/brief_llm")
def brief_llm(top_n: int = Query(15)):
    df_aging, df_customers, df_payments, df_disputes, df_comm = load_data()
    df = build_queue(df_aging, df_customers).sort_values(
        ["priority_score", "days_past_due", "open_amount"],
        ascending=[False, False, False]
    ).head(int(top_n))

    rows = [
        f"{r.customer_name} / {r.invoice_id} | open={r.open_amount:.2f} | dpd={r.days_past_due:.0f} | action={r.recommended_action}"
        for _, r in df.iterrows()
    ]

    fallback = {
        "brief": "Top priorities are driven by aging, balance size, and dispute status. Focus first on older, higher-balance accounts and resolve open disputes before aggressive outreach.",
        "top_n": int(top_n),
        "mode": "fallback" if DEMO_MODE or not ollama_health() else "ollama"
    }

    if DEMO_MODE or not ollama_health():
        return safe_json(fallback)

    prompt = (
        "Write a concise daily collections brief with 5 bullets. "
        "Focus on risk, escalation, and immediate actions.\n\n"
        + "\n".join(rows)
    )
    try:
        text = ollama_chat(prompt, system="Be concise. Business tone.")
        return safe_json({"brief": text, "top_n": int(top_n), "mode": "ollama"})
    except Exception:
        return safe_json(fallback)


@app.get("/api/why_llm")
def why_llm(customer_id: str, invoice_id: str):
    return safe_json(why_payload(customer_id, invoice_id))


@app.get("/api/explain_llm")
def explain_llm(customer_id: str, invoice_id: str):
    return safe_json(why_payload(customer_id, invoice_id))
