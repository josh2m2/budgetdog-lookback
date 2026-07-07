"""
Budgetdog Three-Month Lookback — Backend API

Architecture:
  Pass 1 (per PDF) — Claude Vision reads the PDF, returns each transaction
                     with a category label already attached.
  Python           — Does ALL the maths: totals, averages, trends, subscriptions.
                     Numbers are now deterministic and consistent across runs.
  Pass 2 (once)   — Claude reads the Python-computed summary and writes the
                     narrative revelation cards. Only judgment, no arithmetic.
"""

import os, json, base64, re
from collections import defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to Claude Ops/lookback/.env")

app = Flask(__name__)
CORS(app)

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

CATEGORIES = [
    "Investments & Savings",     # Vanguard, Fidelity, Schwab, River, Coinbase, brokerage transfers
    "Debt Payments",             # Mortgage, car loan, student loan, personal loan
    "Food Delivery",             # DoorDash, UberEats, Deliveroo
    "Dining Out",                # Restaurants, cafes, bars
    "Groceries",                 # Supermarkets, Costco, Walmart grocery
    "Subscriptions & Streaming", # Netflix, Spotify, gym, SaaS — NOT investments
    "Shopping",                  # Amazon, retail, clothing, electronics
    "Transportation",            # Gas, Uber, parking, car maintenance
    "Utilities & Phone",         # Electric, water, internet, phone bill
    "Insurance",                 # Health, auto, home, life insurance premiums
    "Healthcare",                # Doctors, pharmacy, dental, therapy
    "Entertainment",             # Concerts, sports, hobbies, gaming
    "Personal Care",             # Salon, spa, barber, skincare
    "Kids & Family",             # Daycare, school, kids activities
    "Fees & Charges",            # Bank fees, late fees, card annual fees only
    "Other",                     # Genuinely uncategorisable
]

BENCHMARKS = {
    "Food Delivery":              {"avg_monthly": 165,  "caution": 250,  "flag": 400,   "note": "Average US household spends ~$165/mo on food delivery"},
    "Dining Out":                 {"avg_monthly": 350,  "caution": 500,  "flag": 800,   "note": "Average US household spends ~$350/mo dining out"},
    "Subscriptions & Streaming":  {"avg_monthly": 219,  "caution": 300,  "flag": 450,   "note": "Average American pays $219/mo in subscriptions (Forbes 2024)"},
    "Groceries":                  {"avg_monthly": 500,  "caution": 900,  "flag": 1200,  "note": "Average US household spends ~$500/mo on groceries"},
    "Shopping":                   {"avg_monthly": 400,  "caution": 700,  "flag": 1200,  "note": None},
    "Fees & Charges":             {"avg_monthly": 0,    "caution": 1,    "flag": 1,     "note": "Bank and card fees are 100% avoidable with the right system"},
}


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def extract_transactions(pdf_bytes: bytes) -> list[dict]:
    """
    Claude reads the PDF and returns each transaction already categorised.
    Claude's only job here: read text, clean merchant names, pick a category.
    No arithmetic — Python does that.
    """
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    cat_list = ", ".join(CATEGORIES)

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
                },
                {
                    "type": "text",
                    "text": (
                        "Extract every expense transaction from this bank or credit card statement.\n\n"
                        "Return ONLY a valid JSON array. Each item must have exactly these fields:\n"
                        "  - \"date\": transaction date as \"YYYY-MM-DD\"\n"
                        "  - \"merchant\": clean, human-readable name "
                        "(e.g. \"DoorDash\" not \"DOORDASH*ORDER9182\", \"Netflix\" not \"NETFLIX.COM\", "
                        "\"Vanguard\" not \"VANGUARD BROKERAGE\")\n"
                        "  - \"amount\": the charge as a positive number (e.g. 12.99)\n"
                        f"  - \"category\": one of these EXACT values: {cat_list}\n\n"
                        "EXCLUDE entirely (do not include these at all):\n"
                        "  - Credit card bill payments from a checking account "
                        "(e.g. 'Payment to Chase', 'AUTOPAY THANK YOU', 'AMEX PAYMENT') — "
                        "the individual charges on the card are the real spending, not this lump payment\n"
                        "  - Payroll deposits, direct deposits, refunds, cashback, or any income\n"
                        "  - Transfers between the person's own bank accounts (e.g. 'Transfer to Savings')\n\n"
                        "CATEGORISATION GUIDE — follow these exactly:\n\n"
                        "Investments & Savings (transfer to investment accounts — INCLUDE these, do not exclude):\n"
                        "  Vanguard, Fidelity, Charles Schwab, River, Coinbase, Robinhood, E*TRADE,\n"
                        "  TD Ameritrade, Interactive Brokers, Betterment, Wealthfront, Acorns,\n"
                        "  M1 Finance, TIAA, any brokerage or investment platform transfer,\n"
                        "  Bitcoin purchases, cryptocurrency purchases, 401k contributions,\n"
                        "  IRA contributions, any transfer described as 'investment' or 'brokerage'\n\n"
                        "Debt Payments:\n"
                        "  Mortgage payments, car loan payments, student loan payments (Navient, Sallie Mae,\n"
                        "  FedLoan, MOHELA), personal loan payments — any scheduled debt repayment\n\n"
                        "Insurance:\n"
                        "  Health insurance premiums, auto insurance (Geico, Progressive, State Farm, Allstate),\n"
                        "  home/renters insurance, life insurance, umbrella policies\n\n"
                        "Food Delivery: DoorDash, UberEats, Deliveroo, Grubhub, Instacart (restaurant orders)\n"
                        "Dining Out: restaurants, cafes, bars, coffee shops\n"
                        "Groceries: supermarkets, Costco, Walmart grocery, Trader Joe's, Whole Foods\n"
                        "Subscriptions & Streaming: Netflix, Spotify, Apple TV, Disney+, Hulu, gym memberships,\n"
                        "  SaaS tools, magazine subscriptions — NOT investment platforms\n"
                        "Shopping: Amazon (retail purchases), clothing, electronics, home goods\n"
                        "Transportation: gas stations, Uber/Lyft rides, parking, car maintenance, tolls\n"
                        "Utilities & Phone: electric, water, gas utility, internet, mobile phone bill\n"
                        "Healthcare: doctors, pharmacy, dental, vision, therapy, hospital\n"
                        "Entertainment: concerts, sports tickets, movies, gaming, hobbies\n"
                        "Personal Care: salon, spa, barber, skincare, gym (if not subscription)\n"
                        "Kids & Family: daycare, school fees, kids activities, tutoring\n"
                        "Fees & Charges: bank overdraft fees, late payment fees, annual card fees ONLY\n"
                        "  (NOT ATM cash withdrawals — those go to Other)\n\n"
                        "Return ONLY the JSON array. No explanation, no markdown fences."
                    ),
                },
            ],
        }],
    )

    raw = extract_json(response.content[0].text)
    return json.loads(raw)


def aggregate(transactions: list[dict], num_months: int = 3) -> dict:
    """
    Pure Python maths — deterministic, consistent across every run.
    Groups transactions by month from the date field.
    """
    # Sort all transactions by date to determine actual month ordering
    dated = []
    undated = []
    for t in transactions:
        try:
            dt = datetime.strptime(t["date"], "%Y-%m-%d")
            dated.append((dt, t))
        except (ValueError, KeyError):
            undated.append(t)

    dated.sort(key=lambda x: x[0])

    # Derive actual month keys from the data (up to num_months most recent)
    seen_months = []
    for dt, _ in dated:
        mk = f"{dt.year}-{dt.month:02d}"
        if mk not in seen_months:
            seen_months.append(mk)
    month_keys = seen_months[-num_months:] if len(seen_months) > num_months else seen_months
    actual_months = len(month_keys) if month_keys else num_months

    # Totals
    total_spend = sum(float(t.get("amount", 0)) for t in transactions)
    monthly_average = total_spend / actual_months

    # Category aggregation
    cat_total:    dict[str, float]            = defaultdict(float)
    cat_by_month: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for dt, t in dated:
        mk = f"{dt.year}-{dt.month:02d}"
        cat = t.get("category", "Other")
        amt = float(t.get("amount", 0))
        cat_total[cat]       += amt
        cat_by_month[cat][mk] += amt

    categories = []
    for cat_name in CATEGORIES:
        total = cat_total.get(cat_name, 0)
        if total == 0:
            continue

        monthly_avg = total / actual_months

        # Trend: first month vs last month in the data
        trend, trend_pct = "flat", 0
        if len(month_keys) >= 2:
            first = cat_by_month[cat_name].get(month_keys[0], 0)
            last  = cat_by_month[cat_name].get(month_keys[-1], 0)
            if first > 0:
                trend_pct = round(((last - first) / first) * 100)
                trend = "up" if trend_pct >= 15 else ("down" if trend_pct <= -15 else "flat")

        # Benchmark
        bm = BENCHMARKS.get(cat_name)
        bm_fields = {}
        if bm:
            bm_fields = {
                "benchmark_avg":  bm["avg_monthly"],
                "benchmark_note": bm["note"],
                "benchmark_flag": (
                    "high"     if monthly_avg >= bm["flag"]    else
                    "elevated" if monthly_avg >= bm["caution"] else
                    "normal"
                ),
            }

        categories.append({
            "name":           cat_name,
            "monthly_average": round(monthly_avg, 2),
            "total":           round(total, 2),
            "trend":           trend,
            "trend_pct":       trend_pct,
            **bm_fields,
        })

    # Subscription detection: merchants categorised as Subscriptions that
    # appear in multiple months
    sub_merchant: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for dt, t in dated:
        if t.get("category") == "Subscriptions & Streaming":
            mk = f"{dt.year}-{dt.month:02d}"
            sub_merchant[t.get("merchant", "Unknown")][mk] += float(t.get("amount", 0))

    subscriptions = []
    for merchant, months in sub_merchant.items():
        months_seen = len(months)
        avg_cost = sum(months.values()) / months_seen
        subscriptions.append({
            "name":         merchant,
            "monthly_cost": round(avg_cost, 2),
            "months_seen":  months_seen,
        })
    subscriptions.sort(key=lambda x: x["monthly_cost"], reverse=True)

    # Fee detection — only what Claude explicitly categorised as Fees & Charges
    fees = []
    fee_txns = [(dt, t) for dt, t in dated if t.get("category") == "Fees & Charges"]
    if fee_txns:
        fees.append({
            "type":  "Bank and card fees",
            "total": round(sum(float(t.get("amount", 0)) for _, t in fee_txns), 2),
            "count": len(fee_txns),
        })

    return {
        "total_spend":     round(total_spend, 2),
        "monthly_average": round(monthly_average, 2),
        "categories":      categories,
        "subscriptions":   subscriptions,
        "fees":            fees,
        "month_keys":      month_keys,
        "actual_months":   actual_months,
    }


def generate_revelations(summary: dict, transactions: list[dict]) -> list[dict]:
    """
    Claude reads the pre-computed Python summary and writes narrative revelations.
    All amounts come from Python — Claude only writes the prose.
    """
    cat_lines = "\n".join(
        f"  {c['name']}: {c['monthly_average']}/mo avg "
        f"(3-mo total {c['total']}, trend {c['trend']} {c['trend_pct']}%)"
        for c in summary["categories"]
    )
    sub_lines = "\n".join(
        f"  {s['name']}: {s['monthly_cost']}/mo ({s['months_seen']} months detected)"
        for s in summary["subscriptions"]
    ) or "  None detected"
    fee_lines = "\n".join(
        f"  {f['type']}: {f['total']} total, {f['count']} occurrences"
        for f in summary["fees"]
    ) or "  None detected"

    top_txns = sorted(transactions, key=lambda t: float(t.get("amount", 0)), reverse=True)[:20]
    top_lines = "\n".join(
        f"  {t['date']}  {t['merchant']}  {t.get('amount')}  [{t.get('category','Other')}]"
        for t in top_txns
    )

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": (
                "You are a financial coach reviewing a client's 3-month spending summary. "
                "Write 3–5 'aha moment' revelation cards.\n\n"
                f"CATEGORIES (Python-computed, use these exact amounts):\n{cat_lines}\n\n"
                f"SUBSCRIPTIONS:\n{sub_lines}\n\n"
                f"FEES:\n{fee_lines}\n\n"
                f"TOP TRANSACTIONS BY SIZE:\n{top_lines}\n\n"
                "Return a JSON array. Each item:\n"
                "  - \"headline\": one punchy sentence — use real amounts from the data above\n"
                "  - \"detail\": one follow-up sentence with context\n"
                "  - \"type\": spike | subscription | fee | pattern\n"
                "  - \"impact_monthly\": monthly dollar impact as a number "
                "(use the actual category monthly_average or transaction amount — no estimates)\n\n"
                "Priority order: categories trending up >40%, single large transactions, "
                "subscriptions, any fees or cash withdrawals, surprising totals.\n"
                "Framing: discovery not judgment. 'Your dining spend jumped...' not 'You overspend on...'\n"
                "Return ONLY the JSON array. No markdown."
            ),
        }],
    )

    raw = extract_json(response.content[0].text)
    return json.loads(raw)


# ── Route ─────────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    name  = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    files = request.files.getlist("files")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded."}), 400

    all_transactions: list[dict] = []
    warnings: list[str] = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            warnings.append(f"Skipped {f.filename} — only PDF files are supported.")
            continue
        try:
            txns = extract_transactions(f.read())
            all_transactions.extend(txns)
        except Exception as e:
            warnings.append(f"Could not parse {f.filename}: {str(e)}")

    if not all_transactions:
        return jsonify({"error": "Could not extract any transactions. " + " ".join(warnings)}), 400

    try:
        summary      = aggregate(all_transactions)
        revelations  = generate_revelations(summary, all_transactions)

        result = {
            **summary,
            "revelations":      revelations,
            "transaction_count": len(all_transactions),
            "prospect":         {"name": name, "email": email, "phone": phone},
        }
        if warnings:
            result["warnings"] = warnings
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(__file__), "lookback.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting Budgetdog Lookback API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
