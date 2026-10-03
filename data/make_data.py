"""Generate reproducible synthetic customer-support data for CustomerLens.

Produces:
  data/tickets.csv    - ~400 support tickets with messy real-world quirks
  data/customers.csv  - 60 customer records
  data/eval_pairs.json - fixed question/answer pairs for the eval harness

All randomness is seeded, so the output is deterministic.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
N_TICKETS = 400
N_CUSTOMERS = 60
OUT_DIR = Path(__file__).resolve().parent

FIRST = ["Ava", "Liam", "Maya", "Noah", "Priya", "Kenji", "Sofia", "Omar", "Elena", "Raj",
         "Lucas", "Zoe", "Aiden", "Nora", "Ethan", "Ivy", "Kai", "Mila", "Dev", "Ana"]
LAST = ["Sharma", "Garcia", "Kim", "Patel", "Nguyen", "Haddad", "Rossi", "Khan", "Silva",
        "Murphy", "Tanaka", "Ali", "Costa", "Reyes", "Weber", "Okafor", "Lindqvist", "Das"]
COMPANIES = ["Northwind Traders", "Contoso Labs", "Fabrikam Inc", "Acme Retail", "Globex",
             "Initech", "Umbrella Health", "Hooli Media", "Stark Logistics", "Wayne Foods",
             "Massive Dynamic", "Soylent Corp"]
COUNTRIES = ["USA", "Canada", "UK", "Germany", "India", "Brazil", "Australia", "Japan"]
TIERS = ["Standard", "Premium", "Enterprise"]

PRODUCTS = ["Atlas CRM", "Beacon Analytics", "Cobalt Storage", "Drift Mail", "Ember Chat"]
CATEGORIES = ["Billing", "Bug", "Feature Request", "How-to", "Outage", "Refund",
              "Security", "Performance"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]

# Distinctive, template-embedded details that make retrieval deterministic.
TEXT_TEMPLATES = {
    "Billing": [
        "Invoice {inv} charged us twice for the {prod} {tier} plan. The duplicate charge of ${amt} appeared on {date}. Please reverse the extra charge.",
        "Our card was declined for invoice {inv} even though the funds are available. The billing portal shows error code {err} when we retry payment.",
    ],
    "Bug": [
        "The {prod} dashboard throws error {err} whenever we filter by date range. This started after the {date} update and affects all {tier} users on our team.",
        "Exporting reports from {prod} produces a corrupt CSV file. The download fails with error {err} for any report larger than {n} rows.",
    ],
    "Feature Request": [
        "We need {prod} to support SSO via Okta before our security review on {date}. Is single sign-on on the roadmap for {tier} customers?",
        "Please add a bulk-edit option to {prod}. Updating {n} records one by one is not workable for our team of {team} people.",
    ],
    "How-to": [
        "How do we configure data retention in {prod}? We need to keep audit logs for {n} days to satisfy compliance, but the settings page is unclear.",
        "Where can we find API documentation for {prod} webhooks? We want to trigger alerts when error {err} occurs.",
    ],
    "Outage": [
        "{prod} was unreachable from {date} to {date2}. Our status page showed error {err} and our support line had a {n}-minute wait.",
        "Complete outage of {prod} this morning. All {tier} workspaces in {country} returned error {err}. We need a postmortem.",
    ],
    "Refund": [
        "We cancelled {prod} on {date} but were billed ${amt} anyway. Requesting a refund to the original card for invoice {inv}.",
        "Charged ${amt} for {prod} after downgrading to the {tier} plan. Please refund the difference for invoice {inv}.",
    ],
    "Security": [
        "We noticed logins to {prod} from an unknown IP range in {country}. Please confirm whether our {tier} workspace was compromised and enable MFA enforcement.",
        "Our auditor flagged that {prod} exports are not encrypted at rest. We need confirmation of encryption standards before {date}.",
    ],
    "Performance": [
        "{prod} takes over {n} seconds to load the main dashboard. This began after we crossed {big} records. Please advise on performance tuning.",
        "Search in {prod} times out for queries with more than {n} filters. Error {err} appears consistently during peak hours.",
    ],
}


def messy(text: str, rng: random.Random) -> str:
    """Add realistic messiness: odd casing, extra whitespace, typos."""
    roll = rng.random()
    if roll < 0.12:
        return " " + text.replace("  ", " ") + "  "  # stray whitespace
    if roll < 0.20:
        words = text.split()
        idx = rng.randrange(len(words))
        words[idx] = words[idx].upper()  # random SHOUTING word
        return " ".join(words)
    if roll < 0.26:
        return text.replace("the", "teh", 1)  # typo
    return text


def main() -> None:
    rng = random.Random(SEED)
    start = datetime(2026, 1, 5)

    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        cid = f"C-{1000 + i}"
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        customers.append({
            "customer_id": cid,
            "name": name,
            "email": f"{name.lower().replace(' ', '.')}@example.com",
            "company": rng.choice(COMPANIES),
            "tier": rng.choice(TIERS),
            "country": rng.choice(COUNTRIES),
            "joined_at": (start - timedelta(days=rng.randint(30, 900))).date().isoformat(),
        })

    tickets = []
    for i in range(1, N_TICKETS + 1):
        tid = f"T-{2000 + i}"
        cust = rng.choice(customers)
        product = rng.choice(PRODUCTS)
        category = rng.choice(CATEGORIES)
        created = start + timedelta(days=rng.randint(0, 250),
                                   hours=rng.randint(0, 23),
                                   minutes=rng.randint(0, 59))
        template = rng.choice(TEXT_TEMPLATES[category])
        text = template.format(
            prod=product, tier=cust["tier"], inv=f"INV-{rng.randint(80000, 89999)}",
            amt=f"{rng.randint(50, 2000)}.{rng.choice(['00', '50', '99'])}",
            date=(created - timedelta(days=rng.randint(1, 20))).date().isoformat(),
            date2=created.date().isoformat(),
            err=f"ERR-{rng.randint(1000, 9999)}",
            n=rng.randint(10, 5000), big=rng.randint(100000, 900000),
            team=rng.randint(3, 40), country=cust["country"],
        )
        resolved = rng.random() < 0.68
        tickets.append({
            "ticket_id": tid,
            "customer_id": cust["customer_id"],
            "customer_name": cust["name"] if rng.random() > 0.05 else "",  # some missing
            "product": product,
            "category": category,
            "created_at": created.strftime("%Y-%m-%d %H:%M"),
            "priority": rng.choice(PRIORITIES),
            "resolved": str(resolved),
            "text": messy(text, rng),
        })

    # Intentional duplicates: a few tickets appear twice (data-quality realism).
    dupes = rng.sample(tickets, 6)
    tickets.extend([dict(d) for d in dupes])

    tickets_path = OUT_DIR / "tickets.csv"
    customers_path = OUT_DIR / "customers.csv"
    import csv
    with open(tickets_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tickets[0].keys()))
        w.writeheader()
        w.writerows(tickets)
    with open(customers_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(customers[0].keys()))
        w.writeheader()
        w.writerows(customers)

    # Eval pairs: questions built from distinctive tokens in chosen tickets,
    # so a working retriever must surface the right ticket.
    eval_tickets = rng.sample([t for t in tickets[:N_TICKETS]], 12)
    pairs = []
    for t in eval_tickets:
        cust_name = t["customer_name"] or next(
            c["name"] for c in customers if c["customer_id"] == t["customer_id"])
        distinct = [w for w in t["text"].split()
                    if w.startswith(("ERR-", "INV-", "$"))][:2]
        detail = " ".join(distinct) if distinct else t["category"]
        question = (f"What did {cust_name} report about {t['product']} "
                    f"({t['category']})? Details: {detail}")
        pairs.append({
            "question": question,
            "expected_ticket_id": t["ticket_id"],
            "keywords": [t["product"].split()[0], t["category"].split()[0]]
                        + [w.strip(".$") for w in distinct],
        })
    with open(OUT_DIR / "eval_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    print(f"wrote {len(tickets)} tickets -> {tickets_path}")
    print(f"wrote {len(customers)} customers -> {customers_path}")
    print(f"wrote {len(pairs)} eval pairs")


if __name__ == "__main__":
    main()
