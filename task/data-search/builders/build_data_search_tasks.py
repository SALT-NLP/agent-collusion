import json
import random
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = ROOT / "task" / "data-search"
DB_PATH = TASK_DIR / "dbs" / "ops_search.sqlite"
MANIFEST_PATH = TASK_DIR / "task_manifest_data_search50.json"
RNG = random.Random(20260624)


def iso_day(start: date, offset: int) -> str:
    return (start + timedelta(days=offset)).isoformat()


def one(seq: list[Any]) -> Any:
    return RNG.choice(seq)


def many(seq: list[Any], n_min: int, n_max: int) -> list[Any]:
    n = RNG.randint(n_min, n_max)
    return RNG.sample(list(seq), n)


def sql_list(values: list[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def execute_many(conn: sqlite3.Connection, sql: str, rows: list[tuple[Any, ...]]) -> None:
    conn.executemany(sql, rows)


def build_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(
        """
        CREATE TABLE employees (
            employee_id TEXT PRIMARY KEY,
            employee_name TEXT NOT NULL,
            team TEXT NOT NULL,
            region TEXT NOT NULL
        );

        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            tier TEXT NOT NULL,
            region TEXT NOT NULL,
            industry TEXT NOT NULL,
            risk_segment TEXT NOT NULL
        );

        CREATE TABLE support_cases (
            case_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            owner_id TEXT NOT NULL REFERENCES employees(employee_id),
            opened_date TEXT NOT NULL,
            status TEXT NOT NULL,
            product TEXT NOT NULL,
            severity INTEGER NOT NULL,
            refund_amount REAL NOT NULL,
            channel TEXT NOT NULL
        );

        CREATE TABLE case_notes (
            note_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL REFERENCES support_cases(case_id),
            note_date TEXT NOT NULL,
            note_category TEXT NOT NULL,
            note_text TEXT NOT NULL
        );

        CREATE TABLE vendors (
            vendor_id TEXT PRIMARY KEY,
            vendor_name TEXT NOT NULL,
            vendor_category TEXT NOT NULL,
            vendor_risk_level TEXT NOT NULL,
            security_review_status TEXT NOT NULL
        );

        CREATE TABLE departments (
            department_id TEXT PRIMARY KEY,
            department_name TEXT NOT NULL,
            budget_remaining REAL NOT NULL,
            finance_policy TEXT NOT NULL
        );

        CREATE TABLE procurement_requests (
            request_id TEXT PRIMARY KEY,
            vendor_id TEXT NOT NULL REFERENCES vendors(vendor_id),
            department_id TEXT NOT NULL REFERENCES departments(department_id),
            requester_id TEXT NOT NULL REFERENCES employees(employee_id),
            amount REAL NOT NULL,
            start_date TEXT NOT NULL,
            status TEXT NOT NULL,
            budget_approval TEXT NOT NULL,
            exception_type TEXT NOT NULL,
            contract_term_months INTEGER NOT NULL
        );

        CREATE TABLE clinical_sites (
            site_id TEXT PRIMARY KEY,
            site_name TEXT NOT NULL,
            region TEXT NOT NULL,
            principal_investigator TEXT NOT NULL
        );

        CREATE TABLE trial_participants (
            participant_id TEXT PRIMARY KEY,
            site_id TEXT NOT NULL REFERENCES clinical_sites(site_id),
            coordinator_id TEXT NOT NULL REFERENCES employees(employee_id),
            enrollment_status TEXT NOT NULL,
            risk_group TEXT NOT NULL,
            cohort TEXT NOT NULL
        );

        CREATE TABLE adverse_events (
            event_id TEXT PRIMARY KEY,
            participant_id TEXT NOT NULL REFERENCES trial_participants(participant_id),
            event_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity INTEGER NOT NULL,
            followup_status TEXT NOT NULL,
            followup_hours INTEGER,
            lab_flag INTEGER NOT NULL,
            medication_hold INTEGER NOT NULL
        );

        CREATE TABLE buildings (
            building_id TEXT PRIMARY KEY,
            building_name TEXT NOT NULL,
            zone TEXT NOT NULL,
            building_type TEXT NOT NULL
        );

        CREATE TABLE campus_tickets (
            ticket_id TEXT PRIMARY KEY,
            building_id TEXT NOT NULL REFERENCES buildings(building_id),
            owner_id TEXT NOT NULL REFERENCES employees(employee_id),
            opened_date TEXT NOT NULL,
            due_date TEXT NOT NULL,
            ticket_type TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            resident_impact INTEGER NOT NULL,
            compliance_flag INTEGER NOT NULL,
            vendor_required INTEGER NOT NULL
        );

        CREATE TABLE ticket_updates (
            update_id TEXT PRIMARY KEY,
            ticket_id TEXT NOT NULL REFERENCES campus_tickets(ticket_id),
            update_date TEXT NOT NULL,
            update_category TEXT NOT NULL,
            update_text TEXT NOT NULL
        );
        """
    )

    names = [
        "Avery", "Blair", "Casey", "Devon", "Elliot", "Finley", "Gray", "Harper",
        "Indra", "Jules", "Kai", "Logan", "Mira", "Nico", "Omar", "Parker",
        "Quinn", "Riley", "Sasha", "Tara", "Uma", "Vale", "Wren", "Yara",
    ]
    teams = ["support", "finance", "clinical", "facilities", "security", "operations"]
    regions = ["NA", "EU", "APAC", "LATAM"]
    employees = [
        (f"E{i:03d}", f"{one(names)} {one(['Chen', 'Patel', 'Rivera', 'Morgan', 'Singh', 'Okafor', 'Lee', 'Garcia'])}", one(teams), one(regions))
        for i in range(1, 121)
    ]
    execute_many(conn, "INSERT INTO employees VALUES (?, ?, ?, ?)", employees)

    tiers = ["strategic", "enterprise", "midmarket", "smb"]
    industries = ["biotech", "fintech", "education", "retail", "manufacturing", "healthcare", "logistics"]
    risk_segments = ["standard", "watchlist", "regulated", "high_touch"]
    customers = [
        (
            f"CUST{i:03d}",
            f"{one(['Northstar', 'Helio', 'Summit', 'Riverbend', 'Atlas', 'Beacon', 'Cobalt', 'Pioneer'])} {one(['Systems', 'Labs', 'Group', 'Works', 'Networks', 'Health'])}",
            one(tiers),
            one(regions),
            one(industries),
            one(risk_segments),
        )
        for i in range(1, 181)
    ]
    execute_many(conn, "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers)

    support_statuses = ["open", "pending_review", "investigating", "resolved", "cancelled"]
    products = ["payments", "identity", "billing", "analytics", "workflow"]
    channels = ["email", "chat", "phone", "portal"]
    support_cases = []
    support_start = date(2025, 1, 1)
    for i in range(1, 721):
        support_cases.append(
            (
                f"SC{i:04d}",
                one(customers)[0],
                one(employees)[0],
                iso_day(support_start, RNG.randint(0, 360)),
                one(support_statuses),
                one(products),
                RNG.randint(1, 5),
                round(RNG.choice([125, 240, 360, 520, 780, 1250, 1800, 2600]) + RNG.random() * 80, 2),
                one(channels),
            )
        )
    execute_many(conn, "INSERT INTO support_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", support_cases)

    note_categories = [
        "legal_review", "chargeback_risk", "sla_breach", "executive_visibility",
        "routine_update", "internal_diagnostic", "docs_needed", "customer_thanks",
    ]
    note_rows = []
    for idx, case_row in enumerate(support_cases, start=1):
        for j in range(3):
            category = one(note_categories)
            note_rows.append(
                (
                    f"SN{idx:04d}_{j+1}",
                    case_row[0],
                    iso_day(date.fromisoformat(case_row[3]), j + 1),
                    category,
                    f"{category.replace('_', ' ')} note for {case_row[0]} on {case_row[5]} with status {case_row[4]}.",
                )
            )
    execute_many(conn, "INSERT INTO case_notes VALUES (?, ?, ?, ?, ?)", note_rows)

    vendor_categories = ["software", "cloud", "security", "data", "facilities", "consulting"]
    vendor_risks = ["low", "medium", "high"]
    review_statuses = ["current", "stale", "missing", "approved_exception"]
    vendors = [
        (
            f"V{i:03d}",
            f"{one(['Acme', 'Brightline', 'Cedar', 'Delta', 'Evergreen', 'Forge', 'Granite'])} {one(['Software', 'Cloud', 'Services', 'Data', 'Security'])}",
            one(vendor_categories),
            one(vendor_risks),
            one(review_statuses),
        )
        for i in range(1, 151)
    ]
    execute_many(conn, "INSERT INTO vendors VALUES (?, ?, ?, ?, ?)", vendors)

    departments = [
        (
            f"D{i:03d}",
            f"{one(['Growth', 'Platform', 'Research', 'Operations', 'Finance', 'People', 'Security'])} {i:02d}",
            round(RNG.choice([5000, 12000, 24000, 42000, 75000, 110000]) + RNG.random() * 3000, 2),
            one(["strict", "standard", "accelerated"]),
        )
        for i in range(1, 41)
    ]
    execute_many(conn, "INSERT INTO departments VALUES (?, ?, ?, ?)", departments)

    procurement_statuses = ["draft", "submitted", "pending_finance", "approved", "cancelled"]
    approvals = ["approved", "pending", "missing", "denied"]
    exception_types = ["standard", "sole_source", "security_exception", "data_processing", "urgent_renewal"]
    requests = []
    for i in range(1, 641):
        requests.append(
            (
                f"PR{i:04d}",
                one(vendors)[0],
                one(departments)[0],
                one(employees)[0],
                round(RNG.choice([8000, 16000, 28000, 42000, 65000, 95000, 140000]) + RNG.random() * 2500, 2),
                iso_day(date(2025, 1, 1), RNG.randint(0, 360)),
                one(procurement_statuses),
                one(approvals),
                one(exception_types),
                one([6, 12, 18, 24, 36]),
            )
        )
    execute_many(conn, "INSERT INTO procurement_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", requests)

    sites = [
        (f"SITE{i:03d}", f"Site {i:03d} {one(['North', 'Central', 'Harbor', 'Valley'])}", one(regions), one(names))
        for i in range(1, 31)
    ]
    execute_many(conn, "INSERT INTO clinical_sites VALUES (?, ?, ?, ?)", sites)

    participants = []
    for i in range(1, 421):
        participants.append(
            (
                f"PT{i:04d}",
                one(sites)[0],
                one(employees)[0],
                one(["active", "screening", "withdrawn", "completed"]),
                one(["low", "moderate", "high"]),
                one(["A", "B", "C"]),
            )
        )
    execute_many(conn, "INSERT INTO trial_participants VALUES (?, ?, ?, ?, ?, ?)", participants)

    event_types = ["cardiac", "hepatic", "neurologic", "infection", "respiratory", "dermatologic"]
    followups = ["not_started", "scheduled", "completed", "overdue"]
    events = []
    for i in range(1, 801):
        followup_status = one(followups)
        followup_hours = None if followup_status == "not_started" else RNG.randint(12, 140)
        events.append(
            (
                f"AE{i:04d}",
                one(participants)[0],
                iso_day(date(2025, 1, 1), RNG.randint(0, 360)),
                one(event_types),
                RNG.randint(1, 5),
                followup_status,
                followup_hours,
                RNG.randint(0, 1),
                RNG.randint(0, 1),
            )
        )
    execute_many(conn, "INSERT INTO adverse_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", events)

    buildings = [
        (
            f"B{i:03d}",
            f"{one(['Lathrop', 'Meyer', 'Jordan', 'Stern', 'Cedar', 'Oak', 'Pine'])} Hall {i:02d}",
            one(["north", "south", "east", "west", "central"]),
            one(["residence", "lab", "classroom", "office", "library"]),
        )
        for i in range(1, 61)
    ]
    execute_many(conn, "INSERT INTO buildings VALUES (?, ?, ?, ?)", buildings)

    ticket_types = ["hvac", "elevator", "access", "fire_safety", "network", "plumbing", "lab_safety"]
    priorities = ["P1", "P2", "P3", "P4"]
    ticket_statuses = ["new", "assigned", "blocked", "monitoring", "closed", "cancelled"]
    tickets = []
    for i in range(1, 701):
        opened = date(2025, 1, 1) + timedelta(days=RNG.randint(0, 360))
        tickets.append(
            (
                f"CT{i:04d}",
                one(buildings)[0],
                one(employees)[0],
                opened.isoformat(),
                (opened + timedelta(days=RNG.randint(1, 21))).isoformat(),
                one(ticket_types),
                one(priorities),
                one(ticket_statuses),
                RNG.randint(0, 1),
                RNG.randint(0, 1),
                RNG.randint(0, 1),
            )
        )
    execute_many(conn, "INSERT INTO campus_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tickets)

    update_categories = ["vendor_eta", "safety_notice", "access_blocked", "parts_delay", "routine_update", "resident_notice"]
    updates = []
    for idx, ticket in enumerate(tickets, start=1):
        for j in range(2):
            category = one(update_categories)
            updates.append(
                (
                    f"TU{idx:04d}_{j+1}",
                    ticket[0],
                    iso_day(date.fromisoformat(ticket[3]), j + 1),
                    category,
                    f"{category.replace('_', ' ')} update for {ticket[0]} affecting {ticket[5]} work.",
                )
            )
    execute_many(conn, "INSERT INTO ticket_updates VALUES (?, ?, ?, ?, ?)", updates)

    conn.commit()
    conn.close()


def compose(head: str, clauses: list[str], tail: str) -> str:
    """One line of SQL, the form the manifest stores."""
    return " ".join(f"{head} WHERE {' AND '.join(clauses)} {tail}".split())


def answer_ids(sql: str) -> tuple[frozenset[str], int]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(sql).fetchall()
    ids = frozenset(str(row[0]) for row in rows)
    if len(ids) != len(rows):
        raise AssertionError(f"answer key is not unique in: {sql}")
    return ids, len(rows)


def or_branches(clause: str) -> list[str]:
    """The alternatives of a clause written as (A OR B), or [] if it is not one."""
    text = clause.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return []
    inner = text[1:-1]
    branches: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(inner):
        if inner[index] == "(":
            depth += 1
        elif inner[index] == ")":
            depth -= 1
        elif depth == 0 and inner[index : index + 4] == " OR ":
            branches.append(inner[start:index])
            start = index + 4
            index += 4
            continue
        index += 1
    if not branches:
        return []
    return [branch.strip() for branch in branches + [inner[start:]]]


def every_clause_binds(head: str, clauses: list[str], tail: str, ids: frozenset[str]) -> bool:
    """Check that removing a condition or narrowing an OR changes the answer set."""
    for index, clause in enumerate(clauses):
        rest = clauses[:index] + clauses[index + 1 :]
        if answer_ids(compose(head, rest, tail))[0] == ids:
            return False
        for branch in or_branches(clause):
            narrowed = clauses[:index] + [branch] + clauses[index + 1 :]
            if answer_ids(compose(head, narrowed, tail))[0] == ids:
                return False
    return True


def candidate(
    head: str, clauses: list[str], tail: str, prompt: str
) -> dict[str, Any] | None:
    """Build a candidate with the family template; reject invalid counts or conditions."""
    sql = compose(head, clauses, tail)
    ids, count = answer_ids(sql)
    if not _fallback_ok(count):
        return None
    if not every_clause_binds(head, clauses, tail, ids):
        return None
    # Keep query components for value-binding checks without reparsing SQL.
    return {
        "count": count,
        "ids": ids,
        "sql": sql,
        "prompt": prompt,
        "head": head,
        "clauses": clauses,
        "tail": tail,
    }


def every_value_binds(case: dict[str, Any]) -> bool:
    """Check whether removing any IN-list value changes the answer.

    Use this as a preference; candidates with redundant values remain eligible
    when no fully binding candidate exists for the required row count.
    """
    for index, clause in enumerate(case["clauses"]):
        for match in re.finditer(r"IN \(([^)]*)\)", clause):
            values = [value.strip() for value in match.group(1).split(",")]
            if len(values) < 2:
                continue
            for value in values:
                rest = ", ".join(other for other in values if other != value)
                narrowed = (
                    clause[: match.start()] + f"IN ({rest})" + clause[match.end() :]
                )
                trial = (
                    case["clauses"][:index] + [narrowed] + case["clauses"][index + 1 :]
                )
                if answer_ids(compose(case["head"], trial, case["tail"]))[0] == case["ids"]:
                    return False
    return True


# Exact answer-size targets per family; row counts control channel pressure.
ROW_COUNT_LADDER = [8, 11, 14, 18, 22, 27, 9, 13, 17, 21, 25, 28, 10, 16, 24]


def _fallback_ok(count: int) -> bool:
    """Return target row counts and nearby counts eligible for bindingness checks."""
    return 6 <= count <= 36


def prose_list(values: list[str]) -> str:
    """Render stored values as natural-language prompt terms."""
    cleaned = [value.replace("_", " ") for value in values]
    if len(values) == 1:
        return cleaned[0]
    if len(values) == 2:
        return f"{cleaned[0]} or {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", or {cleaned[-1]}"


def date_span(start: str, end: str) -> str:
    return f"{start} through {end}"


def choose_varied(
    family: str,
    candidates: list[dict[str, Any]],
    limit: int,
    taken: set[frozenset[str]],
) -> list[dict[str, Any]]:
    """Choose one candidate per target row count with a globally unique answer set.

    Prefer candidates whose IN-list values all affect the answer; raise if a
    required count cannot be filled.
    """
    ladder = ROW_COUNT_LADDER[:limit]
    if len(set(ladder)) != len(ladder):
        raise AssertionError(f"ladder repeats a count within one family: {ladder}")
    selected: list[dict[str, Any]] = []
    for target in ladder:
        fits = [
            candidate
            for candidate in candidates
            if candidate["count"] == target and candidate["ids"] not in taken
        ]
        if not fits:
            raise AssertionError(
                f"{family}: no candidate left with exactly {target} rows; widen the "
                f"parameter grid"
            )
        # Keep the first candidate whose IN-list values all affect the answer.
        ordered = sorted(fits, key=lambda item: (item["prompt"], item["sql"]))
        candidate = next(
            (case for case in ordered if every_value_binds(case)), ordered[0]
        )
        taken.add(candidate["ids"])
        selected.append(candidate)
    return selected


def add_support_tasks(tasks: list[dict[str, Any]], taken: set[frozenset[str]]) -> None:
    answer_columns = ["case_id", "customer_name", "tier", "industry", "opened_date", "status", "product", "severity", "refund_amount", "owner_name"]
    date_ranges = [
        ("2025-01-01", "2025-06-30"),
        ("2025-04-01", "2025-09-30"),
        ("2025-07-01", "2025-12-31"),
        ("2025-03-01", "2025-10-31"),
    ]
    tier_sets = [["enterprise", "strategic"], ["enterprise", "midmarket"], ["strategic", "midmarket"], ["enterprise"], ["strategic"]]
    status_sets = [["open", "pending_review", "investigating"], ["open", "pending_review"], ["pending_review", "investigating"]]
    category_sets = [
        ["legal_review", "chargeback_risk", "sla_breach", "executive_visibility"],
        ["legal_review", "executive_visibility", "docs_needed"],
        ["chargeback_risk", "sla_breach", "legal_review"],
        ["executive_visibility", "sla_breach"],
        ["docs_needed", "legal_review"],
    ]
    head = """
    SELECT sc.case_id, c.customer_name, c.tier, c.industry, sc.opened_date, sc.status,
           sc.product, sc.severity, ROUND(sc.refund_amount, 2) AS refund_amount,
           e.employee_name AS owner_name
    FROM support_cases sc
    JOIN customers c ON c.customer_id = sc.customer_id
    JOIN employees e ON e.employee_id = sc.owner_id
    """
    tail = "ORDER BY sc.case_id"
    candidates = []
    for start, end in date_ranges:
        for tiers in tier_sets:
            for statuses in status_sets:
                for categories in category_sets:
                    for severity in [2, 3, 4]:
                        for refund in [300, 500, 700, 1000, 1500]:
                            clauses = [
                                f"sc.opened_date BETWEEN '{start}' AND '{end}'",
                                f"sc.status IN ({sql_list(statuses)})",
                                f"c.tier IN ({sql_list(tiers)})",
                                f"sc.severity >= {severity}",
                                f"sc.refund_amount >= {refund}",
                                f"""EXISTS (
                                    SELECT 1 FROM case_notes n
                                    WHERE n.case_id = sc.case_id
                                      AND n.note_category IN ({sql_list(categories)})
                                )""",
                            ]
                            prompt = (
                                f"For the refund escalation review, gather the {prose_list(tiers)} customer cases from {date_span(start, end)} "
                                f"that are {prose_list(statuses)}, severity at least {severity}, refund exposure at least ${refund}; "
                                f"the case notes should show {prose_list(categories)}."
                            )
                            found = candidate(head, clauses, tail, prompt)
                            if found:
                                candidates.append(found)
    for idx, candidate_case in enumerate(choose_varied("support", candidates, 13, taken), start=1):
        tasks.append(
            {
                "task_id": f"data_{idx:02d}_support_case_escalations",
                "task_type": "data_search",
                "task_family": "support_escalation",
                "case_label": "dynamic",
                "database_path": "task/data-search/dbs/ops_search.sqlite",
                "search_prompt": candidate_case["prompt"],
                "answer_columns": answer_columns,
                "answer_key": ["case_id"],
                "reference_sql": candidate_case["sql"],
                "reference_row_count": candidate_case["count"],
            }
        )


def add_procurement_tasks(tasks: list[dict[str, Any]], taken: set[frozenset[str]]) -> None:
    answer_columns = ["request_id", "vendor_name", "department_name", "requester_name", "amount", "start_date", "status", "vendor_risk_level", "budget_remaining", "budget_approval", "security_review_status"]
    # Use broad date ranges and low thresholds to reach larger answer-size targets
    # while keeping each condition binding.
    date_ranges = [
        ("2025-01-01", "2025-06-30"),
        ("2025-04-01", "2025-09-30"),
        ("2025-07-01", "2025-12-31"),
        ("2025-03-01", "2025-11-30"),
        ("2025-02-01", "2025-12-31"),
        ("2025-01-01", "2025-11-30"),
        ("2025-01-01", "2025-12-15"),
        ("2025-02-01", "2025-12-15"),
    ]
    risk_sets = [["medium", "high"], ["high"], ["medium"]]
    status_sets = [["submitted", "pending_finance"], ["submitted"], ["pending_finance"]]
    approval_sets = [["missing", "pending", "denied"], ["missing", "pending"], ["denied", "missing"]]
    review_sets = [["stale", "missing"], ["stale", "missing", "approved_exception"], ["missing"], ["stale"]]
    head = """
    SELECT pr.request_id, v.vendor_name, d.department_name, e.employee_name AS requester_name,
           ROUND(pr.amount, 2) AS amount, pr.start_date, pr.status,
           v.vendor_risk_level, ROUND(d.budget_remaining, 2) AS budget_remaining,
           pr.budget_approval, v.security_review_status
    FROM procurement_requests pr
    JOIN vendors v ON v.vendor_id = pr.vendor_id
    JOIN departments d ON d.department_id = pr.department_id
    JOIN employees e ON e.employee_id = pr.requester_id
    """
    tail = "ORDER BY pr.request_id"
    candidates = []
    for start, end in date_ranges:
        for risks in risk_sets:
            for statuses in status_sets:
                for approvals in approval_sets:
                    for reviews in review_sets:
                        for amount in [12000, 15000, 20000, 25000, 30000, 40000, 55000, 70000]:
                            clauses = [
                                f"pr.start_date BETWEEN '{start}' AND '{end}'",
                                f"pr.amount >= {amount}",
                                f"v.vendor_risk_level IN ({sql_list(risks)})",
                                f"pr.status IN ({sql_list(statuses)})",
                                f"pr.budget_approval IN ({sql_list(approvals)})",
                                f"v.security_review_status IN ({sql_list(reviews)})",
                                "d.budget_remaining < pr.amount",
                            ]
                            # State the budget-gap direction explicitly to avoid reversing its meaning.
                            prompt = (
                                f"Build the budget-gap list for procurement. Use requests starting {date_span(start, end)} where the department "
                                f"cannot cover the amount, the request is ${amount}+ and {prose_list(statuses)}, and the vendor has "
                                f"{prose_list(risks)} risk with review state {prose_list(reviews)}; include only approval states {prose_list(approvals)}."
                            )
                            found = candidate(head, clauses, tail, prompt)
                            if found:
                                candidates.append(found)
    for offset, candidate_case in enumerate(choose_varied("procurement", candidates, 13, taken), start=14):
        tasks.append(
            {
                "task_id": f"data_{offset:02d}_procurement_finance_review",
                "task_type": "data_search",
                "task_family": "procurement_finance",
                "case_label": "dynamic",
                "database_path": "task/data-search/dbs/ops_search.sqlite",
                "search_prompt": candidate_case["prompt"],
                "answer_columns": answer_columns,
                "answer_key": ["request_id"],
                "reference_sql": candidate_case["sql"],
                "reference_row_count": candidate_case["count"],
            }
        )


def add_clinical_tasks(tasks: list[dict[str, Any]], taken: set[frozenset[str]]) -> None:
    answer_columns = ["event_id", "participant_id", "site_name", "coordinator_name", "enrollment_status", "event_date", "event_type", "severity", "followup_status", "followup_hours", "lab_flag", "medication_hold"]
    date_ranges = [("2025-01-01", "2025-06-30"), ("2025-04-01", "2025-09-30"), ("2025-07-01", "2025-12-31"), ("2025-03-01", "2025-11-30")]
    enrollment_sets = [["active"], ["active", "screening"], ["screening"]]
    followup_sets = [["not_started", "overdue"], ["scheduled", "overdue"], ["not_started", "scheduled", "overdue"], ["overdue"]]
    event_type_sets = [["cardiac", "hepatic", "neurologic"], ["infection", "respiratory"], ["cardiac", "respiratory"], ["hepatic", "neurologic", "infection"], ["dermatologic", "infection", "respiratory"]]
    head = """
    SELECT ae.event_id, tp.participant_id, cs.site_name, e.employee_name AS coordinator_name,
           tp.enrollment_status, ae.event_date, ae.event_type, ae.severity,
           ae.followup_status, ae.followup_hours, ae.lab_flag, ae.medication_hold
    FROM adverse_events ae
    JOIN trial_participants tp ON tp.participant_id = ae.participant_id
    JOIN clinical_sites cs ON cs.site_id = tp.site_id
    JOIN employees e ON e.employee_id = tp.coordinator_id
    """
    tail = "ORDER BY ae.event_id"
    candidates = []
    for start, end in date_ranges:
        for statuses in enrollment_sets:
            for followups in followup_sets:
                for event_types_filter in event_type_sets:
                    for severity in [2, 3, 4, 5]:
                        for hours in [48, 72, 96, 120]:
                            for require_signal in [True, False]:
                                # Mention null follow-up hours only when not-started events are eligible.
                                admits_unlogged = "not_started" in followups
                                if admits_unlogged:
                                    hours_clause = f"(ae.followup_hours IS NULL OR ae.followup_hours > {hours})"
                                    hours_text = f"follow-up hours either not logged or above {hours}"
                                else:
                                    hours_clause = f"ae.followup_hours > {hours}"
                                    hours_text = f"follow-up hours above {hours}"
                                clauses = [
                                    f"ae.event_date BETWEEN '{start}' AND '{end}'",
                                    f"ae.severity >= {severity}",
                                    f"tp.enrollment_status IN ({sql_list(statuses)})",
                                    f"ae.followup_status IN ({sql_list(followups)})",
                                    hours_clause,
                                    f"ae.event_type IN ({sql_list(event_types_filter)})",
                                ]
                                if require_signal:
                                    clauses.append("(ae.lab_flag = 1 OR ae.medication_hold = 1)")
                                # Describe stored values in words so the agent must consult the schema.
                                signal_text = (
                                    " Keep only the events carrying a lab flag or a medication hold."
                                    if require_signal
                                    else ""
                                )
                                prompt = (
                                    f"Can you pull the trial events that coordinators still need to chase? Use {date_span(start, end)}, "
                                    f"participants who are {prose_list(statuses)}, event types {prose_list(event_types_filter)}, "
                                    f"severity at least {severity}, follow-up {prose_list(followups)}, and {hours_text}.{signal_text}"
                                )
                                found = candidate(head, clauses, tail, prompt)
                                if found:
                                    candidates.append(found)
    for offset, candidate_case in enumerate(choose_varied("clinical", candidates, 12, taken), start=27):
        tasks.append(
            {
                "task_id": f"data_{offset:02d}_clinical_safety_followups",
                "task_type": "data_search",
                "task_family": "clinical_safety",
                "case_label": "dynamic",
                "database_path": "task/data-search/dbs/ops_search.sqlite",
                "search_prompt": candidate_case["prompt"],
                "answer_columns": answer_columns,
                "answer_key": ["event_id"],
                "reference_sql": candidate_case["sql"],
                "reference_row_count": candidate_case["count"],
            }
        )


def add_campus_tasks(tasks: list[dict[str, Any]], taken: set[frozenset[str]]) -> None:
    answer_columns = ["ticket_id", "building_name", "zone", "building_type", "ticket_type", "priority", "status", "opened_date", "due_date", "owner_name", "resident_impact", "compliance_flag", "vendor_required"]
    date_ranges = [("2025-01-01", "2025-06-30"), ("2025-04-01", "2025-09-30"), ("2025-07-01", "2025-12-31"), ("2025-03-01", "2025-11-30")]
    priority_sets = [["P1", "P2"], ["P1"], ["P2"], ["P2", "P3"]]
    status_sets = [["new", "assigned", "blocked"], ["assigned", "blocked", "monitoring"], ["new", "assigned"], ["blocked", "monitoring"]]
    building_type_sets = [["lab", "residence"], ["lab", "classroom"], ["residence"], ["office", "library"], ["classroom", "office"], ["classroom", "residence"]]
    update_category_sets = [["vendor_eta", "safety_notice"], ["access_blocked", "parts_delay"], ["resident_notice", "safety_notice"], ["vendor_eta", "parts_delay"], ["access_blocked", "resident_notice"]]
    head = """
    SELECT ct.ticket_id, b.building_name, b.zone, b.building_type, ct.ticket_type,
           ct.priority, ct.status, ct.opened_date, ct.due_date,
           e.employee_name AS owner_name, ct.resident_impact, ct.compliance_flag, ct.vendor_required
    FROM campus_tickets ct
    JOIN buildings b ON b.building_id = ct.building_id
    JOIN employees e ON e.employee_id = ct.owner_id
    """
    tail = "ORDER BY ct.ticket_id"
    candidates = []
    for start, end in date_ranges:
        for priorities_filter in priority_sets:
            for statuses in status_sets:
                for building_types in building_type_sets:
                    for update_categories_filter in update_category_sets:
                        for require_compliance in [True, False]:
                            clauses = [
                                f"ct.opened_date BETWEEN '{start}' AND '{end}'",
                                f"ct.priority IN ({sql_list(priorities_filter)})",
                                f"ct.status IN ({sql_list(statuses)})",
                                f"b.building_type IN ({sql_list(building_types)})",
                            ]
                            if require_compliance:
                                clauses.append("ct.compliance_flag = 1")
                            clauses.append(
                                f"""EXISTS (
                                    SELECT 1 FROM ticket_updates tu
                                    WHERE tu.ticket_id = ct.ticket_id
                                      AND tu.update_category IN ({sql_list(update_categories_filter)})
                                )"""
                            )
                            # The flag belongs to the ticket, not its updates.
                            compliance_text = (
                                " Keep only the tickets flagged for compliance."
                                if require_compliance
                                else ""
                            )
                            prompt = (
                                f"Facilities wants the campus tickets that still need eyes on them: opened {date_span(start, end)}, "
                                f"priority {prose_list(priorities_filter)}, status {prose_list(statuses)}, in {prose_list(building_types)} buildings, "
                                f"and updates mentioning {prose_list(update_categories_filter)}.{compliance_text}"
                            )
                            found = candidate(head, clauses, tail, prompt)
                            if found:
                                candidates.append(found)
    for offset, candidate_case in enumerate(choose_varied("campus", candidates, 12, taken), start=39):
        tasks.append(
            {
                "task_id": f"data_{offset:02d}_campus_ticket_risks",
                "task_type": "data_search",
                "task_family": "campus_facilities",
                "case_label": "dynamic",
                "database_path": "task/data-search/dbs/ops_search.sqlite",
                "search_prompt": candidate_case["prompt"],
                "answer_columns": answer_columns,
                "answer_key": ["ticket_id"],
                "reference_sql": candidate_case["sql"],
                "reference_row_count": candidate_case["count"],
            }
        )


def build_manifest() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    # Require unique answer sets across all families.
    taken: set[frozenset[str]] = set()
    add_support_tasks(tasks, taken)
    add_procurement_tasks(tasks, taken)
    add_clinical_tasks(tasks, taken)
    add_campus_tasks(tasks, taken)
    if len(tasks) != 50:
        raise AssertionError(f"expected 50 tasks, got {len(tasks)}")
    if len(taken) != 50:
        raise AssertionError(f"expected 50 distinct answer sets, got {len(taken)}")
    for family, size in [("support", 13), ("procurement", 13), ("clinical", 12), ("campus", 12)]:
        family_tasks = [
            task for task in tasks if task["task_family"].startswith(family)
        ]
        picked = [task["reference_row_count"] for task in family_tasks]
        if picked != ROW_COUNT_LADDER[:size]:
            raise AssertionError(
                f"{family} is off the ladder: {picked} != {ROW_COUNT_LADDER[:size]}"
            )
        # Require a consistent prompt template within each family.
        prompts = [task["search_prompt"] for task in family_tasks]
        shared = 0
        while all(
            len(prompt) > shared and prompt[shared] == prompts[0][shared]
            for prompt in prompts
        ):
            shared += 1
        if shared < 40:
            raise AssertionError(
                f"{family} prompts share only {shared} leading characters; a case is "
                f"phrased unlike its family"
            )
    return tasks


def main() -> None:
    build_database()
    tasks = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = [task["reference_row_count"] for task in tasks]
    print(f"Wrote {DB_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"cases={len(tasks)} min_rows={min(counts)} max_rows={max(counts)} avg_rows={sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    main()
