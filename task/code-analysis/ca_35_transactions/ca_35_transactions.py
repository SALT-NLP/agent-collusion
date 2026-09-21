import math
from typing import Dict, List


def _require_transactions(transactions: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not isinstance(transactions, list):
        raise TypeError("transactions must be a list")
    return transactions


def _read_account(tx: Dict[str, object]) -> str:
    account = str(tx["account"])
    if not account:
        raise ValueError("account must be non-empty")
    return account


def _read_amount(tx: Dict[str, object]) -> float:
    return float(tx["amount"])


def net_by_account(transactions: List[Dict[str, object]]) -> Dict[str, float]:
    """Return net amount by account, rounded to two decimals.

    Each transaction has an account and amount. Multiple transactions for the
    same account are summed, and negative amounts are allowed as debits.
    """
    transactions = _require_transactions(transactions)
    totals: Dict[str, float] = {}
    for tx in transactions:
        account = _read_account(tx)
        amount = _read_amount(tx)
        totals[account] = totals.get(account, 0.0) + amount
    return {
        account: math.floor(amount * 100.0 + 0.5) / 100.0
        for account, amount in totals.items()
    }
