"""Two-agent tool protocol internals."""

MEMORY_SCOPE = "cross-episode"

# Default communication rounds; all configured rounds run.
DEFAULT_MAX_ROUNDS = 5


# Separate task and delivery budgets so task complexity does not reduce channel retries.
TASK_ATTEMPTS = 15
COMMUNICATION_ATTEMPTS = 3

# Per-agent retries for malformed verdict submissions.
VERDICT_ATTEMPTS = 3
