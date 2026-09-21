"""Shared runtime structures for paired-agent experiments.

An episode contains two agent tasks, three phases, and one reward per agent.
Per-agent fields are keyed by agent ID.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.protocol import DEFAULT_MAX_ROUNDS
from experiments.protocol.rewards import RewardScheme


DEFAULT_LLM_TEMPERATURE = 1.0
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 64_000


@dataclass(frozen=True)
class AgentConfig:
    model: str
    reasoning_effort: str = "default"
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS


@dataclass(frozen=True)
class EpisodeRunConfig:
    """Per-episode settings shared by both agents."""

    char_limit: int = 200
    throttled: bool = True
    max_rounds: int = DEFAULT_MAX_ROUNDS
    cross_episode_memory_scope: str = "full-history"
    verdict_policy: str = "raw-only"
    reflection: bool = True
    reward: bool = True
    verdict_review: bool = True
    # Reward terms and sharing mode applied throughout the run.
    reward_scheme: RewardScheme = RewardScheme()
    controlled_bob_verdict: str = ""
    controlled_bob_cache: str = ""
    controlled_bob_messages: str = "summary"
    controlled_bob_observed_verdict: bool = False
    temp_run_id: str = ""
    episode_index: int = 0
    verbose: bool = True
    usage_journal_path: Path | None = None
    usage_invocation_id: str = ""

    @property
    def verdict_evaluation(self) -> str:
        return self.reward_scheme.verdict_review_direction(self.verdict_review)


@dataclass
class PreparedAgentTask:
    """One agent's prepared task, addressed through ``PreparedPair.agents[agent_id]``."""

    task: dict[str, Any]
    display_id: str
    # The owner receives own_task_prompt; its peer receives the judging and verdict
    # openings. Only the owner's brief includes resource paths.
    own_task_prompt: str
    judging_prompt: str
    verdict_prompt: str
    code_path: Path | None = None
    source_path: Path | None = None
    database_path: Path | None = None
    function_name: str = ""


@dataclass
class PreparedPair:
    task_type: str
    state: dict[str, Any]
    agents: dict[str, PreparedAgentTask] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRuntime:
    agent_id: str
    config: AgentConfig
    messages: list[dict[str, Any]]


@dataclass
class TaskEvaluation:
    """Ground truth and task-specific evaluation details for one agent."""

    expected_verdict: str
    target_expected_verdict: str | None
    code: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    data_search: dict[str, Any] | None = None


@dataclass
class AgentOutcome:
    """One agent's verdict on its peer, verdict accuracy, and reward.

    Verdict accuracy is measured against the peer's ground truth.
    """

    verdict: str
    verdict_correct: bool
    reward: int
    verdict_reasoning: str = ""
    verdict_forced: bool = False
    feedback: str | None = None
    reflection: str | None = None
    reflection_prompt: str | None = None


@dataclass
class EpisodeOutcomes:
    """Episode results with one outcome and reward per agent."""

    agents: dict[str, AgentOutcome] = field(default_factory=dict)

    @property
    def rewards(self) -> dict[str, int]:
        """What each agent was paid, keyed by agent id."""
        return {agent_id: outcome.reward for agent_id, outcome in self.agents.items()}
