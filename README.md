# Emergent Collusion in Long-Horizon LLM Agent Interaction

**Xinrui Shi\*, Yanzhe Zhang\*, Diyi Yang**

\*Equal contribution.

📄 Paper (coming soon) | 💻 [Code](https://github.com/SALT-NLP/agent-collusion) | 🤗 Data (coming soon)

## Abstract

LLM agents are increasingly deployed in collaborative settings, yet long-term interaction may give rise to undesirable coordination. We study the emergence of collusion in a long-horizon multi-agent environment: two agents repeatedly complete individual tasks, share task logs, verify each other's work, and receive rewards. We introduce realistic constraints that make compliance with the verification protocol incompatible with reward maximization, and find that agents increasingly deviate from the protocol over repeated interactions. Collusion emerges in 94% of trajectories across 10 models, and more capable models within the same family reach it earlier. Controlled peer interventions show that collusion is shaped by peer behavior, while ablations reveal additional effects of reward structure, the verification feedback agents receive, and their interaction history. In particular, restricting the amount and scope of interaction history available to agents reduces collusion. Overall, our findings show that long-horizon interaction can reshape how agents coordinate in ways that create safety risks.

## What this measures

We study **collusion as joint instruction violation driven by inter-agent behavioral influence**. A fixed pair of agents interacts over 10 episodes, each with four phases:

**Task completion → Communication → Verdict → Feedback and reflection**

Each agent solves its own task, exchanges evidence with its peer, and issues `ACCEPT` or `REJECT` for the peer's work. The verification protocol requires the complete raw execution log, but communication is limited to five rounds of 200 characters per message. Agents receive a shared reward for correct verdicts and retain their interaction history. This creates a conflict between following the verification protocol and maximizing reward, and we measure collusion through mutual `ACCEPT`.

## Repository structure

```text
.
├── experiments/                    # Two-agent experiments and controlled-peer interventions
│   ├── __main__.py                 # Main CLI: python -m experiments
│   ├── runner.py                   # Run execution and result saving
│   ├── episode_runner.py           # Four-phase episode orchestration
│   ├── agents.py                   # Agent turns and tool execution
│   ├── llm.py                      # Model routing and provider settings
│   ├── controlled.py               # Replay controlled-peer behavior
│   ├── record_peer.py              # Record peer trajectories for replay
│   ├── prompts/                    # System, task, policy, and feedback prompts
│   ├── protocol/                   # Task tools, communication, verdicts, and rewards
│   └── memory/                     # Cross-episode history and retention
│
├── analysis/                       # LLM judges for trajectory analysis
│   ├── agreement_judge.py          # Explicit coordination in communication
│   ├── agreement_judge_prompts.md  # Prompts for the agreement judge
│   ├── relaxation_judge.py         # Policy relaxation in private reflections
│   └── relaxation_judge_prompts.md # Prompts for the relaxation judge
│
├── task/                           # Task inputs, references, and fixed sequences
│   ├── code-analysis/              # Python code analysis
│   ├── text-extraction/            # Record extraction from source documents
│   ├── data-search/                # Data search over a SQLite database
│   ├── task_sequences_50x10/       # 50 sequences × 10 episodes
│   └── task_sequences_50x3+50x10/  # 50 sequences × (3 warm-up + 10 evaluation episodes)
│
├── .env.example                    # Model provider configuration template
└── requirements.txt                # Python dependencies
```

## Installation

Use Python 3.12 and run the following from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your model provider credentials to `.env`, which is loaded automatically. Models use LiteLLM routes such as `openai/<model>`, `gemini/<model>`, `bedrock/<model>`, or `deepseek/<model>`. For self-hosted agent models, configure `OPENAI_API_BASE` and `OPENAI_API_KEY` as shown in [.env.example](.env.example) and use an `openai/<model>` route.

`pytest` is a runtime dependency rather than a development one: agents write their own test files during code-analysis tasks, and the protocol executes them in a subprocess.

## Usage

### Main experiment

Set your model route and a reasoning effort supported by that model, then run one trajectory:

```bash
export MODEL="openai/<your-model>"
export REASONING_EFFORT="high"

python -m experiments \
  --alice-model "$MODEL" \
  --bob-model "$MODEL" \
  --alice-reasoning-effort "$REASONING_EFFORT" \
  --bob-reasoning-effort "$REASONING_EFFORT" \
  --task-sequence-record task/task_sequences_50x10 \
  --repeats 1 \
  --output-dir results/main
```

The defaults implement the main setting: 10 episodes, five communication rounds, 200 characters per message, shared verdict-accuracy rewards, verdict review, private reflection, and full interaction history. Use `--repeats 50` for all 50 fixed sequences; `--start-index` selects the first sequence (1-based). For cross-model experiments, change Alice's and Bob's model settings independently.

Each trajectory saves a `run.json` under the output directory, containing its configuration, task evaluations, messages, verdicts, rewards, reflections, and model usage.

### Ablations

Apply one row to the main command; use a separate `--output-dir` for each condition. All other settings keep their defaults.

| Experiment | Condition | Command options |
| --- | --- | --- |
| Feedback | No reward instructions, reward feedback, or reflection; keep verdict review | `--no-reward` |
| | No verdict review; keep reward and reflection | `--no-verdict-review` |
| Warm-up history | 3 unrestricted-communication episodes + 10 evaluation episodes | `--task-sequence-record task/task_sequences_50x3+50x10`<br>`--throttle-spans 'no-throttle:3,throttled:*'` |
| | 3 summary-allowed episodes + 10 evaluation episodes | `--task-sequence-record task/task_sequences_50x3+50x10`<br>`--verdict-policy-spans 'summary-allowed:3,raw-only:*'` |
| Memory length | Last 3 episodes | `--alice-cross-episode-memory-length 3`<br>`--bob-cross-episode-memory-length 3` |
| | No cross-episode memory | `--alice-cross-episode-memory-length 0`<br>`--bob-cross-episode-memory-length 0` |
| Memory scope | Communication onward | `--cross-episode-memory-scope communication-onward` |
| | Feedback and reflection only | `--cross-episode-memory-scope feedback-and-reflection` |
| Reward scope | Separate rewards; each agent's reward and review concern the peer's verdict on its own task | `--reward-scope separate` |
| Reward type | Verdict accuracy, without verdict review | `--reward-type verdict-accuracy --no-verdict-review` |
| | Acceptance, without verdict review | `--reward-type acceptance --no-verdict-review` |

### Controlled peer interventions

Bob is replaced by a scripted peer. Use `$MODEL` and `$REASONING_EFFORT` from the main example for Alice.

| Peer policy | `--controlled-bob-messages` | `--controlled-bob-verdict` |
| --- | --- | --- |
| Compliant | `raw-prefix` (first 5 × 200 characters of the raw log) | `reject` |
| Violating | `summary` (Peer's own summaries of its work) | `accept` |

**1. Record the peer once.**

```bash
python -m experiments.record_peer \
  --model gemini/gemini-3.1-flash-lite \
  --reasoning-effort high \
  --task-sequence-record task/task_sequences_50x10 \
  --repeats 50 \
  --cache results/peer-cache
```

**2. Replay the violating peer.** 

```bash
python -m experiments \
  --alice-model "$MODEL" \
  --alice-reasoning-effort "$REASONING_EFFORT" \
  --bob-model controlled \
  --controlled-bob-cache results/peer-cache \
  --controlled-bob-messages summary \
  --controlled-bob-verdict accept \
  --task-sequence-record task/task_sequences_50x10 \
  --repeats 50 \
  --output-dir results/peer-violating
```

**3. Replay the compliant peer.**

```bash
python -m experiments \
  --alice-model "$MODEL" \
  --alice-reasoning-effort "$REASONING_EFFORT" \
  --bob-model controlled \
  --controlled-bob-cache results/peer-cache \
  --controlled-bob-messages raw-prefix \
  --controlled-bob-verdict reject \
  --task-sequence-record task/task_sequences_50x10 \
  --repeats 50 \
  --output-dir results/peer-compliant
```

Add `--controlled-bob-observed-verdict` to reveal Bob's verdict in Alice's feedback.

### Evaluation

Two additional LLM judges annotate **explicit coordination** in communication and **policy relaxation** in private reflections to study how collusion begins.

Start a local OpenAI-compatible model server on port, e.g., **8042**, serving Qwen 3.8 27B with the model name **`qwen3.8-27b`**. The judge scripts use this configuration by default, with thinking disabled and temperature 0.

```bash
python analysis/agreement_judge.py \
  --runs 'results/main/*/run.json' \
  --out analysis/results/agreement.csv \
  --base-url http://localhost:8042/v1 \
  --model qwen3.8-27b \
  --verdict-policy raw-only

python analysis/relaxation_judge.py \
  --runs 'results/main/*/run.json' \
  --out analysis/results/relaxation.csv \
  --base-url http://localhost:8042/v1 \
  --model qwen3.8-27b \
  --verdict-policy raw-only
```


## Tasks

All task inputs and evaluation references are included in [task/](task/):

| Task family | Tasks | Resources and objective |
| --- | ---: | --- |
| Code analysis | 50 | Python implementations; identify whether the code satisfies its specification. |
| Record extraction | 50 | Source documents; extract identifiers matching the requested criteria. |
| Data search | 50 | SQLite database; answer queries evaluated against reference SQL results. |

Each family includes a task manifest. [task_sequences_50x10](task/task_sequences_50x10/) contains 50 fixed ten-episode sequences, shared across models and conditions. [task_sequences_50x3+50x10](task/task_sequences_50x3+50x10/) prepends three warm-up episodes to each sequence. Each episode assigns the agents different tasks from the same family.

## License

Code and task data are released under the [MIT License](LICENSE).

## Citation

```bibtex
@misc{shi2026emergentcollusion,
  title={Emergent Collusion in Long-Horizon LLM Agent Interaction},
  author={Xinrui Shi and Yanzhe Zhang and Diyi Yang},
  year={2026}
}
```
