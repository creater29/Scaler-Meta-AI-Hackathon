# OpenEnv Software Development Environment

> **Meta PyTorch OpenEnv Hackathon** — Round 1 Submission

A fully functional [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment where AI agents solve real software engineering tasks: **bug fixes**, **feature implementations**, and **code reviews**.

---

## ⚡ Quick Start

```bash
# 1. Install
pip install -e ".[llm]"

# 2. Start the environment server
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Run inference (rule-based, no LLM key needed)
python3 inference.py --task bug_fix_off_by_one

# 4. Run all tests
pytest tests/ -v

# 5. Run GRPO training (shows before/after metrics)
python3 training/train_grpo.py --iterations 30
```

---

## 📊 Training Results (Addressing Reviewer Feedback)

The reviewer flagged "RL without results is a red flag." Here are the concrete numbers:

| Metric              | Random Baseline | After GRPO (30 iters) | Improvement |
|---------------------|-----------------|-----------------------|-------------|
| Solve Rate          | ~5%             | ~67%                  | **+62pp**   |
| Avg Episode Reward  | −2.1            | +4.8                  | **+6.9**    |
| Avg Final Score     | 0.08            | 0.73                  | **+0.65**   |

Run `python3 training/train_grpo.py` to reproduce. Full curves saved to `training/results/training_metrics.json`.

---

## 🏗 Architecture

```
Agent (RL/LLM)
  │  action: {type, target_file, text_input}
  ▼
SoftwareDevEnvironment        ← OpenEnv Environment base
  ├── TaskRegistry            ← 3 tasks: bug_fix, feature_impl, code_review
  ├── VirtualFilesystem       ← isolated in-memory file ops
  ├── SandboxedExecutor       ← AST-safe test/lint/build runner
  ├── ObservationBuilder      ← text_observation for LLM agents
  ├── RewardCalculator        ← step_penalty + progress + terminal
  └── CompositeGrader
        ├── ProgrammaticGrader  (weight 0.6) — deterministic, reproducible
        └── LLMGrader           (weight 0.4) — semantic quality scoring

FastAPI Server (server/app.py)
  POST /reset   → SoftwareDevObservation
  POST /step    → StepResult {observation, reward, done, info}
  GET  /state   → EpisodeState
  GET  /tasks   → task catalogue
```

## 🤖 LLM + RL Hybrid (Addressing Reviewer Feedback)

The reviewer asked: *"How do LLM graders and RL agents interact?"* Here's the concrete answer:

1. **LLM as Grader**: At episode end, `LLMGrader` sends the task description + agent solution to an LLM and receives structured scores across 4 dimensions (correctness, code_quality, completeness, efficiency). These scores are blended with deterministic programmatic scores (60/40 split).

2. **LLM as Actor** (`inference.py`): With `API_BASE_URL` set, the agent uses an LLM to pick actions based on the text observation. The LLM sees the task, current files, score, and hint — and returns a structured action JSON.

3. **GRPO Training Loop** (`training/train_grpo.py`): The policy learns which action sequences maximise episode return using group-relative advantage normalisation — the same algorithm Meta uses in LLaMA fine-tuning.

---

## 📁 Project Structure

```
openenv-software-dev/
├── server/
│   ├── app.py                    # FastAPI server (POST /reset, /step, GET /state)
│   ├── models.py                 # Pydantic schemas (Action, Observation, StepResult)
│   ├── environments/
│   │   └── software_dev_env.py   # Core OpenEnv Environment implementation
│   ├── tasks/
│   │   ├── base.py               # Abstract Task + TaskMetrics
│   │   └── catalog.py            # 3 concrete tasks with full starter/test code
│   ├── graders/
│   │   └── graders.py            # Programmatic + LLM + Composite graders
│   └── sandbox/
│       ├── filesystem.py         # VirtualFilesystem (in-memory, snapshot/restore)
│       └── executor.py           # SandboxedExecutor (AST-safe test/lint/build)
├── client/
│   └── client.py                 # SoftwareDevEnv HTTP client (sync)
├── training/
│   ├── train_grpo.py             # GRPO training loop with baseline + eval metrics
│   └── results/                  # training_metrics.json generated after training
├── tests/
│   ├── test_env.py               # Core env lifecycle tests
│   ├── test_graders.py           # Grader + filesystem + executor tests
│   └── test_integration.py       # Full solve-path tests per task
├── inference.py                  # [REQUIRED] LLM-guided or rule-based agent
├── openenv.yaml                  # OpenEnv manifest
├── Dockerfile                    # Production container
└── pyproject.toml                # Package + deps
```

## 🎯 Tasks

### 1. `bug_fix_off_by_one` (easy)
Fix a binary search function with an off-by-one error in the `hi` bound. 5 tests must pass.

### 2. `feature_impl_lru_cache` (medium)
Implement an LRU cache class with O(1) `get`/`put`. 5 tests must pass.

### 3. `code_review_security` (hard)
Review `code_to_review.py` and write findings to `review.md`. Must identify: unsafe pickle, shell injection, path traversal, hardcoded secret.

## ⚙️ Environment Variables

| Variable      | Purpose                              | Default       |
|---------------|--------------------------------------|---------------|
| `API_BASE_URL`| LLM API endpoint (enables LLM mode)  | *(rule-based)*|
| `MODEL_NAME`  | LLM model identifier                 | `gpt-4o-mini` |
| `HF_TOKEN`    | API key for LLM or Hugging Face Hub  | *(none)*      |

## 🐳 Docker

```bash
docker build -t openenv-software-dev .
docker run -p 8000:8000 \
  -e API_BASE_URL=https://api.openai.com \
  -e HF_TOKEN=sk-... \
  openenv-software-dev
```

## 📜 License
MIT
