# Divide21X — A Deterministic Symbolic Reasoning Benchmark for Large Language Models (LLMs)

Divide21X is a daily, automated benchmark that evaluates the symbolic mathematical reasoning of Large Language Models (LLMs).
Every day, a challenge is generated, sent to multiple LLM APIs, and graded against a deterministic simulator.
The goal is to measure true reasoning ability, not pattern recognition, memorization, or heuristics.

It introduces a form of evaluation that many current benchmarks do not capture: *deterministic symbolic reasoning under explicitly defined mathematical rules*. It forces models to operate inside a formal, fully deterministic, state-transition environment, where every field has one and only one mathematically correct value.

It is a pure reasoning benchmark, because it isolates:

- No language

- No world knowledge

- No stylistic expectations

- No pattern shortcuts

- No ambiguous goals

Just mathematical, symbolic, rule-based cognition.


# What does Divide21X Measure?

### 1. True Symbolic Mathematical Reasoning

LLMs must apply formally defined operations to transform an initial environment state.
This involves:

- Manipulating integers and nested structures

- Executing mathematically defined transitions

- Ensuring internal consistency

- Avoiding hallucinations

- Models must "simulate" reasoning, not approximate it.


### 2. Deterministic Environment-State Transitions

Each daily task provides:

- a solved example (`example_1`)

- a second solved example (`example_2`)

- an unsolved `challenge`

Each example and challenge includes:

- initial state (`z`)

- action (`a`)

- final state (`o`) — missing in the challenge

Models must compute the correct final state, following the rules of the Divide21 environment.

This tests:

- Multi-step logical inference

- Rule-following under constraints

- Ability to generalize from examples

- Precision in symbolic manipulation


### 3. Exact, Verifiable Outputs

Every predicted final state is compared field-by-field to the ground-truth computed by the Divide21 simulator.
Scores are deterministic, numeric, and interpretable.

Scoring measures:

- Structural and Value correctness

- Action-rule compliance

- Error distance for each field


In addition, Divide21X:

- Requires mathematical exactness

	- The output must be a precise numerical structure, not "something that fits the pattern."

- Has zero ambiguity

	- Every challenge has exactly one correct result.

- Uses stateful multi-step symbolic logic

	- Models must simulate reasoning chains, not just identify patterns.

- Tests internal model consistency, not creativity

	- The model must not hallucinate or approximate.

- Produces numerical, falsifiable outputs

	- Which makes statistical scoring extremely precise over time.


Essentially, Divide21X is a stress test for true reasoning. LLMs that perform well demonstrate:

- Arithmetic precision

- Ability to generalize symbolic rules

- Systematic logical consistency

- Reliability in deterministic simulation tasks

- Reduced hallucination tendencies


# Challenge Format
```
{
  "example_1": {
    "z": { ... initial state ... },
    "a": { ... action ... },
    "o": { ... final state ... }
  },
  "example_2": {
    "z": { ... },
    "a": { ... },
    "o": { ... }
  },
  "challenge": {
    "z": { ... },
    "a": { ... },
    "o": missing --> LLM MUST FILL THIS IN
  }
}
```

# Daily Pipeline Overview

The benchmark runs once per day via GitHub Actions:

### 1. Generate Daily Challenge

A JSON challenge is created at:

`divide21x/challenges/<year-month>/<day>.json`


### 2. Query All LLM APIs

Each provider receives the challenge and must output the missing final state (o).

Supported APIs include:

- OpenAI (GPT-4o and o1)

- Anthropic

- Google (Gemini)

- Mistral

- HuggingFace

- XAI

- Others as configured


### 3. Evaluate Model Outputs

The Divide21 simulator computes the real final state.
Each LLM’s prediction is compared numerically and structurally.


### 4. Store Results & Scores

Stored at:

`divide21x/results/<year-month>/<day>.json`


### 5. Commit to Repository

The entire day's update is automatically committed to GitHub.
This creates a public, chronological leaderboard of reasoning performance.


## Cite This Project

If you use Divide21X in your research, projects, or publications, please cite it as:

Jacinto Jeje Matamba Quimua (2025). Divide21X: A Deterministic Symbolic Reasoning Benchmark for Large Language Models (LLMs). GitHub repository: https://github.com/jaci-hub/divide21X


### BibTeX

```bibtex
@misc{divide21x2025,
  author       = {Jacinto Jeje Matamba Quimua},
  title        = {Divide21X: A Deterministic Symbolic Reasoning Benchmark for Large Language Models (LLMs)},
  year         = 2025,
  howpublished = {\url{https://github.com/jaci-hub/divide21X}},
}
```
