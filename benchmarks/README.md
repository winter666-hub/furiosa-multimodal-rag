# Benchmark Report

This document publishes the core results produced by the local benchmark CSVs without committing
the raw runtime artifacts. The evaluation datasets and benchmark code remain versioned, while
`benchmarks/*_results.csv` remains ignored because those files contain environment-specific run
output.

## Evaluation Scope

The end-to-end set contains 15 questions about *Attention Is All You Need*:

| Category | Questions |
|---|---:|
| Text | 5 |
| Explicit visual | 5 |
| Implicit visual | 5 |
| **Total** | **15** |

The router holdout contains 60 questions and is evaluated separately from retrieval, vision, and
answer generation. This separation prevents router latency from being confused with E2E latency.

Versioned inputs:

- [`e2e_eval_small.jsonl`](e2e_eval_small.jsonl)
- [`router_holdout_eval.jsonl`](router_holdout_eval.jsonl)
- [`router_eval.jsonl`](router_eval.jsonl)
- [`answer_quality_reference.jsonl`](answer_quality_reference.jsonl)

## End-to-End Results

All three strategies completed all 15 E2E questions without a recorded pipeline error.

| Strategy | Vision Calls | Avg E2E Latency | Text-Query Latency |
|---|---:|---:|---:|
| Always Vision | 15 | 13.053 s | 14.076 s |
| Pure LLM Router | 9 | 10.490 s | 6.600 s |
| Adaptive Router | 9 | 10.479 s | 6.560 s |

Relative to Always Vision, Adaptive routing produced:

| Metric | Change |
|---|---:|
| Vision calls | -40.0% |
| Average E2E latency | -19.7% |
| Text-query latency | -53.4% |

These values are arithmetic means from the current local CSVs. They describe this evaluation run,
not a general service-level latency guarantee; model serving load and network conditions can alter
absolute timing.

## Adaptive Routing on the E2E Set

Text questions are expected to route to `TEXT_ONLY`; explicit- and implicit-visual questions are
expected to route to `VISUAL_REQUIRED`.

| Metric | Result |
|---|---:|
| Correct routes | 14 / 15 |
| Accuracy | 93.3% |
| Visual precision | 100.0% |
| Visual recall | 90.0% |
| False positives | 0 |
| False negatives | 1 |

The only false negative was `E_A02`, an implicit-visual question asking for the sequence from input
embedding to the first encoder attention operation. It routed to `TEXT_ONLY`.

## Router-Only Holdout

The 60-question holdout measures routing in isolation.

| Metric | Pure LLM Router | Adaptive Router |
|---|---:|---:|
| Correct | 59 / 60 | 59 / 60 |
| Accuracy | 98.3% | 98.3% |
| Visual precision | 97.6% | 97.6% |
| Visual recall | 100.0% | 100.0% |
| False positives | 1 | 1 |
| False negatives | 0 | 0 |
| Avg routing latency | 281.34 ms | 135.97 ms |

Adaptive routing reduced average router-only latency by **51.67%** in this run while preserving the
same holdout decisions.

## Preliminary Answer Quality

Answer quality is **preliminary**, not a complete final evaluation. The local result file contains
45 strategy-question rows: 44 successful judge results and one judge JSON parsing error.

| Strategy | Successfully Judged | Mean Quality Score (0-10) |
|---|---:|---:|
| Always Vision | 14 / 15 | 8.86 |
| Pure LLM Router | 15 / 15 | 8.80 |
| Adaptive Router | 15 / 15 | 8.80 |

Because Always Vision has one missing judgment, its mean is not directly comparable as a complete
15-question strategy average. For the routing false negative `E_A02`, the recorded scores were:

| Strategy | `E_A02` Quality Score |
|---|---:|
| Always Vision | 10 |
| Pure LLM Router | 7 |
| Adaptive Router | 7 |

This case suggests that an implicit-visual routing miss can propagate into answer-quality loss, but
the small dataset and incomplete judge output do not support a broad statistical claim.

## Artifact Policy

The raw files used for this report are intentionally local and ignored by Git:

```text
benchmarks/e2e_always_vision_results.csv
benchmarks/e2e_llm_results.csv
benchmarks/e2e_adaptive_results.csv
benchmarks/router_llm_holdout_results.csv
benchmarks/router_adaptive_holdout_results.csv
benchmarks/answer_quality_results.csv
```

They can be regenerated with the benchmark CLIs in `src/furiosa_rag/cli/`. This Markdown report is
the stable, reviewable GitHub artifact; raw timing CSVs remain excluded by
`benchmarks/*_results.csv` in `.gitignore`.
