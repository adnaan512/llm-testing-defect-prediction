# AI-Driven Software Quality Assurance  Research Portfolio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Research](https://img.shields.io/badge/Research-SE%20%7C%20AI4SE-green)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-orange?logo=github-actions)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Affiliation](https://img.shields.io/badge/NUST%20Pakistan-BS%20CS-red)

**Adnan Hassnain** · BS CS, NUST Pakistan  
Research interests: LLM for Software Engineering · AI-Driven Testing · Automated Fault Localization

</div>

---

## Overview

This repository contains three interconnected research projects exploring the application of **Artificial Intelligence and Large Language Models to Software Quality Assurance**. The work is directly inspired by the research programme of [Prof. Chin-Yu Huang's SE Lab at NTHU](https://nthu-se.github.io/), whose publications on LLM-based artifact generation, deep-learning defect prediction, and coverage-guided fault localization motivate each project.

All implementations are in **pure Python** with minimal external dependencies — maximising algorithmic transparency and reproducibility.

---

## Research Projects

| Project | Topic | Key Technique | Research Alignment |
|---|---|---|---|
| [Project 1](#-project-1--llm-based-automated-test-case-generator) | LLM Test Generation | BVA · EP · Fault-Based prompting | Huang et al. (2024)  QRS |
| [Project 2](#-project-2--software-defect-prediction) | Defect Prediction | MLP · CNN · Self-Attention | Huang et al. (2026) — Attention SDP |
| [Project 3](#-project-3--automated-api-fault-localization) | Fault Localization | Tarantula · Ochiai · ML Ensemble | Huang et al. (2025)  Coverage DL |

---

##  Project 1 LLM-Based Automated Test Case Generator

> `project1_llm_testgen/`

**Research Question:** Can structured prompts encoding BVA, EP, and Fault-Based Testing techniques enable LLMs to generate test suites with ≥80% line coverage and measurable fault-detection capability?

**Pipeline:**
```
Source Code
  │  ast.iter_child_nodes() — top-level function extraction
  ▼
[CodeParser]         Extracts: name · args · docstring · cyclomatic complexity
  │  Anthropic claude-sonnet-4-6 · Retry with exponential back-off
  ▼
[LLMTestGenerator]   System prompt + BVA/EP/FBT user prompt
  │  pytest-cov · isolated subprocess · regex output parsing
  ▼
[TestExecutor]       Real line coverage · pass/fail · fault detection
  │
  ▼
[JSON Report]        Timestamped · per-function metrics · model metadata
```

**Key contributions:**
- Structured prompt engineering with explicit testing-technique labels
- Real `pytest-cov` line coverage (not test pass-rate as proxy)
- AST-based cyclomatic complexity estimation for prompt context
- Fault-injection benchmark (`binary_search` off-by-one) for ground-truth evaluation
- Production-ready CLI with `--dry-run`, `--max-functions`, retry logic

 [Full README](project1_llm_testgen/README.md)

---

##  Project 2  Software Defect Prediction

> `project2_defect_prediction/`

**Research Question:** Does a self-attention mechanism over CK metric vectors outperform CNN and MLP baselines for software defect prediction on NASA PROMISE-style data?

**Models implemented from scratch (pure Python):**

| Model | Inspiration | Key Mechanism |
|---|---|---|
| MLP Baseline | Traditional SDP | Dense layers + ReLU + binary cross-entropy |
| CNN Predictor | Huang et al. (2024) | 1-D convolution (k=2,3) + global max-pooling |
| Attention Predictor | Huang et al. (2026) | Learned softmax attention over metric vector |

**Evaluation:** Stratified 5-fold CV · Accuracy · Precision · Recall (PD) · F1 · **AUC-ROC** · PF

 [Full README](project2_defect_prediction/README.md)

---

##  Project 3 Automated API Fault Localization

> `project3_fault_localization/`

**Research Question:** Does an ML ensemble combining spectrum-based scores (Tarantula + Ochiai) with execution-trace features improve fault-localization rank over either algorithm alone?

**Algorithms implemented:**
- **Tarantula** (Jones & Harrold, 2005)  spectrum-based FL
- **Ochiai** (Abreu et al., 2007)  cosine-similarity-inspired FL  
- **ML Ensemble** — weighted combination with response-time ratio, failure density, and isolation score

**Context:** Inspired by QA experience tracing API failures across microservices at MyTechPassport, automating a previously manual root-cause analysis process.

 [Full README](project3_fault_localization/README.md)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adnaan512/llm-testing-defect-prediction
cd llm-testing-defect-prediction

# Project 2 :no API key needed, runs immediately
cd project2_defect_prediction
python defect_predictor.py

# Project 3: no API key needed, runs immediately
cd ../project3_fault_localization
python fault_localizer.py

# Project 1: requires Anthropic API key
cd ../project1_llm_testgen
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python test_generator.py                  # run on built-in benchmark
python test_generator.py --dry-run        # parse only, no API call
python test_generator.py --input myfile.py
```

---

## Repository Structure

```
llm-testing-defect-prediction/
├── README.md                          ← You are here
├── CITATION.cff                       ← Academic citation file
├── .github/
│   └── workflows/
│       └── ci.yml                     ← GitHub Actions CI
│
├── project1_llm_testgen/
│   ├── test_generator.py              ← Full pipeline (parser, LLM, executor)
│   ├── requirements.txt
│   └── README.md
│
├── project2_defect_prediction/
│   ├── defect_predictor.py            ← MLP · CNN · Attention (pure Python)
│   ├── requirements.txt
│   └── README.md
│
└── project3_fault_localization/
    ├── fault_localizer.py             ← Tarantula · Ochiai · ML Ensemble
    ├── requirements.txt
    └── README.md
```

---

## Research References

All three projects are grounded in recent publications from Prof. Chin-Yu Huang's Software Engineering Lab at NTHU:

1. Huang, C.-Y. et al. (2024). *Automated software artifact generation using large language models*. QRS 2024.
2. Huang, C.-Y. et al. (2026). *Bidirectional program dependency-guided attention for software defect prediction*. NTHU SE Lab.
3. Huang, C.-Y. et al. (2025). *Deep learning for fault localization with coverage data reduction*. NTHU SE Lab.

Additional foundational references:
- Jones, J.A. & Harrold, M.J. (2005). *Empirical evaluation of Tarantula*. ASE 2005.
- Abreu, R. et al. (2007). *On the accuracy of spectrum-based fault localization*. TAIC PART 2007.
- Menzies, T. et al. (2007). *Data mining static code attributes to learn defect predictors*. IEEE TSE.
- Chidamber, S.R. & Kemerer, C.F. (1994). *A metrics suite for object-oriented design*. IEEE TSE.
- Schafer, M. et al. (2023). *An empirical evaluation of using LLMs for automated unit test generation*. IEEE TSE 2024.

---

## License

MIT License  see [LICENSE](LICENSE) for details.

---

<div align="center">



</div>
