"""
LLM-Based Automated Test Case Generator
========================================
Research Area : Software Testing Automation | LLM for Software Engineering
Inspired by   : Chin-Yu Huang et al. (2024) — "Automated software artifact
                generation using large language models," QRS 2024, NTHU SE Lab.

Abstract
--------
This tool demonstrates an end-to-end pipeline for automated unit-test
generation using large language models (LLMs). Given a Python source file,
the system:

  1. Parses function signatures and docstrings via Python's ``ast`` module.
  2. Constructs structured prompts embedding established software-testing
     techniques: Boundary Value Analysis (BVA), Equivalence Partitioning (EP),
     and Fault-Based Testing (FBT).
  3. Calls the Anthropic Claude API to generate pytest-compatible test suites.
  4. Executes the generated tests in an isolated subprocess with real line
     coverage measurement via ``pytest-cov``.
  5. Produces a structured JSON report with per-function metrics including
     pass/fail counts, line coverage percentage, and fault-detection status.

Research Questions Addressed
-----------------------------
  RQ1: Can LLMs produce test suites achieving ≥80% line coverage on
       moderately complex Python functions?
  RQ2: Do structured prompts (BVA + EP + FBT) outperform naive "generate
       tests" prompts in fault-detection rate?
  RQ3: How does generated-test quality correlate with function cyclomatic
       complexity?

Limitations
-----------
  - Coverage measurement depends on ``pytest-cov`` being installed.
  - API latency and cost scale linearly with the number of functions.
  - The synthetic dataset contains intentionally introduced bugs to provide
    ground truth for fault-detection evaluation.

Author  : Adnan Hassnain
Affil.  : BS CS, NUST Pakistan
Contact : adnanhassnain39@gmail.com
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool = False) -> logging.Logger:
    """Set up structured console logging with optional verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionInfo:
    """Metadata extracted from a single Python function via AST analysis."""
    name: str
    args: List[str]
    docstring: str
    source: str
    return_annotation: str
    lineno: int                   # Source line number (for traceability)
    cyclomatic_complexity: int    # McCabe complexity estimate


@dataclass
class TestResult:
    """
    Aggregated results for tests generated for a single function.

    Attributes
    ----------
    function_name        : Name of the function under test.
    generated_tests      : Raw test code produced by the LLM.
    test_names           : Names of individual test functions detected.
    passed               : Number of tests that passed.
    failed               : Number of tests that failed.
    errors               : Number of tests that produced errors (not failures).
    line_coverage_pct    : Real line coverage from pytest-cov (0–100).
    fault_detected       : True if any test exposed a defect.
    llm_model            : Model identifier used for generation.
    generation_time_ms   : Wall-clock time spent on the API call.
    execution_time_ms    : Wall-clock time spent running the test suite.
    prompt_chars         : Length of the prompt sent (proxy for token usage).
    timestamp_utc        : ISO-8601 UTC timestamp of this run.
    """
    function_name: str
    generated_tests: str
    test_names: List[str]
    passed: int
    failed: int
    errors: int
    line_coverage_pct: float
    fault_detected: bool
    llm_model: str
    generation_time_ms: float
    execution_time_ms: float
    prompt_chars: int
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Code Parser
# ─────────────────────────────────────────────────────────────────────────────

class CodeParser:
    """
    Extracts function metadata from Python source using the ``ast`` module.

    Design Notes
    ------------
    We iterate over *direct* children of the module node (not ``ast.walk``),
    so inner/nested functions are not mistakenly treated as top-level
    functions.  This avoids over-counting and ensures tests are generated
    only for the intended public API.
    """

    # Simple heuristic: count branch points as a proxy for cyclomatic complexity
    _BRANCH_NODES = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )

    def parse_file(self, filepath: str) -> List[FunctionInfo]:
        """Parse a source file and return metadata for each top-level function."""
        source = Path(filepath).read_text(encoding="utf-8")
        return self._extract_functions(source)

    def parse_code_string(self, code: str) -> List[FunctionInfo]:
        """Parse functions directly from a code string."""
        return self._extract_functions(code)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_functions(self, source: str) -> List[FunctionInfo]:
        tree = ast.parse(source)
        functions: List[FunctionInfo] = []

        for node in ast.iter_child_nodes(tree):   # top-level only
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._build_info(node, source))

        return functions

    def _build_info(
        self, node: ast.FunctionDef, source: str
    ) -> FunctionInfo:
        args = [arg.arg for arg in node.args.args]
        docstring = ast.get_docstring(node) or ""
        return_ann = ast.unparse(node.returns) if node.returns else ""
        func_source = ast.get_source_segment(source, node) or ""
        complexity = self._cyclomatic_complexity(node)

        return FunctionInfo(
            name=node.name,
            args=args,
            docstring=docstring,
            source=func_source,
            return_annotation=return_ann,
            lineno=node.lineno,
            cyclomatic_complexity=complexity,
        )

    def _cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Estimate McCabe cyclomatic complexity (branches + 1)."""
        branches = sum(
            1 for child in ast.walk(node)
            if isinstance(child, self._BRANCH_NODES)
        )
        return branches + 1


# ─────────────────────────────────────────────────────────────────────────────
# LLM Test Generator
# ─────────────────────────────────────────────────────────────────────────────

class LLMTestGenerator:
    """
    Generates unit tests using the Anthropic Claude API.

    Prompt Engineering Strategy
    ---------------------------
    The system prompt establishes the LLM as a software-testing expert, then
    the user prompt provides:
      (a) the function's source code and docstring,
      (b) the full module context (so the LLM understands imports/helpers),
      (c) explicit instructions to apply BVA, EP, and FBT — three canonical
          testing techniques from software-engineering curricula.

    This structured prompting strategy is informed by research showing that
    technique-specific prompts yield higher fault-detection rates than generic
    "write tests" prompts (Schafer et al., 2023; Yuan et al., 2023).

    Retry Policy
    ------------
    Transient API errors (rate limits, timeouts) are retried with exponential
    back-off (max ``max_retries`` attempts).
    """

    SYSTEM_PROMPT = (
        "You are an expert software test engineer with deep knowledge of "
        "pytest, Python testing best practices, and software-testing theory. "
        "You generate only executable pytest test code — no prose, no markdown "
        "fences, no explanation text whatsoever."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.api_url = "https://api.anthropic.com/v1/messages"

    def generate_tests(
        self, func_info: FunctionInfo, module_code: str
    ) -> Tuple[str, float, int]:
        """
        Generate pytest tests for *func_info*.

        Returns
        -------
        (test_code, generation_time_ms, prompt_chars)
        """
        prompt = self._build_prompt(func_info, module_code)
        t0 = time.perf_counter()
        test_code = self._call_api_with_retry(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Strip accidental markdown fences if the model wraps output
        test_code = self._strip_fences(test_code)
        return test_code, round(elapsed_ms, 1), len(prompt)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, func: FunctionInfo, module_code: str) -> str:
        return f"""Generate comprehensive pytest unit tests for the Python function below.

Apply ALL of the following software-testing techniques (label each test with a comment):
  1. **Boundary Value Analysis (BVA)** — test minimum, maximum, and boundary±1 values
  2. **Equivalence Partitioning (EP)** — one representative test per valid/invalid class
  3. **Fault-Based Testing (FBT)** — design tests specifically likely to catch:
       off-by-one errors, incorrect operator precedence, wrong comparison (<= vs <),
       None/empty handling, type errors, and incorrect exception types
  4. **Happy Path** — at least one test confirming correct behaviour for typical input
  5. **Error/Exception Path** — test that the correct exception is raised for invalid input

Function to test (cyclomatic complexity: {func.cyclomatic_complexity}):
```python
{func.source}
```

Full module context (for any helpers or constants the function depends on):
```python
{module_code}
```

Hard requirements:
- Use pytest; do NOT use unittest.
- Each test function name must follow the pattern: test_<function>_<what_it_tests>
- Place a single-line comment ABOVE each test naming the technique: # BVA | EP | FBT | Happy | Error
- Generate at least 8 test cases total.
- Import only from the standard library and pytest — the module under test
  is already importable as `module_under_test`.
- Do NOT wrap output in markdown code fences.
- Return ONLY valid Python — zero prose."""

    def _call_api_with_retry(self, prompt: str) -> str:
        """Call the Anthropic Messages API with exponential back-off retry."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=60
                )
                resp.raise_for_status()
                return resp.json()["content"][0]["text"]
            except requests.exceptions.Timeout:
                logger.warning("API timeout (attempt %d/%d)", attempt, self.max_retries)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status in (429, 529):   # rate-limit / overload
                    logger.warning(
                        "Rate-limited by API (attempt %d/%d). Waiting…",
                        attempt, self.max_retries,
                    )
                else:
                    raise   # Non-retriable HTTP error
            except requests.exceptions.RequestException as exc:
                logger.warning("Request error (attempt %d/%d): %s", attempt, self.max_retries, exc)

            if attempt < self.max_retries:
                wait = 2 ** attempt   # 2s, 4s, 8s …
                logger.debug("Retrying in %ds…", wait)
                time.sleep(wait)

        raise RuntimeError(
            f"Anthropic API call failed after {self.max_retries} attempts."
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```python … ``` fences the model may emit despite instructions."""
        text = re.sub(r"^```(?:python)?\n?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text.strip())
        return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Test Executor & Coverage Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutor:
    """
    Executes generated tests in an isolated subprocess and measures real
    line coverage via ``pytest-cov``.

    Isolation Strategy
    ------------------
    Each run uses a fresh ``tempfile.TemporaryDirectory``.  The module under
    test is written as ``module_under_test.py`` and the generated test file
    imports from it.  This prevents any state leakage between runs and
    mirrors real CI environments.

    Coverage Measurement
    --------------------
    We invoke ``pytest --cov=module_under_test --cov-report=json`` to obtain
    real line-level coverage from ``coverage.py`` (via ``pytest-cov``).
    If ``pytest-cov`` is not installed, we gracefully fall back to a
    warning and report ``-1.0`` for coverage.
    """

    # Regex patterns for parsing pytest summary line, e.g.
    # "3 passed, 1 failed, 1 error in 0.42s"
    _RE_PASSED = re.compile(r"(\d+)\s+passed")
    _RE_FAILED = re.compile(r"(\d+)\s+failed")
    _RE_ERROR = re.compile(r"(\d+)\s+error")
    _RE_TESTS = re.compile(r"def\s+(test_\w+)\s*\(", re.MULTILINE)

    def run(
        self, source_code: str, test_code: str, func_name: str
    ) -> Tuple[TestResult, float]:
        """
        Run the generated test suite and return a partial ``TestResult``.

        The caller fills in LLM-specific fields (model, generation_time, etc.).
        Returns (partial_result, execution_time_ms).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "module_under_test.py"
            test_path = Path(tmpdir) / "test_generated.py"
            cov_json = Path(tmpdir) / "coverage.json"

            src_path.write_text(source_code, encoding="utf-8")

            full_test = (
                "from module_under_test import *\n"
                "import pytest\n\n"
                + test_code
            )
            test_path.write_text(full_test, encoding="utf-8")

            t0 = time.perf_counter()
            result = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(test_path),
                    "-v", "--tb=short",
                    f"--cov={src_path.stem}",
                    "--cov-report=json",
                    f"--cov-report=json:{cov_json}",
                    "--no-header",
                    "-q",
                ],
                capture_output=True,
                text=True,
                cwd=tmpdir,
                timeout=120,
            )
            exec_ms = (time.perf_counter() - t0) * 1000

            logger.debug("pytest stdout:\n%s", result.stdout[-2000:])
            if result.stderr:
                logger.debug("pytest stderr:\n%s", result.stderr[-1000:])

            passed, failed, errors = self._parse_output(result.stdout)
            coverage = self._read_coverage(cov_json)
            test_names = self._RE_TESTS.findall(test_code)

            return (
                TestResult(
                    function_name=func_name,
                    generated_tests=test_code,
                    test_names=test_names,
                    passed=passed,
                    failed=failed,
                    errors=errors,
                    line_coverage_pct=coverage,
                    fault_detected=(failed > 0 or errors > 0),
                    llm_model="",            # filled by pipeline
                    generation_time_ms=0.0,  # filled by pipeline
                    execution_time_ms=round(exec_ms, 1),
                    prompt_chars=0,          # filled by pipeline
                ),
                round(exec_ms, 1),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_output(self, output: str) -> Tuple[int, int, int]:
        """
        Parse pytest's summary line using regular expressions.

        Example lines handled:
          ``5 passed in 0.12s``
          ``3 passed, 2 failed, 1 error in 0.45s``
          ``no tests ran``
        """
        passed = int(m.group(1)) if (m := self._RE_PASSED.search(output)) else 0
        failed = int(m.group(1)) if (m := self._RE_FAILED.search(output)) else 0
        errors = int(m.group(1)) if (m := self._RE_ERROR.search(output)) else 0
        return passed, failed, errors

    def _read_coverage(self, cov_json_path: Path) -> float:
        """
        Extract the total line-coverage percentage from pytest-cov's JSON report.

        Falls back to -1.0 if the report doesn't exist (pytest-cov not installed).
        """
        if not cov_json_path.exists():
            logger.warning(
                "pytest-cov JSON report not found. "
                "Install pytest-cov for real coverage: pip install pytest-cov"
            )
            return -1.0
        try:
            data = json.loads(cov_json_path.read_text(encoding="utf-8"))
            pct = data.get("totals", {}).get("percent_covered", -1.0)
            return round(float(pct), 2)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not parse coverage JSON: %s", exc)
            return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerationPipeline:
    """
    End-to-end orchestration pipeline.

    Flow
    ----
    Source Code → [CodeParser] → [LLMTestGenerator] → [TestExecutor] → Report

    This mirrors the artifact-generation pipeline described in:
    Huang et al. (2024), "Automated software artifact generation using LLMs,"
    QRS 2024.

    Usage
    -----
    Instantiate with an Anthropic API key (or set ``ANTHROPIC_API_KEY``),
    then call ``run(source_code)`` or ``run_file(filepath)``.
    For evaluation without making API calls, use ``--dry-run`` via the CLI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        output_dir: str = ".",
        dry_run: bool = False,
        max_functions: Optional[int] = None,
    ) -> None:
        self.parser = CodeParser()
        self.generator = LLMTestGenerator(api_key, model=model)
        self.executor = TestExecutor()
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.max_functions = max_functions
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source_code: str) -> List[TestResult]:
        """Run the full pipeline on *source_code* string."""
        logger.info("=" * 62)
        logger.info("  LLM-BASED AUTOMATED TEST CASE GENERATOR")
        logger.info("  Author: Adnan Hassnain | BS CS, NUST Pakistan")
        logger.info("=" * 62)

        functions = self.parser.parse_code_string(source_code)
        if not functions:
            logger.warning("No top-level functions found in source code.")
            return []

        if self.max_functions:
            functions = functions[: self.max_functions]

        logger.info(
            "[PARSE] Found %d function(s): %s",
            len(functions),
            [f.name for f in functions],
        )

        if self.dry_run:
            logger.info("[DRY-RUN] Skipping LLM calls and test execution.")
            self._report_functions(functions)
            return []

        results: List[TestResult] = []
        for func in functions:
            result = self._process_function(func, source_code)
            results.append(result)

        self._print_summary(results)
        self._save_report(results, source_code)
        return results

    def run_file(self, filepath: str) -> List[TestResult]:
        """Run the full pipeline on a Python source file."""
        source = Path(filepath).read_text(encoding="utf-8")
        logger.info("[INPUT] Source file: %s (%d lines)", filepath, len(source.splitlines()))
        return self.run(source)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_function(
        self, func: FunctionInfo, source_code: str
    ) -> TestResult:
        logger.info(
            "[GENERATE] %s()  (complexity=%d, args=%s)",
            func.name, func.cyclomatic_complexity, func.args,
        )
        test_code, gen_ms, prompt_chars = self.generator.generate_tests(
            func, source_code
        )
        n_tests = test_code.count("def test_")
        logger.info("[GENERATE] ✓  %d test case(s) generated in %.0fms", n_tests, gen_ms)

        logger.info("[EXECUTE] Running test suite…")
        partial_result, exec_ms = self.executor.run(source_code, test_code, func.name)

        # Fill in LLM-specific fields
        partial_result.llm_model = self.generator.model
        partial_result.generation_time_ms = gen_ms
        partial_result.prompt_chars = prompt_chars

        cov_str = (
            f"{partial_result.line_coverage_pct:.1f}%"
            if partial_result.line_coverage_pct >= 0
            else "N/A (install pytest-cov)"
        )
        logger.info(
            "[RESULT]  ✓ Passed: %d | ✗ Failed: %d | ⚠ Errors: %d | "
            "Line Coverage: %s | Fault Detected: %s",
            partial_result.passed,
            partial_result.failed,
            partial_result.errors,
            cov_str,
            "YES" if partial_result.fault_detected else "no",
        )
        return partial_result

    def _report_functions(self, functions: List[FunctionInfo]) -> None:
        """Print parsed function info (dry-run mode)."""
        logger.info("[DRY-RUN] Parsed functions:")
        for f in functions:
            logger.info(
                "  • %s(%s)  → %s  [complexity=%d, line=%d]",
                f.name, ", ".join(f.args),
                f.return_annotation or "Any",
                f.cyclomatic_complexity,
                f.lineno,
            )

    def _print_summary(self, results: List[TestResult]) -> None:
        total_passed = sum(r.passed for r in results)
        total_failed = sum(r.failed for r in results)
        total_tests = total_passed + total_failed
        cov_values = [r.line_coverage_pct for r in results if r.line_coverage_pct >= 0]
        avg_cov = sum(cov_values) / len(cov_values) if cov_values else -1
        faults = sum(1 for r in results if r.fault_detected)
        avg_gen_ms = sum(r.generation_time_ms for r in results) / max(len(results), 1)

        logger.info("=" * 62)
        logger.info("  SUMMARY REPORT")
        logger.info("=" * 62)
        logger.info("  Functions Tested   : %d", len(results))
        logger.info("  Total Tests Run    : %d", total_tests)
        logger.info("  Tests Passed       : %d", total_passed)
        logger.info("  Tests Failed       : %d", total_failed)
        logger.info(
            "  Avg Line Coverage  : %s",
            f"{avg_cov:.1f}%" if avg_cov >= 0 else "N/A",
        )
        logger.info(
            "  Faults Detected    : %d / %d functions", faults, len(results)
        )
        logger.info("  Avg Generation Time: %.0fms per function", avg_gen_ms)
        logger.info("=" * 62)

    def _save_report(self, results: List[TestResult], source_code: str) -> None:
        report = {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "llm_model": self.generator.model,
                "source_lines": len(source_code.splitlines()),
            },
            "summary": {
                "functions_analyzed": len(results),
                "total_tests_run": sum(r.passed + r.failed for r in results),
                "total_passed": sum(r.passed for r in results),
                "total_failed": sum(r.failed for r in results),
                "faults_detected": sum(1 for r in results if r.fault_detected),
            },
            "results": [asdict(r) for r in results],
        }
        out_path = self.output_dir / "test_generation_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("[SAVED] Report → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Sample Code — Evaluation Benchmark
# ─────────────────────────────────────────────────────────────────────────────
# The functions below serve as a self-contained benchmark for evaluating
# the test generator.  They include a deliberately incorrect implementation
# (marked with [BUG]) to provide ground truth for fault-detection assessment.

SAMPLE_CODE = '''
def calculate_discount(price: float, discount_pct: float) -> float:
    """
    Calculate the final price after applying a percentage discount.

    Args:
        price:        Original price (must be non-negative).
        discount_pct: Discount percentage in the range [0, 100].

    Returns:
        Final price after discount, rounded to 2 decimal places.

    Raises:
        ValueError: If discount_pct > 100 or price < 0.
    """
    if price < 0:
        raise ValueError(f"Price must be non-negative, got {price}")
    if discount_pct > 100:
        raise ValueError("Discount cannot exceed 100%")
    discounted = price - (price * discount_pct / 100)
    return round(discounted, 2)


def find_max_subarray_sum(nums: list) -> int:
    """
    Find the maximum sum of a contiguous subarray (Kadane\'s algorithm).

    Args:
        nums: List of integers (may be empty).

    Returns:
        Maximum subarray sum, or 0 for an empty list.
    """
    if not nums:
        return 0
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum


def classify_software_defect(severity_score: int) -> str:
    """
    Classify a software defect by severity score (QA defect tracking).

    Args:
        severity_score: Integer in [1, 10].

    Returns:
        One of "LOW", "MEDIUM", "HIGH", or "CRITICAL".

    Raises:
        ValueError: If severity_score is outside [1, 10].
    """
    if severity_score <= 0 or severity_score > 10:
        raise ValueError(f"Score must be 1-10, got {severity_score}")
    if severity_score <= 3:
        return "LOW"
    elif severity_score <= 6:
        return "MEDIUM"
    elif severity_score <= 8:
        return "HIGH"
    else:
        return "CRITICAL"


def binary_search(arr: list, target: int) -> int:
    """
    Search for *target* in a sorted list and return its index.

    [BUG] This implementation contains an intentional off-by-one error
    (``right = len(arr)`` instead of ``len(arr) - 1``) to serve as a
    ground-truth fault for evaluating the generator\'s fault-detection rate.

    Args:
        arr:    Sorted list of integers.
        target: Value to search for.

    Returns:
        Index of *target*, or -1 if not found.
    """
    left, right = 0, len(arr)  # BUG: should be len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
'''


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_generator",
        description=(
            "LLM-Based Automated Test Case Generator\n"
            "Generates pytest unit tests for Python functions using Claude.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run on built-in sample code (requires API key):\n"
            "  python test_generator.py\n\n"
            "  # Run on your own source file:\n"
            "  python test_generator.py --input mymodule.py\n\n"
            "  # Parse only — no API call (useful for CI validation):\n"
            "  python test_generator.py --dry-run\n\n"
            "  # Limit to first 2 functions, save report elsewhere:\n"
            "  python test_generator.py --max-functions 2 --output-dir results/\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Python source file to generate tests for. "
             "Defaults to the built-in sample benchmark.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        metavar="DIR",
        help="Directory to write the JSON report (default: current directory).",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-6",
        metavar="MODEL",
        help="Anthropic model to use (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--max-functions",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N functions (useful for quick trials).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display function metadata only; skip LLM and test execution.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if not args.dry_run:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "  Set it with:  export ANTHROPIC_API_KEY='sk-ant-…'\n"
                "  Or run with:  --dry-run to skip API calls."
            )
            sys.exit(1)
    else:
        api_key = None

    pipeline = TestGenerationPipeline(
        api_key=api_key,
        model=args.model,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_functions=args.max_functions,
    )

    if args.input:
        pipeline.run_file(args.input)
    else:
        logger.info("[INPUT] Using built-in benchmark code (4 functions, 1 intentional bug)")
        pipeline.run(SAMPLE_CODE)


if __name__ == "__main__":
    main()
