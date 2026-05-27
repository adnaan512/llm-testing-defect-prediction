"""
Automated API Fault Localization via Spectrum-Based and ML-Enhanced Analysis
=============================================================================
Research Reference:
  Chin-Yu Huang et al. (2025) — "Deep learning for fault localization
  with coverage data reduction" — NTHU SE Lab.

Abstract
--------
Fault localization (FL) is the process of automatically identifying the root
cause of a software failure.  In microservice architectures, a single failure
may propagate through many services, making manual root-cause analysis
extremely time-consuming.

This module implements and compares two FL techniques applied to API-level
test execution traces:

  Technique 1 — Tarantula (Jones & Harrold, 2005):
    Classic spectrum-based FL using the suspiciousness formula:
      S(e) = (ef / totalFailed) / (ef / totalFailed + ep / totalPassed)

  Technique 2 — Ochiai (Abreu et al., 2007):
    Cosine-similarity-inspired suspiciousness formula that empirically
    outperforms Tarantula on many benchmarks:
      S(e) = ef / sqrt((ef + nf) * (ef + ep))

  Technique 3 — ML-Enhanced Scorer:
    A weighted ensemble of Tarantula score, response-time ratio,
    failure density, and isolation score — inspired by Huang et al. (2025),
    which applies deep learning to reweight coverage signals.

Evaluation uses standard FL metrics: rank of first faulty endpoint,
Top-N accuracy (N = 1, 3, 5), and EXAM score (%).

Author  : Adnan Hassnain
Affil.  : BS CS, NUST Pakistan

Real-World Motivation
---------------------
During QA work at MyTechPassport, fault tracing across microservices was
performed manually.  This tool automates that process using coverage-based
FL techniques from academic research, demonstrating the direct applicability
of software-engineering research to industry problems.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APIEndpoint:
    """Represents a single microservice API endpoint."""
    id: str
    method: str      # HTTP method: GET | POST | PUT | DELETE
    path: str        # URL path template
    service: str     # Owning microservice
    is_faulty: bool = False   # Ground truth (for evaluation)


@dataclass
class TestCase:
    """A single API test case with its execution trace."""
    id: str
    name: str
    endpoints_hit: List[str]   # Ordered list of endpoint IDs exercised
    passed: bool
    response_time_ms: float
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CoverageMatrix:
    """
    Binary test-endpoint coverage matrix.

    This is the core data structure for spectrum-based fault localization.
    ``matrix[test_id][endpoint_id] = 1`` iff the test exercised the endpoint.

    The statistical summary per endpoint — (ef, ep, nf, np) — feeds directly
    into the Tarantula and Ochiai suspiciousness formulae.
    """
    matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    test_results: Dict[str, bool] = field(default_factory=dict)

    def record(self, test_id: str, endpoint_id: str, covered: bool) -> None:
        self.matrix.setdefault(test_id, {})[endpoint_id] = int(covered)

    def record_result(self, test_id: str, passed: bool) -> None:
        self.test_results[test_id] = passed

    def endpoint_stats(
        self, endpoint_id: str, all_test_ids: List[str]
    ) -> Dict[str, int]:
        """
        Compute the four Tarantula/Ochiai statistics for *endpoint_id*:
          ef : covered by failing tests
          ep : covered by passing tests
          nf : not covered by failing tests
          np : not covered by passing tests
        """
        ef = ep = nf = np_ = 0
        for tid in all_test_ids:
            covered = self.matrix.get(tid, {}).get(endpoint_id, 0)
            passed = self.test_results.get(tid, True)
            if covered and not passed:
                ef += 1
            elif covered and passed:
                ep += 1
            elif not covered and not passed:
                nf += 1
            else:
                np_ += 1
        return {"ef": ef, "ep": ep, "nf": nf, "np": np_}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic API Test Log Generator
# ─────────────────────────────────────────────────────────────────────────────

class APITestLogGenerator:
    """
    Generates realistic synthetic API test execution logs for a
    microservice system with four services: auth, user, payment, notification.

    Fault injection: specified endpoints are marked faulty.  Tests that
    exercise a faulty endpoint fail with probability ~0.92 (8% noise to
    simulate real-world flakiness).  Faulty endpoints also exhibit 2–5×
    elevated response times, a common real-world symptom.
    """

    SCENARIOS = [
        ("Login Flow",    ["auth_login",    "auth_verify",     "user_get"]),
        ("Register",      ["user_create",   "auth_login",      "notify_email"]),
        ("Payment Flow",  ["auth_verify",   "payment_init",    "payment_confirm", "notify_email"]),
        ("Profile Update", ["auth_verify",   "user_get",        "user_update"]),
        ("Delete Account", ["auth_verify",   "user_get",        "user_delete"]),
        ("Check Payment", ["auth_verify",   "payment_status"]),
        ("SMS Notify",    ["auth_verify",   "notify_sms"]),
        ("Token Refresh", ["auth_refresh",  "auth_verify"]),
    ]

    HTTP_ERRORS = {
        "500": "Internal Server Error",
        "503": "Service Unavailable",
        "504": "Gateway Timeout",
        "422": "Unprocessable Entity",
    }

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        self.endpoints = self._define_endpoints()

    def _define_endpoints(self) -> List[APIEndpoint]:
        return [
            APIEndpoint("auth_login",     "POST",   "/auth/login",       "auth"),
            APIEndpoint("auth_verify",    "GET",    "/auth/verify",      "auth"),
            APIEndpoint("auth_refresh",   "POST",   "/auth/refresh",     "auth"),
            APIEndpoint("user_get",       "GET",    "/users/{id}",       "user"),
            APIEndpoint("user_create",    "POST",   "/users",            "user"),
            APIEndpoint("user_update",    "PUT",    "/users/{id}",       "user"),
            APIEndpoint("user_delete",    "DELETE", "/users/{id}",       "user"),
            APIEndpoint("payment_init",   "POST",   "/payments/init",    "payment"),
            APIEndpoint("payment_confirm", "POST",   "/payments/confirm", "payment"),
            APIEndpoint("payment_status", "GET",    "/payments/{id}",    "payment"),
            APIEndpoint("notify_email",   "POST",   "/notify/email",     "notification"),
            APIEndpoint("notify_sms",     "POST",   "/notify/sms",       "notification"),
        ]

    def inject_faults(self, faulty_endpoint_ids: List[str]) -> None:
        """Mark specified endpoints as faulty (ground truth)."""
        for ep in self.endpoints:
            ep.is_faulty = ep.id in faulty_endpoint_ids

    def generate_tests(self, n_tests: int = 200) -> List[TestCase]:
        faulty_ids = {ep.id for ep in self.endpoints if ep.is_faulty}
        tests: List[TestCase] = []

        for i in range(n_tests):
            scenario_name, endpoint_ids = random.choice(self.SCENARIOS)
            endpoint_ids = list(endpoint_ids)   # copy

            # Add random endpoint noise (10% chance) — simulates trace noise
            if random.random() < 0.10:
                endpoint_ids.append(random.choice(
                    [ep.id for ep in self.endpoints]
                ))

            hits_faulty = any(eid in faulty_ids for eid in endpoint_ids)
            flaky = random.random() < 0.08   # 8% noise
            passed = (not hits_faulty) != flaky

            rt = random.gauss(120.0, 30.0)
            if hits_faulty:
                rt *= random.uniform(2.0, 5.0)

            error_code = error_msg = None
            if not passed:
                error_code = random.choice(list(self.HTTP_ERRORS))
                error_msg = self.HTTP_ERRORS[error_code]

            tests.append(TestCase(
                id=f"test_{i:04d}",
                name=f"{scenario_name}_{i}",
                endpoints_hit=endpoint_ids,
                passed=passed,
                response_time_ms=round(max(10.0, rt), 1),
                error_code=error_code,
                error_message=error_msg,
            ))

        return tests


# ─────────────────────────────────────────────────────────────────────────────
# Tarantula  (Jones & Harrold, 2005)
# ─────────────────────────────────────────────────────────────────────────────

class TarantulaLocalizer:
    """
    Spectrum-based fault localization using the Tarantula formula.

    Reference
    ---------
    Jones, J.A. & Harrold, M.J. (2005). Empirical evaluation of the
    Tarantula automatic fault-localization technique. ASE 2005.

    Formula
    -------
    S(e) = (ef/totalFailed) / (ef/totalFailed + ep/totalPassed)

    A score of 1.0 means the endpoint was covered exclusively by failing
    tests (maximally suspicious); 0.0 means covered only by passing tests.
    """

    def compute(
        self,
        coverage: CoverageMatrix,
        endpoint_ids: List[str],
    ) -> Dict[str, float]:
        all_tests = list(coverage.test_results.keys())
        total_failed = sum(1 for v in coverage.test_results.values() if not v)
        total_passed = sum(1 for v in coverage.test_results.values() if v)
        scores: Dict[str, float] = {}

        for eid in endpoint_ids:
            s = coverage.endpoint_stats(eid, all_tests)
            ef, ep = s["ef"], s["ep"]

            if total_failed == 0:
                scores[eid] = 0.0
                continue

            ef_n = ef / total_failed
            ep_n = ep / total_passed if total_passed > 0 else 0.0

            scores[eid] = round(
                ef_n / (ef_n + ep_n) if (ef_n + ep_n) > 0 else 0.0, 4
            )

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Ochiai  (Abreu et al., 2007)
# ─────────────────────────────────────────────────────────────────────────────

class OchiaiLocalizer:
    """
    Spectrum-based fault localization using the Ochiai formula.

    Reference
    ---------
    Abreu, R. et al. (2007). On the accuracy of spectrum-based fault
    localization. TAIC PART 2007.

    Formula
    -------
    S(e) = ef / sqrt((ef + nf) * (ef + ep))

    Ochiai consistently outperforms Tarantula on most benchmark suites
    (Naish et al., 2011), making it a key baseline in FL research.
    """

    def compute(
        self,
        coverage: CoverageMatrix,
        endpoint_ids: List[str],
    ) -> Dict[str, float]:
        all_tests = list(coverage.test_results.keys())
        scores: Dict[str, float] = {}

        for eid in endpoint_ids:
            s = coverage.endpoint_stats(eid, all_tests)
            ef, ep, nf = s["ef"], s["ep"], s["nf"]

            denominator = math.sqrt((ef + nf) * (ef + ep))
            scores[eid] = round(ef / denominator if denominator > 0 else 0.0, 4)

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# ML-Enhanced Fault Localizer
# ─────────────────────────────────────────────────────────────────────────────

class MLFaultLocalizer:
    """
    Weighted ensemble fault scorer combining spectrum-based scores with
    execution-trace features.

    Features
    --------
    tarantula          : Classic Tarantula suspiciousness score
    ochiai             : Ochiai suspiciousness score (usually better baseline)
    response_time_ratio: Ratio of avg failure response time to avg pass time
                         (normalised to [0, 1] over a 1–5× range)
    failure_density    : Proportion of tests covering this endpoint that fail
    isolation_score    : How exclusively the endpoint appears in failing tests

    Weights are informed by the feature-importance insight in Huang et al.
    (2025): spectrum scores dominate, but response time and density provide
    complementary signal, especially for performance-related faults.
    """

    WEIGHTS: Dict[str, float] = {
        "tarantula":           0.30,
        "ochiai":              0.30,
        "response_time_ratio": 0.20,
        "failure_density":     0.12,
        "isolation_score":     0.08,
    }

    def compute_features(
        self,
        endpoint_id: str,
        tests: List[TestCase],
        tarantula_score: float,
        ochiai_score: float,
    ) -> Dict[str, float]:
        hit_tests = [t for t in tests if endpoint_id in t.endpoints_hit]
        failed_hits = [t for t in hit_tests if not t.passed]
        passed_hits = [t for t in hit_tests if t.passed]

        if not hit_tests:
            return {k: 0.0 for k in self.WEIGHTS}

        avg_fail_rt = (sum(t.response_time_ms for t in failed_hits) / len(failed_hits)
                       if failed_hits else 0.0)
        avg_pass_rt = (sum(t.response_time_ms for t in passed_hits) / len(passed_hits)
                       if passed_hits else avg_fail_rt)

        rt_ratio = avg_fail_rt / avg_pass_rt if avg_pass_rt > 0 else 1.0
        rt_ratio_norm = min(1.0, max(0.0, (rt_ratio - 1.0) / 4.0))

        failure_density = len(failed_hits) / len(hit_tests)
        isolation_score = len(failed_hits) / (len(failed_hits) + len(passed_hits) + 1)

        return {
            "tarantula":           tarantula_score,
            "ochiai":              ochiai_score,
            "response_time_ratio": round(rt_ratio_norm, 4),
            "failure_density":     round(failure_density, 4),
            "isolation_score":     round(isolation_score, 4),
        }

    def score(
        self,
        endpoint_id: str,
        tests: List[TestCase],
        tarantula_score: float,
        ochiai_score: float,
    ) -> float:
        features = self.compute_features(
            endpoint_id, tests, tarantula_score, ochiai_score
        )
        return round(sum(features[k] * self.WEIGHTS[k] for k in self.WEIGHTS), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Service-Level Fault Clustering
# ─────────────────────────────────────────────────────────────────────────────

class FaultClusterAnalyzer:
    """
    Aggregates endpoint suspiciousness scores to the service level.

    In a microservice architecture, knowing *which service* contains the
    faulty endpoint allows QA teams to escalate to the correct team first,
    reducing mean time to repair (MTTR).
    """

    def analyze(
        self,
        endpoint_scores: Dict[str, float],
        endpoints: List[APIEndpoint],
    ) -> Dict[str, float]:
        service_scores: Dict[str, List[float]] = defaultdict(list)
        ep_map = {ep.id: ep for ep in endpoints}

        for eid, score in endpoint_scores.items():
            service = ep_map[eid].service if eid in ep_map else "unknown"
            service_scores[service].append(score)

        return {
            svc: round(sum(scores) / len(scores), 4)
            for svc, scores in service_scores.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Localization Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationEvaluator:
    """
    Standard FL evaluation metrics.

    rank_first_fault : Position of the first faulty endpoint in the ranked list
                       (1 = perfect; lower is better)
    Top-N accuracy   : Is any faulty endpoint in the top N? (N = 1, 3, 5)
    EXAM score       : Percentage of endpoints inspected before finding a fault
                       (lower is better; 0% = perfect)
    """

    def evaluate(
        self,
        ranked: List[Tuple[str, float]],
        faulty_ids: Set[str],
    ) -> Dict:
        rank_first = next(
            (i + 1 for i, (eid, _) in enumerate(ranked) if eid in faulty_ids),
            None,
        )
        total = len(ranked)
        exam = (rank_first / total * 100) if rank_first else 100.0

        return {
            "rank_first_fault": rank_first,
            "top1_acc": any(eid in faulty_ids for eid, _ in ranked[:1]),
            "top3_acc": any(eid in faulty_ids for eid, _ in ranked[:3]),
            "top5_acc": any(eid in faulty_ids for eid, _ in ranked[:5]),
            "exam_score_pct": round(exam, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(
    n_tests: int = 300,
    faulty_endpoints: Optional[List[str]] = None,
    output_dir: str = ".",
) -> None:
    if faulty_endpoints is None:
        faulty_endpoints = ["payment_confirm", "notify_email"]

    logger.info("=" * 68)
    logger.info("  AUTOMATED API FAULT LOCALIZATION")
    logger.info("  Tarantula · Ochiai · ML-Enhanced Ensemble")
    logger.info("  Inspired by Huang et al. (2025) — NTHU SE Lab")
    logger.info("  Author: Adnan Hassnain | BS CS, NUST Pakistan")
    logger.info("=" * 68)

    # 1. Initialise environment
    logger.info("[SETUP] Initialising microservice API test environment…")
    generator = APITestLogGenerator(seed=42)
    generator.inject_faults(faulty_endpoints)
    n_services = len({ep.service for ep in generator.endpoints})
    logger.info("[SETUP] Injected faults: %s", faulty_endpoints)
    logger.info(
        "[SETUP] %d endpoints across %d microservices",
        len(generator.endpoints), n_services,
    )

    # 2. Generate test execution logs
    logger.info("[TEST]  Generating %d API test execution traces…", n_tests)
    tests = generator.generate_tests(n_tests=n_tests)
    failed = [t for t in tests if not t.passed]
    logger.info(
        "[TEST]  %d tests | %d failed (%.1f%% failure rate)",
        len(tests), len(failed), len(failed) / len(tests) * 100,
    )

    # 3. Build coverage matrix
    logger.info("[COVER] Building test–endpoint coverage matrix…")
    coverage = CoverageMatrix()
    endpoint_ids = [ep.id for ep in generator.endpoints]

    for test in tests:
        coverage.record_result(test.id, test.passed)
        for eid in endpoint_ids:
            coverage.record(test.id, eid, eid in test.endpoints_hit)

    # 4. Compute suspiciousness scores
    logger.info("[FL]    Computing Tarantula suspiciousness scores…")
    tarantula_scores = TarantulaLocalizer().compute(coverage, endpoint_ids)

    logger.info("[FL]    Computing Ochiai suspiciousness scores…")
    ochiai_scores = OchiaiLocalizer().compute(coverage, endpoint_ids)

    logger.info("[FL]    Computing ML-enhanced ensemble scores…")
    ml_localizer = MLFaultLocalizer()
    ml_scores = {
        eid: ml_localizer.score(
            eid, tests, tarantula_scores[eid], ochiai_scores[eid]
        )
        for eid in endpoint_ids
    }

    # 5. Rank endpoints
    tarantula_ranked = sorted(tarantula_scores.items(), key=lambda t: t[1], reverse=True)
    ochiai_ranked = sorted(ochiai_scores.items(),    key=lambda t: t[1], reverse=True)
    ml_ranked = sorted(ml_scores.items(),        key=lambda t: t[1], reverse=True)

    # 6. Service-level clustering
    service_scores = FaultClusterAnalyzer().analyze(ml_scores, generator.endpoints)

    # 7. Evaluate all methods
    faulty_set = set(faulty_endpoints)
    evaluator = LocalizationEvaluator()
    eval_results = {
        "Tarantula":   evaluator.evaluate(tarantula_ranked, faulty_set),
        "Ochiai":      evaluator.evaluate(ochiai_ranked,    faulty_set),
        "ML Enhanced": evaluator.evaluate(ml_ranked,        faulty_set),
    }

    # 8. Print results
    logger.info("")
    logger.info("=" * 68)
    logger.info("  TOP-5 SUSPICIOUS ENDPOINTS  (ML-Enhanced Ranking)")
    logger.info("=" * 68)
    ep_map = {ep.id: ep for ep in generator.endpoints}
    header = (
        f"  {'Rank':<5} {'Endpoint':<22} {'Service':<14}"
        f" {'Tarantula':>10} {'Ochiai':>7} {'ML Score':>9} {'Faulty?':>8}"
    )
    logger.info(header)
    logger.info("  " + "-" * 68)

    for i, (eid, ml_score) in enumerate(ml_ranked[:5]):
        ep = ep_map.get(eid)
        service = ep.service if ep else "?"
        faulty_s = "🔴 YES" if eid in faulty_set else "✅  no"
        t_score = tarantula_scores.get(eid, 0.0)
        o_score = ochiai_scores.get(eid, 0.0)
        logger.info(
            "  #%-4d %-22s %-14s %10.4f %7.4f %9.4f %8s",
            i + 1, eid, service, t_score, o_score, ml_score, faulty_s,
        )

    logger.info("")
    logger.info("  SERVICE FAULT RANKING")
    logger.info("  " + "-" * 40)
    for svc, score in sorted(service_scores.items(), key=lambda t: t[1], reverse=True):
        bar = "█" * int(score * 30)
        logger.info("    %-16s  %.4f  %s", svc, score, bar)

    logger.info("")
    logger.info("  EVALUATION METRICS")
    logger.info("  " + "-" * 58)
    logger.info(
        "  %-22s %6s %6s %6s %6s %8s",
        "Method", "Rank", "Top-1", "Top-3", "Top-5", "EXAM%",
    )
    logger.info("  " + "-" * 58)
    for method, ev in eval_results.items():
        logger.info(
            "  %-22s %6s %6s %6s %6s %7.1f%%",
            method,
            str(ev["rank_first_fault"]),
            "✓" if ev["top1_acc"] else "✗",
            "✓" if ev["top3_acc"] else "✗",
            "✓" if ev["top5_acc"] else "✗",
            ev["exam_score_pct"],
        )

    # 9. Save report
    report = {
        "experiment": {
            "total_tests":       len(tests),
            "failed_tests":      len(failed),
            "faulty_endpoints":  faulty_endpoints,
            "failure_rate_pct":  round(len(failed) / len(tests) * 100, 1),
        },
        "scores": {
            "tarantula": dict(tarantula_ranked),
            "ochiai":    dict(ochiai_ranked),
            "ml":        dict(ml_ranked),
        },
        "service_scores": service_scores,
        "evaluation": {
            method: ev for method, ev in eval_results.items()
        },
    }
    out_path = Path(output_dir) / "fault_localization_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("")
    logger.info("[SAVED] Report → %s", out_path)
    logger.info("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fault_localizer",
        description=(
            "Automated API Fault Localization\n"
            "Compares Tarantula, Ochiai, and ML-Enhanced spectrum-based FL.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tests", type=int, default=300, metavar="N",
        help="Number of synthetic API test cases to generate (default: 300)",
    )
    parser.add_argument(
        "--faulty", nargs="+",
        default=["payment_confirm", "notify_email"],
        metavar="ENDPOINT_ID",
        help="Endpoint IDs to inject faults into (default: payment_confirm notify_email)",
    )
    parser.add_argument(
        "--output-dir", default=".", metavar="DIR",
        help="Directory to write the JSON report",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _configure_logging(args.verbose)
    main(n_tests=args.tests, faulty_endpoints=args.faulty, output_dir=args.output_dir)
