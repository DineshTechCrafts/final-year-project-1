"""
Decision Agent parameter and evidence-robustness harness.

Synthetic evidence only. Does not run segmentation, FAISS, or HCC-TACE-Seg.

Usage (from LiverCancer-MultiAgent-Retrieval):
    python scripts/test_decision_agent_parameters.py
"""

from __future__ import annotations

import copy
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.decision_agent import (  # noqa: E402
    REQUIRED_OUTPUT_KEYS,
    DecisionResultsAgent,
    MalformedResponseError,
    ModelNotInstalledError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from utils.config import load_config  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
MODEL_NAME = "llama3.1:8b"

CLINICAL_CANARIES = (
    "afp 412",
    "child-pugh",
    "bilirubin",
    "sorafenib",
    "chemotherapy",
    "biopsy-proven",
    "patient reports",
    "complains of",
    "weight loss",
    "night sweats",
    "prior tace",
    "transplant listed",
)

INVENTED_CASE_PATTERNS = (
    r"\bSYNTH_CASE_C\b",
    r"\bHCC_\d{3,}\b",
    r"\bCASE_[A-Z]\b",
    r"\bFAKE_CASE\b",
)

PARAMETER_TESTS = (
    {"id": "A", "temperature": 0.0, "max_tokens": 512},
    {"id": "B", "temperature": 0.1, "max_tokens": 512},
    {"id": "C", "temperature": 0.3, "max_tokens": 512},
    {"id": "D", "temperature": 0.5, "max_tokens": 1024},
)


def _base_config() -> dict[str, Any]:
    loaded = load_config(DEFAULT_CONFIG_PATH)
    cfg = dict(loaded.get("decision_agent", loaded))
    cfg["model"] = MODEL_NAME
    cfg["provider"] = "ollama"
    return cfg


def make_agent(temperature: float, max_tokens: int) -> DecisionResultsAgent:
    cfg = _base_config()
    cfg["temperature"] = temperature
    cfg["max_tokens"] = max_tokens
    return DecisionResultsAgent(config=cfg)


def complete_evidence() -> dict[str, Any]:
    return {
        "patient_id": "SYNTH_PARAM_001",
        "imaging": {
            "modality": "CT",
            "image_quality": "adequate",
            "tumor_location": "right hepatic lobe, segment VII",
            "tumor_volume": 21.6,
            "tumor_dimensions": {"x": 35.0, "y": 31.0, "z": 26.0},
        },
        "segmentation": {
            "model": "unet_hcc_baseline",
            "tumor_detected": True,
            "tumor_voxels": 5300,
            "confidence": 0.86,
        },
        "deep_features": {
            "embedding_model": "resnet50_3d",
            "embedding_available": True,
            "feature_summary": "Dense visual embedding available for similarity search.",
        },
        "radiomics": {
            "available": True,
            "shape_features": {"sphericity": 0.68, "compactness": 0.59},
            "first_order_features": {"mean_hu": 58.4, "entropy": 3.1},
            "texture_features": {"glcm_contrast": 11.4},
        },
        "clinical_context": {
            "available": True,
            "data": {
                "ecog": 1,
                "synthetic_note": "No real patient record. ECOG 1 supplied only as synthetic context.",
            },
        },
        "retrieval": {
            "number_of_cases": 2,
            "similar_cases": [
                {
                    "case_id": "SYNTH_CASE_A",
                    "similarity_score": 0.92,
                    "relevant_features": {"location": "segment VII", "volume_ml": 20.9},
                },
                {
                    "case_id": "SYNTH_CASE_B",
                    "similarity_score": 0.81,
                    "relevant_features": {"location": "right lobe", "sphericity": 0.67},
                },
            ],
        },
        "fusion": {
            "final_scores": {"SYNTH_CASE_A": 0.89, "SYNTH_CASE_B": 0.78},
            "ranking_method": "weighted multimodal fusion",
        },
    }


def scenario_missing_clinical() -> dict[str, Any]:
    evidence = complete_evidence()
    evidence["patient_id"] = "SYNTH_PARAM_002"
    evidence["clinical_context"] = {"available": False, "data": {}}
    return evidence


def scenario_empty_retrieval() -> dict[str, Any]:
    evidence = complete_evidence()
    evidence["patient_id"] = "SYNTH_PARAM_003"
    evidence["clinical_context"] = {"available": False, "data": {}}
    evidence["retrieval"] = {"number_of_cases": 0, "similar_cases": []}
    evidence["fusion"] = {"final_scores": {}, "ranking_method": "weighted multimodal fusion"}
    return evidence


def scenario_no_tumor() -> dict[str, Any]:
    evidence = complete_evidence()
    evidence["patient_id"] = "SYNTH_PARAM_004"
    evidence["imaging"]["tumor_location"] = "not applicable"
    evidence["imaging"]["tumor_volume"] = 0.0
    evidence["imaging"]["tumor_dimensions"] = {"x": 0.0, "y": 0.0, "z": 0.0}
    evidence["segmentation"]["tumor_detected"] = False
    evidence["segmentation"]["tumor_voxels"] = 0
    evidence["segmentation"]["confidence"] = 0.90
    evidence["radiomics"] = {
        "available": False,
        "shape_features": {},
        "first_order_features": {},
        "texture_features": {},
    }
    evidence["clinical_context"] = {"available": False, "data": {}}
    evidence["retrieval"] = {
        "number_of_cases": 1,
        "similar_cases": [
            {
                "case_id": "SYNTH_CASE_NONTUMOR",
                "similarity_score": 0.55,
                "relevant_features": {"tumor_present": False},
            }
        ],
    }
    evidence["fusion"] = {
        "final_scores": {"SYNTH_CASE_NONTUMOR": 0.52},
        "ranking_method": "weighted multimodal fusion",
    }
    return evidence


def scenario_low_confidence() -> dict[str, Any]:
    evidence = complete_evidence()
    evidence["patient_id"] = "SYNTH_PARAM_005"
    evidence["clinical_context"] = {"available": False, "data": {}}
    evidence["segmentation"]["confidence"] = 0.35
    return evidence


def scenario_conflicting() -> dict[str, Any]:
    evidence = complete_evidence()
    evidence["patient_id"] = "SYNTH_PARAM_006"
    evidence["clinical_context"] = {"available": False, "data": {}}
    evidence["segmentation"]["tumor_detected"] = True
    evidence["segmentation"]["tumor_voxels"] = 4800
    evidence["segmentation"]["confidence"] = 0.79
    evidence["retrieval"] = {
        "number_of_cases": 2,
        "similar_cases": [
            {
                "case_id": "SYNTH_CASE_LOW_1",
                "similarity_score": 0.21,
                "relevant_features": {"location": "left lobe"},
            },
            {
                "case_id": "SYNTH_CASE_LOW_2",
                "similarity_score": 0.18,
                "relevant_features": {"volume_ml": 4.2},
            },
        ],
    }
    evidence["fusion"] = {
        "final_scores": {"SYNTH_CASE_LOW_1": 0.24, "SYNTH_CASE_LOW_2": 0.19},
        "ranking_method": "weighted multimodal fusion",
    }
    return evidence


def flatten_result(result: dict[str, Any]) -> str:
    parts = []
    for key, value in result.items():
        if key == "meta":
            continue
        parts.append(json.dumps(value, default=str))
    return " ".join(parts)


def known_case_ids(evidence: dict[str, Any]) -> set[str]:
    cases = (evidence.get("retrieval") or {}).get("similar_cases") or []
    return {str(case["case_id"]) for case in cases if isinstance(case, dict) and case.get("case_id")}


def evidence_numbers(evidence: dict[str, Any]) -> set[str]:
    blob = json.dumps(evidence, default=str)
    found = set(re.findall(r"-?\d+\.\d+|-?\d+", blob))
    return found


def mutated_numbers(evidence: dict[str, Any], text: str) -> list[str]:
    """Flag close-but-different numbers that look like altered evidence values."""
    known = evidence_numbers(evidence)
    known_floats = []
    for token in known:
        try:
            known_floats.append((token, float(token)))
        except ValueError:
            continue
    flagged = []
    for match in re.findall(r"-?\d+\.\d+", text):
        try:
            value = float(match)
        except ValueError:
            continue
        if match in known:
            continue
        for token, original in known_floats:
            if original == 0:
                continue
            if abs(original - value) > 0 and abs(original - value) / max(abs(original), 1.0) < 0.03:
                if abs(original - value) >= 0.05:
                    flagged.append(f"{token}->{match}")
    return flagged


def validate_schema(result: dict[str, Any] | None) -> tuple[bool, bool, list[str]]:
    issues: list[str] = []
    json_ok = isinstance(result, dict)
    if not json_ok:
        return False, False, ["response is not a JSON object"]
    missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in result]
    if missing:
        issues.append("missing fields: " + ", ".join(missing))
    similar = result.get("similar_cases")
    if not isinstance(similar, list):
        issues.append("similar_cases is not a list")
    fields_ok = not missing and isinstance(similar, list)
    return json_ok, fields_ok, issues


def check_no_fabricated_cases(evidence: dict[str, Any], result: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    allowed = known_case_ids(evidence)
    for item in result.get("similar_cases") or []:
        case_id = str(item.get("case_id", ""))
        if case_id and case_id not in allowed:
            issues.append(f"fabricated case_id in similar_cases: {case_id}")
    blob = flatten_result(result)
    if not allowed:
        for pattern in INVENTED_CASE_PATTERNS:
            if re.search(pattern, blob, flags=re.IGNORECASE):
                issues.append(f"invented case identifier in text: {pattern}")
    return issues


def check_clinical_not_invented(evidence: dict[str, Any], result: dict[str, Any]) -> list[str]:
    clinical = evidence.get("clinical_context") or {}
    if clinical.get("available"):
        return []
    blob = flatten_result(result).lower()
    issues = []
    for canary in CLINICAL_CANARIES:
        if canary in blob:
            issues.append(f"invented clinical content: {canary}")
    return issues


def check_common_output(evidence: dict[str, Any], result: dict[str, Any]) -> list[str]:
    issues = []
    issues.extend(check_no_fabricated_cases(evidence, result))
    issues.extend(check_clinical_not_invented(evidence, result))
    blob = flatten_result(result)
    issues.extend([f"mutated numeric value {item}" for item in mutated_numbers(evidence, blob)])
    return issues


def run_synthesis(agent: DecisionResultsAgent, evidence: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "ok": False,
        "json_valid": False,
        "fields_ok": False,
        "elapsed": 0.0,
        "length": 0,
        "issues": [],
        "result": None,
        "error": None,
    }
    try:
        result = agent.synthesize(copy.deepcopy(evidence))
        elapsed = time.perf_counter() - started
        payload = {k: v for k, v in result.items() if k != "meta"}
        encoded = json.dumps(payload, default=str)
        json_ok, fields_ok, schema_issues = validate_schema(result)
        extra_issues = check_common_output(evidence, result)
        record.update(
            {
                "ok": json_ok and fields_ok and not extra_issues,
                "json_valid": json_ok,
                "fields_ok": fields_ok,
                "elapsed": elapsed,
                "length": len(encoded),
                "issues": schema_issues + extra_issues,
                "result": result,
            }
        )
    except (
        OllamaUnavailableError,
        ModelNotInstalledError,
        OllamaTimeoutError,
        MalformedResponseError,
    ) as exc:
        record["elapsed"] = time.perf_counter() - started
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["issues"] = [record["error"]]
        if isinstance(exc, MalformedResponseError) and exc.raw_response:
            record["length"] = len(exc.raw_response)
            print("--- Raw model output (not hidden) ---")
            print(exc.raw_response[:4000])
    return record


def print_parameter_report(test_id: str, temperature: float, max_tokens: int, record: dict[str, Any]) -> None:
    print(f"\n--- Parameter Test {test_id} ---")
    print(f"temperature: {temperature}")
    print(f"max_tokens: {max_tokens}")
    print(f"execution time: {record['elapsed']:.2f}s")
    print(f"response validity: {'PASS' if record['ok'] else 'FAIL'}")
    print(f"JSON validity: {'PASS' if record['json_valid'] else 'FAIL'}")
    print(f"required fields present: {'PASS' if record['fields_ok'] else 'FAIL'}")
    print(f"response length: {record['length']} chars")
    if record["issues"]:
        print("issues: " + "; ".join(record["issues"]))


def robustness_complete(record: dict[str, Any]) -> list[str]:
    return [] if record["ok"] else record["issues"] or ["invalid structured explanation"]


def robustness_missing_clinical(record: dict[str, Any]) -> list[str]:
    issues = list(record["issues"])
    result = record.get("result") or {}
    blob = flatten_result(result).lower() if result else ""
    if "unavailable" not in blob and "not available" not in blob and "no clinical" not in blob:
        issues.append("did not state that clinical information is unavailable")
    return issues


def robustness_empty_retrieval(record: dict[str, Any]) -> list[str]:
    issues = list(record["issues"])
    result = record.get("result") or {}
    similar = result.get("similar_cases") if result else None
    if similar:
        issues.append("similar_cases was not empty")
    blob = flatten_result(result).lower() if result else ""
    markers = ("no similar", "none were retrieved", "no cases", "empty", "not retrieved", "0 cases", "zero")
    if not any(marker in blob for marker in markers):
        issues.append("did not state that no similar cases were retrieved")
    if re.search(r"similarity[_ ]score.{0,12}0\.\d{2}", blob) and "0.00" not in blob:
        if similar:
            issues.append("invented similarity scores")
    return issues


def robustness_no_tumor(record: dict[str, Any]) -> list[str]:
    issues = list(record["issues"])
    result = record.get("result") or {}
    blob = flatten_result(result).lower() if result else ""
    positive_claims = (
        "tumor was detected",
        "tumor is present",
        "detected a tumor",
        "confirms a tumor",
        "tumor_detected\": true",
    )
    for claim in positive_claims:
        if claim in blob:
            issues.append(f"claimed tumor detection: {claim}")
    if "not detected" not in blob and "no tumor" not in blob and "tumor_detected\": false" not in blob and "absent" not in blob and "false" not in blob:
        issues.append("did not acknowledge that no tumor was detected")
    return issues


def robustness_low_confidence(record: dict[str, Any]) -> list[str]:
    issues = list(record["issues"])
    result = record.get("result") or {}
    blob = flatten_result(result).lower() if result else ""
    markers = ("0.35", "low confidence", "low segmentation", "uncertain", "uncertainty", "limited confidence")
    if not any(marker in blob for marker in markers):
        issues.append("did not acknowledge low segmentation confidence")
    return issues


def robustness_conflicting(record: dict[str, Any]) -> list[str]:
    issues = list(record["issues"])
    result = record.get("result") or {}
    blob = flatten_result(result).lower() if result else ""
    mentions_lesion = any(
        token in blob
        for token in ("lesion", "tumor", "segmentation indicates", "tumor_detected", "detected")
    )
    mentions_weak_retrieval = any(
        token in blob
        for token in (
            "disagreement",
            "disagree",
            "conflict",
            "inconsistent",
            "low similarity",
            "similarity is low",
            "weak retrieval",
            "weak similarity",
            "not similar",
            "limited similarity",
            "does not confirm",
            "do not strongly",
            "not corroborated",
            "in contrast",
            "0.21",
            "0.18",
            "low score",
        )
    )
    if not (mentions_lesion and mentions_weak_retrieval):
        issues.append(
            "did not describe disagreement between segmentation and low retrieval similarity"
        )
        synthesis = (result.get("evidence_synthesis") or "")[:400]
        if synthesis:
            issues.append("synthesis excerpt: " + synthesis.replace("\n", " "))
    if "definitely" in blob and "cancer" in blob:
        issues.append("forced a positive diagnostic conclusion")
    return issues


def recommend_config(param_rows: list[dict[str, Any]]) -> str:
    passing = [row for row in param_rows if row["record"]["ok"]]
    if not passing:
        return "No passing parameter configuration. Keep Ollama healthy and retry."
    complete = [row for row in passing if row["record"]["fields_ok"] and row["record"]["json_valid"]]
    pool = complete or passing
    preferred = [row for row in pool if row["temperature"] == 0.1]
    if preferred:
        chosen = min(preferred, key=lambda row: (row["max_tokens"] < 1024, row["record"]["elapsed"]))
    else:
        chosen = min(pool, key=lambda row: (row["temperature"], -row["max_tokens"]))
    return (
        f"temperature={chosen['temperature']}, max_tokens={chosen['max_tokens']} "
        f"(stable JSON, lowest suitable temperature among passing runs; "
        f"{chosen['record']['elapsed']:.1f}s)"
    )


def yn(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("Decision Agent parameter/evaluation harness")
    print("Synthetic evidence only. Model: llama3.1:8b via local Ollama.\n")

    try:
        probe = make_agent(0.1, 512)
        probe.check_ollama()
        print("Ollama: reachable; model llama3.1:8b installed.\n")
    except (OllamaUnavailableError, ModelNotInstalledError, OllamaTimeoutError) as exc:
        print(f"Ollama/model check failed: {exc}")
        print("\nDecision Agent Parameter Testing")
        print("===============================")
        print("Model: llama3.1:8b")
        print("Overall: FAIL")
        return 1

    param_evidence = complete_evidence()
    param_rows: list[dict[str, Any]] = []
    print("PART 1 — MODEL PARAMETER TESTS")
    for spec in PARAMETER_TESTS:
        agent = make_agent(spec["temperature"], spec["max_tokens"])
        record = run_synthesis(agent, param_evidence)
        print_parameter_report(spec["id"], spec["temperature"], spec["max_tokens"], record)
        param_rows.append({**spec, "record": record})

    robustness_specs = (
        ("S1", "Complete evidence", complete_evidence(), robustness_complete),
        ("S2", "Missing clinical context", scenario_missing_clinical(), robustness_missing_clinical),
        ("S3", "Empty retrieval", scenario_empty_retrieval(), robustness_empty_retrieval),
        ("S4", "No tumor detected", scenario_no_tumor(), robustness_no_tumor),
        ("S5", "Low segmentation confidence", scenario_low_confidence(), robustness_low_confidence),
        ("S6", "Conflicting evidence", scenario_conflicting(), robustness_conflicting),
    )

    print("\nPART 2 — EVIDENCE ROBUSTNESS TESTS")
    robustness_rows: list[dict[str, Any]] = []
    demo_agent = make_agent(0.1, 1024)
    for test_id, title, evidence, checker in robustness_specs:
        record = run_synthesis(demo_agent, evidence)
        extra = checker(record)
        record["issues"] = extra
        record["ok"] = record["json_valid"] and record["fields_ok"] and not extra
        print(f"\n--- {test_id} {title} ---")
        print(f"JSON validity: {yn(record['json_valid'])}")
        print(f"required fields: {yn(record['fields_ok'])}")
        print(f"execution time: {record['elapsed']:.2f}s")
        print(f"result: {yn(record['ok'])}")
        if extra:
            print("issues: " + "; ".join(extra))
        robustness_rows.append(
            {
                "id": test_id,
                "temperature": 0.1,
                "max_tokens": 1024,
                "record": record,
            }
        )

    all_rows = param_rows + robustness_rows
    print("\nPART 4 — PERFORMANCE SUMMARY")
    print("\nTest | Temperature | Max Tokens | JSON | Required Fields | Time | Result")
    print("-----|-------------|------------|------|-----------------|------|--------")
    for row in all_rows:
        rec = row["record"]
        print(
            f"{row['id']} | {row['temperature']} | {row['max_tokens']} | "
            f"{yn(rec['json_valid'])} | {yn(rec['fields_ok'])} | "
            f"{rec['elapsed']:.2f}s | {yn(rec['ok'])}"
        )

    param_pass = all(row["record"]["ok"] for row in param_rows)
    s1 = next(row for row in robustness_rows if row["id"] == "S1")
    s2 = next(row for row in robustness_rows if row["id"] == "S2")
    s3 = next(row for row in robustness_rows if row["id"] == "S3")
    s4 = next(row for row in robustness_rows if row["id"] == "S4")
    s5 = next(row for row in robustness_rows if row["id"] == "S5")
    s6 = next(row for row in robustness_rows if row["id"] == "S6")
    robustness_pass = all(row["record"]["ok"] for row in robustness_rows)
    overall = param_pass and robustness_pass

    print("\nDecision Agent Parameter Testing")
    print("===============================")
    print(f"\nModel: {MODEL_NAME}")
    print(f"\nParameter tests:\n{yn(param_pass)}")
    print(f"\nEvidence robustness:\n{yn(robustness_pass)}")
    print(f"\nMissing evidence handling:\n{yn(s2['record']['ok'])}")
    print(f"\nEmpty retrieval handling:\n{yn(s3['record']['ok'])}")
    print(f"\nNo-tumor handling:\n{yn(s4['record']['ok'])}")
    print(f"\nLow-confidence handling:\n{yn(s5['record']['ok'])}")
    print(f"\nConflicting-evidence handling:\n{yn(s6['record']['ok'])}")
    print(f"\nOverall:\n{yn(overall)}")
    print("\nRecommended demo configuration:")
    print(recommend_config(param_rows))
    if not s1["record"]["ok"]:
        print("Complete-evidence scenario (S1) did not produce a valid structured explanation.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
