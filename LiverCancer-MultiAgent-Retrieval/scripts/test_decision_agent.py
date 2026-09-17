"""
Standalone Decision / Results Agent test.

Uses a fully synthetic evidence object. Does not load the HCC-TACE-Seg dataset
or run segmentation / retrieval pipelines.

Usage (from LiverCancer-MultiAgent-Retrieval):
    python scripts/test_decision_agent.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.decision_agent import (  # noqa: E402
    DecisionResultsAgent,
    MalformedResponseError,
    MissingEvidenceError,
    ModelNotInstalledError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    extract_json_object,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

HALLUCINATION_CANARIES = (
    "SYNTH_CASE_C",
    "AFP 412",
    "Child-Pugh B",
    "prior TACE in 2019",
    "biopsy-proven",
    "definitely has cancer",
)

SYNTHETIC_EVIDENCE = {
    "patient_id": "SYNTH_001",
    "imaging": {
        "modality": "CT",
        "image_quality": "adequate, mild motion near the dome",
        "tumor_location": "right hepatic lobe, segment VIII",
        "tumor_volume": 18.4,
        "tumor_dimensions": {"x": 32.0, "y": 28.0, "z": 24.0},
    },
    "segmentation": {
        "model": "unet_hcc_baseline",
        "tumor_detected": True,
        "tumor_voxels": 4120,
        "confidence": 0.81,
    },
    "deep_features": {
        "embedding_model": "resnet50_3d",
        "embedding_available": True,
        "feature_summary": "Dense visual embedding available for FAISS query.",
    },
    "radiomics": {
        "available": True,
        "shape_features": {"sphericity": 0.72, "compactness": 0.61},
        "first_order_features": {"mean_hu": 61.2, "entropy": 3.4},
        "texture_features": {"glcm_contrast": 12.1},
    },
    "clinical_context": {
        "available": False,
        "data": {},
    },
    "retrieval": {
        "number_of_cases": 2,
        "similar_cases": [
            {
                "case_id": "SYNTH_CASE_A",
                "similarity_score": 0.91,
                "relevant_features": {"location": "segment VIII", "volume_ml": 17.8},
            },
            {
                "case_id": "SYNTH_CASE_B",
                "similarity_score": 0.84,
                "relevant_features": {"location": "right lobe", "sphericity": 0.70},
            },
        ],
    },
    "fusion": {
        "final_scores": {"SYNTH_CASE_A": 0.88, "SYNTH_CASE_B": 0.81},
        "ranking_method": "weighted multimodal fusion",
    },
}


def _flatten_result_text(result: dict) -> str:
    parts = []
    for key, value in result.items():
        if key == "meta":
            continue
        parts.append(json.dumps(value, default=str))
    return " ".join(parts)


def test_json_extraction() -> None:
    parsed = extract_json_object('prefix {"summary": "ok", "n": 1} suffix')
    assert parsed["summary"] == "ok"
    try:
        extract_json_object("this is not json")
    except MalformedResponseError as exc:
        assert exc.raw_response == "this is not json"
        return
    raise AssertionError("Malformed JSON should not be silently accepted.")


def test_missing_evidence(agent: DecisionResultsAgent) -> None:
    try:
        agent.synthesize(None)
        raise AssertionError("Missing evidence should raise MissingEvidenceError.")
    except MissingEvidenceError:
        pass

    try:
        agent.synthesize({"patient_id": "only"})
        raise AssertionError("Incomplete evidence should raise MissingEvidenceError.")
    except MissingEvidenceError:
        pass


def test_empty_retrieval_handling(agent: DecisionResultsAgent, ollama_ok: bool) -> bool:
    if not ollama_ok:
        return False
    empty = json.loads(json.dumps(SYNTHETIC_EVIDENCE))
    empty["retrieval"] = {"number_of_cases": 0, "similar_cases": []}
    result = agent.synthesize(empty)
    if result["similar_cases"]:
        raise AssertionError("Empty retrieval must not produce invented similar_cases.")
    text = (result["retrieval_findings"] + " " + result["limitations"]).lower()
    if "retriev" not in text and "no " not in text and "unavailable" not in text and "empty" not in text:
        raise AssertionError("Empty retrieval was not described in the findings.")
    return True


def evaluate_hallucination(result: dict) -> bool:
    blob = _flatten_result_text(result)
    blob_l = blob.lower()
    for canary in HALLUCINATION_CANARIES:
        if canary.lower() in blob_l:
            print(f"  Hallucination canary found: {canary}")
            return False

    allowed = {"SYNTH_CASE_A", "SYNTH_CASE_B"}
    for item in result.get("similar_cases", []):
        if item.get("case_id") not in allowed:
            print(f"  Invented similar case_id: {item.get('case_id')}")
            return False

    combined = blob_l
    if "unavailable" not in combined and "not available" not in combined and "no clinical" not in combined:
        print("  Missing statement that clinical context is unavailable.")
        return False
    return True


def evaluate_structure(result: dict) -> bool:
    required = (
        "summary",
        "segmentation_findings",
        "feature_findings",
        "retrieval_findings",
        "evidence_synthesis",
        "similar_cases",
        "limitations",
        "clinical_note",
    )
    for key in required:
        if key not in result:
            print(f"  Missing output key: {key}")
            return False
    ids = {item["case_id"] for item in result["similar_cases"]}
    if "SYNTH_CASE_A" not in ids or "SYNTH_CASE_B" not in ids:
        print(f"  similar_cases did not summarize both retrieved IDs: {ids}")
        return False
    return True


def main() -> int:
    agent = DecisionResultsAgent()
    ollama_pass = False
    structured_pass = False
    hallucination_pass = False
    decision_pass = False
    model_name = agent.model

    print("Running Decision / Results Agent tests (synthetic evidence only)...\n")
    test_json_extraction()
    test_missing_evidence(agent)
    print("Error handling (missing/incomplete evidence): PASS")

    ollama_reachable = False
    model_installed = False
    try:
        agent.check_ollama()
        ollama_reachable = True
        model_installed = True
        print("Ollama reachability: PASS")
        print(f"Model install ({agent.model}): PASS")
    except ModelNotInstalledError as exc:
        ollama_reachable = True
        print("Ollama reachability: PASS (server up)")
        print(f"Model install: FAIL ({exc})")
    except OllamaUnavailableError as exc:
        print(f"Ollama reachability: FAIL ({exc})")
    except OllamaTimeoutError as exc:
        print(f"Ollama reachability: FAIL ({exc})")

    ollama_pass = ollama_reachable
    if ollama_pass and model_installed:
        try:
            result = agent.synthesize(SYNTHETIC_EVIDENCE)
            structured_pass = evaluate_structure(result)
            hallucination_pass = evaluate_hallucination(result)
            test_empty_retrieval_handling(agent, ollama_ok=True)
            print("Empty retrieval handling: PASS")
            decision_pass = structured_pass and hallucination_pass
            print("\n--- Model output (truncated) ---")
            print(json.dumps({k: result[k] for k in result if k != "meta"}, indent=2)[:2000])
        except (
            OllamaUnavailableError,
            ModelNotInstalledError,
            OllamaTimeoutError,
            MalformedResponseError,
            MissingEvidenceError,
        ) as exc:
            print(f"Generation: FAIL ({type(exc).__name__}: {exc})")
            if isinstance(exc, MalformedResponseError) and exc.raw_response:
                print("--- Raw model output (not hidden) ---")
                print(exc.raw_response[:4000])

    print("\n" + "=" * 40)
    print(f"Decision Agent: {'PASS' if decision_pass else 'FAIL'}")
    print(f"Ollama: {'PASS' if ollama_pass else 'FAIL'}")
    print(f"Model: {model_name}")
    print(f"Structured output: {'PASS' if structured_pass else 'FAIL'}")
    print(f"Hallucination guard: {'PASS' if hallucination_pass else 'FAIL'}")
    print("=" * 40)
    return 0 if decision_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
