"""
Decision / Results Agent.

Local Llama 3.1 8B (Ollama) synthesizes evidence from upstream agents.
It does not segment, extract features, retrieve cases, or call cloud LLMs.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from prompts.decision_prompt import build_messages
from utils.config import load_config

logger = logging.getLogger("decision_agent")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

REQUIRED_OUTPUT_KEYS = (
    "summary",
    "segmentation_findings",
    "feature_findings",
    "retrieval_findings",
    "evidence_synthesis",
    "similar_cases",
    "limitations",
    "clinical_note",
)

REQUIRED_EVIDENCE_SECTIONS = (
    "patient_id",
    "imaging",
    "segmentation",
    "deep_features",
    "radiomics",
    "clinical_context",
    "retrieval",
    "fusion",
)


class DecisionAgentError(Exception):
    """Base error for the Decision / Results Agent."""


class OllamaUnavailableError(DecisionAgentError):
    """Ollama server is not reachable."""


class ModelNotInstalledError(DecisionAgentError):
    """Configured Ollama model is not installed locally."""


class OllamaTimeoutError(DecisionAgentError):
    """Ollama request timed out."""


class MalformedResponseError(DecisionAgentError):
    """Model output could not be parsed or failed schema validation."""

    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message)
        self.raw_response = raw_response


class MissingEvidenceError(DecisionAgentError):
    """Structured evidence object is missing or incomplete."""


class DecisionResultsAgent:
    """Evidence-synthesis agent backed by a local Ollama Llama 3.1 8B model."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ):
        if config is None:
            loaded = load_config(config_path or DEFAULT_CONFIG_PATH)
            config = loaded.get("decision_agent", loaded)

        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.provider = str(config.get("provider", "ollama"))
        self.model = str(config.get("model", "llama3.1:8b"))
        self.host = str(config.get("host", "http://127.0.0.1:11434")).rstrip("/")
        self.temperature = float(config.get("temperature", 0.1))
        self.max_tokens = int(config.get("max_tokens", 1024))
        self.timeout_seconds = float(config.get("timeout_seconds", 120))

        if self.provider != "ollama":
            raise DecisionAgentError(
                f"Unsupported provider '{self.provider}'. "
                "Only local Ollama is allowed; cloud APIs are disabled."
            )

        logger.info(
            "DecisionResultsAgent initialized (provider=%s, model=%s, host=%s)",
            self.provider,
            self.model,
            self.host,
        )

    def check_ollama(self) -> dict[str, Any]:
        """Verify Ollama is reachable and the configured model is installed."""
        tags = self._request_json("GET", "/api/tags", payload=None)
        names = [m.get("name", "") for m in tags.get("models", [])]
        installed = self._model_is_installed(names)
        logger.info("Ollama reachable. Installed models: %s", names)
        if not installed:
            raise ModelNotInstalledError(
                f"Ollama is running but model '{self.model}' is not installed. "
                f"Installed models: {names or '[none]'}. "
                f"Install with: ollama pull {self.model}"
            )
        return {"reachable": True, "models": names, "selected": self.model}

    def synthesize(self, evidence: dict[str, Any] | None) -> dict[str, Any]:
        """Call Llama 3.1 8B to synthesize a structured explanation from evidence."""
        if not self.enabled:
            raise DecisionAgentError("Decision agent is disabled in configuration.")

        validated_evidence = self._validate_evidence(evidence)
        known_case_ids = self._known_case_ids(validated_evidence)
        empty_retrieval = not known_case_ids
        if empty_retrieval:
            logger.warning("Retrieval results are empty; synthesis will state that explicitly.")

        self.check_ollama()
        messages = build_messages(validated_evidence)
        raw = self._generate(messages["system"], messages["user"])
        parsed = extract_json_object(raw)
        result = self._validate_output(parsed, raw)
        result["similar_cases"] = self._filter_similar_cases(
            result["similar_cases"], known_case_ids, empty_retrieval
        )
        self._apply_language_guard(result)
        result["meta"] = {
            "provider": self.provider,
            "model": self.model,
            "empty_retrieval": empty_retrieval,
        }
        logger.info("Decision agent synthesis completed for patient_id=%s", validated_evidence.get("patient_id"))
        return result

    def _validate_evidence(self, evidence: dict[str, Any] | None) -> dict[str, Any]:
        if not evidence or not isinstance(evidence, dict):
            raise MissingEvidenceError("Structured evidence object is missing.")

        missing = [key for key in REQUIRED_EVIDENCE_SECTIONS if key not in evidence]
        if missing:
            raise MissingEvidenceError(
                "Structured evidence is incomplete. Missing keys: " + ", ".join(missing)
            )
        return evidence

    def _known_case_ids(self, evidence: dict[str, Any]) -> list[str]:
        retrieval = evidence.get("retrieval") or {}
        cases = retrieval.get("similar_cases") or []
        ids: list[str] = []
        for case in cases:
            if isinstance(case, dict) and case.get("case_id"):
                ids.append(str(case["case_id"]))
        return ids

    def _model_is_installed(self, names: list[str]) -> bool:
        target = self.model.lower()
        for name in names:
            n = name.lower()
            if n == target or n.startswith(target + ":") or target.startswith(n + ":"):
                return True
            if n.split(":")[0] == target.split(":")[0] and (
                "8b" in n and "8b" in target
            ):
                return True
        return False

    def _generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        body = self._request_json("POST", "/api/generate", payload)
        if body.get("error"):
            error_text = str(body["error"])
            if "not found" in error_text.lower():
                raise ModelNotInstalledError(error_text)
            raise DecisionAgentError(error_text)
        response = body.get("response", "")
        if not isinstance(response, str) or not response.strip():
            raise MalformedResponseError("Ollama returned an empty response.", raw_response=str(body))
        return response

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = f"{self.host}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.error("Ollama HTTP error %s for %s: %s", exc.code, url, detail)
            if exc.code == 404:
                raise ModelNotInstalledError(
                    f"Ollama returned 404 for {path}. Model may not be installed: {detail}"
                ) from exc
            raise DecisionAgentError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout) as exc:
            logger.error("Ollama timeout after %ss calling %s", self.timeout_seconds, url)
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self.timeout_seconds} seconds."
            ) from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower():
                raise OllamaTimeoutError(
                    f"Ollama request timed out after {self.timeout_seconds} seconds."
                ) from exc
            logger.error("Ollama unavailable at %s: %s", url, exc)
            raise OllamaUnavailableError(
                f"Ollama is unavailable at {self.host}. "
                "Start Ollama locally. Cloud APIs are not used as a fallback."
            ) from exc
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise MalformedResponseError("Ollama returned non-JSON HTTP body.", raw_response=raw) from exc
        if not isinstance(parsed, dict):
            raise MalformedResponseError("Ollama JSON body was not an object.", raw_response=raw)
        return parsed

    def _validate_output(self, parsed: dict[str, Any], raw: str) -> dict[str, Any]:
        missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in parsed]
        if missing:
            raise MalformedResponseError(
                "Model JSON is missing required keys: " + ", ".join(missing),
                raw_response=raw,
            )

        text_keys = [k for k in REQUIRED_OUTPUT_KEYS if k != "similar_cases"]
        for key in text_keys:
            if not isinstance(parsed[key], str) or not parsed[key].strip():
                raise MalformedResponseError(
                    f"Field '{key}' must be a non-empty string.",
                    raw_response=raw,
                )

        similar = parsed["similar_cases"]
        if not isinstance(similar, list):
            raise MalformedResponseError(
                "Field 'similar_cases' must be a list.",
                raw_response=raw,
            )
        normalized = []
        for item in similar:
            if not isinstance(item, dict) or "case_id" not in item or "reason" not in item:
                raise MalformedResponseError(
                    "Each similar_cases item must contain case_id and reason.",
                    raw_response=raw,
                )
            normalized.append(
                {"case_id": str(item["case_id"]), "reason": str(item["reason"])}
            )
        parsed["similar_cases"] = normalized
        return parsed

    def _filter_similar_cases(
        self,
        similar_cases: list[dict[str, str]],
        known_ids: list[str],
        empty_retrieval: bool,
    ) -> list[dict[str, str]]:
        if empty_retrieval:
            if similar_cases:
                logger.warning(
                    "Model listed similar cases despite empty retrieval; dropping invented IDs: %s",
                    [c["case_id"] for c in similar_cases],
                )
            return []

        known = set(known_ids)
        kept = []
        dropped = []
        for item in similar_cases:
            if item["case_id"] in known:
                kept.append(item)
            else:
                dropped.append(item["case_id"])
        if dropped:
            logger.warning("Dropped similar_cases IDs not present in evidence: %s", dropped)
        return kept

    def _apply_language_guard(self, result: dict[str, Any]) -> None:
        combined = " ".join(
            str(result.get(k, ""))
            for k in REQUIRED_OUTPUT_KEYS
            if k != "similar_cases"
        ).lower()
        forbidden = (
            "definitely has cancer",
            "confirmed diagnosis of",
            "autonomous treatment",
            "start chemotherapy",
            "i recommend treatment",
        )
        hits = [phrase for phrase in forbidden if phrase in combined]
        if hits:
            raise MalformedResponseError(
                "Response used forbidden diagnostic/treatment language: " + ", ".join(hits),
                raw_response=json.dumps(result),
            )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from model text without hiding the raw payload on failure."""
    if not text or not str(text).strip():
        raise MalformedResponseError("Empty model response.", raw_response=text)

    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        errors.append(f"Parsed JSON type was {type(parsed).__name__}, expected object.")

    raise MalformedResponseError(
        "Could not extract a JSON object from the model response. "
        "Parse errors: " + " | ".join(errors),
        raw_response=text,
    )
