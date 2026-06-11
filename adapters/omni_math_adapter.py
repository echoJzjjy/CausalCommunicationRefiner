from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from typing import Any


@dataclass
class OmniMathRecord:
    problem: str
    answer: str
    solution: str = ""
    domain: str = ""
    difficulty: str = ""
    source: str = ""


def raw_answer_text(raw_answer: Any) -> str:
    if isinstance(raw_answer, list):
        return str(raw_answer[0]) if raw_answer else ""
    return str(raw_answer)


def normalize_math_text(text: str) -> str:
    value = str(text).strip()
    if not value:
        return ""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", value)
    if boxed:
        value = boxed[-1]
    value = value.replace("$", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(" ", "")
    value = value.replace(",", "")
    value = value.replace("\n", "")
    value = value.strip(".")
    return value


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
SIMPLE_EQUATION_RE = re.compile(r"(?<![A-Za-z])([-+*/().\d\s]{1,80})=([-+*/().\d\s]{1,80})(?![A-Za-z])")


def parse_number(value: Any) -> float | None:
    text = normalize_math_text(str(value))
    if not text:
        return None
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", text):
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except ZeroDivisionError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def extract_last_number(text: str) -> str:
    matches = NUMBER_RE.findall(str(text))
    return matches[-1].replace(",", "") if matches else ""


def extract_math_candidate(raw_answer: Any) -> str:
    text = raw_answer_text(raw_answer)
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1]
    match = re.search(r"(?:final answer|answer)\s*(?:is|:)?\s*(.+)", text, re.I)
    if match:
        candidate = extract_last_number(match.group(1))
        if candidate:
            return candidate
    return extract_last_number(text)


def sympy_equivalent(a: Any, b: Any) -> bool:
    try:
        import sympy as sp
    except Exception:
        return False
    try:
        left = sp.sympify(normalize_math_text(str(a)))
        right = sp.sympify(normalize_math_text(str(b)))
        return bool(sp.simplify(left - right) == 0)
    except Exception:
        return False


def safe_arithmetic_eval(expr: str) -> float | None:
    if not re.fullmatch(r"[-+*/().\d\s]+", expr):
        return None
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - restricted to arithmetic characters above.
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def equation_consistency_score(text: str, max_equations: int = 8) -> float:
    checked = 0
    valid = 0
    for left, right in SIMPLE_EQUATION_RE.findall(str(text)):
        left_value = safe_arithmetic_eval(left)
        right_value = safe_arithmetic_eval(right)
        if left_value is None or right_value is None:
            continue
        checked += 1
        valid += int(math.isclose(left_value, right_value, rel_tol=1e-6, abs_tol=1e-6))
        if checked >= max_equations:
            break
    return valid / checked if checked else 0.0


def _field_text(value: Any) -> str:
    if isinstance(value, list):
        return " > ".join(str(item) for item in value if str(item).strip())
    return str(value)


def load_omni_math_records(split: str = "train", local_path: Path | None = None) -> list[OmniMathRecord]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        load_dataset = None  # type: ignore[assignment]

    dataset = None
    if load_dataset is not None:
        try:
            dataset = load_dataset("Heng1999/Omni-MATH-512", split=split)
        except Exception:
            dataset = None
    if dataset is None:
        if local_path is None:
            raise RuntimeError(
                "Omni-MATH-512 is not cached locally and the Hugging Face download path is unavailable. "
                "Provide `local_path` to a json/jsonl snapshot if you want offline smoke runs."
            )
        with local_path.open(encoding="utf-8") as f:
            if local_path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                rows = json.load(f)
    else:
        rows = list(dataset)

    records: list[OmniMathRecord] = []
    for row in rows:
        records.append(
            OmniMathRecord(
                problem=_field_text(row.get("problem") or row.get("question") or row.get("prompt") or ""),
                answer=str(row.get("answer") or row.get("final_answer") or row.get("target") or ""),
                solution=_field_text(row.get("solution", "")),
                domain=_field_text(row.get("domain", "")),
                difficulty=_field_text(row.get("difficulty", "")),
                source=_field_text(row.get("source", "")),
            )
        )
    return records


class OmniMathAdapter:
    def __init__(self, args: Any, local_path: Path | None = None) -> None:
        self.name = "omni"
        self.graph_domain = "gsm8k"
        self.agent_names = ["MathSolver"]
        self.agent_nums = [4]
        self.decision_method = "FinalRefer"
        self.kind = "freeform_math"
        self.drop_remainder = False
        self.note = "Omni-MATH-512 exposes a single train split; we treat the prefix as calibration and the suffix as held-out evaluation."
        self.trace_code_timeout = getattr(args, "trace_code_timeout", 2)
        self.final_code_timeout = getattr(args, "final_code_timeout", 100)
        self.train_records: list[Any] = []
        self.eval_records = load_omni_math_records("train", local_path=local_path)

    def input_for(self, record: OmniMathRecord) -> dict[str, str]:
        return {"task": record.problem}

    def train_input_for(self, record: OmniMathRecord) -> dict[str, str]:
        return self.input_for(record)

    def target_for(self, record: OmniMathRecord, train: bool = False) -> str:
        return normalize_math_text(record.answer)

    def postprocess_final(self, raw_answer: Any, record: Any | None = None) -> str:
        return normalize_math_text(raw_answer_text(raw_answer))

    def postprocess_train(self, raw_answer: Any, record: Any | None = None) -> str:
        return self.postprocess_final(raw_answer, record)

    def is_correct(self, prediction: Any, target: Any) -> bool:
        return normalize_math_text(str(prediction)) == normalize_math_text(str(target))

    def quality_score(self, prediction: Any, target: Any, raw_answer: Any | None = None, record: Any | None = None) -> float:
        if self.is_correct(prediction, target):
            return 1.0
        raw_text = raw_answer_text(raw_answer)
        candidate = extract_math_candidate(raw_answer)
        format_score = 1.0 if parse_number(candidate) is not None else 0.0
        symbolic_score = 1.0 if sympy_equivalent(candidate, target) else 0.0
        equation_score = equation_consistency_score(raw_text)
        final_marker_score = 1.0 if re.search(r"boxed|final answer|answer is|the answer is", raw_text, re.I) else 0.0
        return min(
            0.95,
            0.35 * symbolic_score
            + 0.2 * equation_score
            + 0.1 * format_score
            + 0.05 * final_marker_score,
        )
