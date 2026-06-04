from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
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
