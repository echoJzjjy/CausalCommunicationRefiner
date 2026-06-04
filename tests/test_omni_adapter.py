from __future__ import annotations

import json
from pathlib import Path

from adapters.omni_math_adapter import OmniMathAdapter, load_omni_math_records, normalize_math_text


class _Args:
    trace_code_timeout = 2
    final_code_timeout = 100


def test_load_omni_math_records_from_local_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "omni.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "problem": "Compute 2+2.",
                        "answer": r"\\boxed{4}",
                        "solution": "By inspection.",
                        "domain": "arithmetic",
                        "difficulty": 1.0,
                        "source": "toy",
                    }
                ),
                json.dumps(
                    {
                        "question": "Find x.",
                        "final_answer": "7",
                        "solution": "x=7",
                        "domain": "algebra",
                        "difficulty": 2.0,
                        "source": "toy",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    records = load_omni_math_records(local_path=path)
    assert len(records) == 2
    assert records[0].problem == "Compute 2+2."
    assert records[0].answer == r"\\boxed{4}"
    assert records[1].problem == "Find x."
    assert records[1].answer == "7"


def test_omni_adapter_normalizes_answers(tmp_path: Path) -> None:
    path = tmp_path / "fixtures_omni.jsonl"
    path.write_text(json.dumps({"problem": "P", "answer": "4"}), encoding="utf-8")
    adapter = OmniMathAdapter(_Args(), local_path=path)
    assert normalize_math_text(r"\boxed{ 4 }") == "4"
    assert adapter.is_correct("4", "4")
