import json

import pytest

from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models import ModelReply, ResponsesHistory
from parallel_search_evals.results import EvaluationSummary, Grade, ScoredAnswer, read_answer, write_answer


class TestGrade:
    @pytest.mark.parametrize("score,correct", [(1.0, True), (0.0, False), (2 / 3, False)])
    def test_explicit_official_score_and_strict_correctness(self, score, correct):
        grade = Grade(score=score, correct=correct, grade_text='{"fixture":true}')
        assert grade.model_dump() == {"score": score, "correct": correct, "grade_text": '{"fixture":true}'}

    @pytest.mark.parametrize("invalid", [True, "0.5", float("nan"), float("inf"), -0.1, 1.1])
    def test_invalid_persisted_score_is_rejected(self, invalid):
        with pytest.raises(ValueError):
            Grade(score=invalid, correct=False, grade_text='{"fixture":true}')

    @pytest.mark.parametrize("invalid", ["true", 1, None])
    def test_nonboolean_correctness_is_rejected(self, invalid):
        with pytest.raises(ValueError):
            Grade(score=0.5, correct=invalid, grade_text='{"fixture":true}')


def test_record_with_unrecognized_grade_fields_is_rejected(tmp_path):
    path = tmp_path / "invalid-grade.json"
    path.write_text(
        '{"status":"completed","datum":{"id":"q","problem":"Q","answer":"A"},'
        '"convo":{"assistant_text":"","messages":[]},"grader_result":{"grade_type":1,'
        '"problem":"Q","answer":"A","response":"A","grade_text":"{}","score_value":1}}'
    )
    assert read_answer(path) is None


class TestEvaluationSummary:
    def test_summary_keeps_denominator_and_correct_count(self):
        summary = EvaluationSummary(score=0.375, total_samples=8, total_correct=2, failed_samples=1)
        assert summary.model_dump() == {"score": 0.375, "total_samples": 8, "total_correct": 2, "failed_samples": 1}
        assert EvaluationSummary(score=0.5, total_samples=2, total_correct=1).failed_samples == 0

    def test_summary_is_written_next_to_the_run(self, tmp_path):
        run_dir = tmp_path / "survey-alpha"
        summary = EvaluationSummary(score=0.625, total_samples=4, total_correct=2, failed_samples=1)
        summary.save(run_dir)
        saved = tmp_path / "results" / "survey-alpha.json"
        assert json.loads(saved.read_text()) == summary.model_dump()
        assert not (run_dir / "results").exists()

    def test_summary_creates_its_parent_directory(self, tmp_path):
        run_dir = tmp_path / "uncreated" / "campaign" / "survey-beta"
        assert not run_dir.parent.exists()
        summary = EvaluationSummary(score=0.125, total_samples=8, total_correct=1)
        summary.save(run_dir)
        assert json.loads((run_dir.parent / "results" / "survey-beta.json").read_text()) == summary.model_dump()


class TestAnswerRecord:
    @pytest.fixture
    def completed_history(self):
        convo = ResponsesHistory(assistant_text="", system="Use the coastal register.").append_question(
            "Name the signal tower."
        )
        convo.accept_reply(
            ModelReply(
                texts=["The copper tower."],
                tool_calls=[],
                native=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The copper tower."}],
                    },
                    {"type": "future_record", "opaque": {"measurements": [None, 2.75, False], "key": "sector-kappa"}},
                ],
            )
        )
        return convo

    @pytest.fixture
    def scored_record(self, completed_history):
        return ScoredAnswer(
            datum=Question(
                id="tower-83",
                problem="Name the signal tower.",
                answer="The copper tower.",
                metadata={"coast": "west", "survey": "autumn-43"},
            ),
            convo=completed_history,
            grader_result=Grade(
                score=0.375, correct=False, grade_text='{"credit":0.375,"reason":"partial identification"}'
            ),
        )

    def test_record_roundtrip_keeps_fractional_credit(self, scored_record, tmp_path):
        path = tmp_path / "fractional.json"
        write_answer(scored_record, path)
        restored = read_answer(path)
        assert isinstance(restored, ScoredAnswer)
        assert restored.grader_result.score == 0.375
        assert restored.grader_result.correct is False
        assert restored.grader_result.grade_text == scored_record.grader_result.grade_text

    def test_answer_write_creates_complete_terminal_json(self, scored_record, tmp_path):
        path = tmp_path / "terminal.json"
        assert not path.exists()
        write_answer(scored_record, path)
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["status"] == "completed"
        assert set(payload) == {"status", "datum", "convo", "grader_result"}
        assert payload == scored_record.model_dump()

    def test_answer_read_restores_the_scored_record(self, scored_record, tmp_path):
        path = tmp_path / "tower-result.json"
        write_answer(scored_record, path)
        restored = read_answer(path)
        assert isinstance(restored, ScoredAnswer)
        assert restored == scored_record

    def test_missing_answer_has_no_terminal_record(self, tmp_path):
        assert read_answer(tmp_path / "not-recorded.json") is None

    @pytest.mark.parametrize("content", ["", '{"status":', "not a task record"])
    def test_invalid_answer_json_is_not_admitted(self, tmp_path, content):
        path = tmp_path / "broken.json"
        path.write_text(content)
        assert read_answer(path) is None

    @pytest.mark.parametrize(
        "change",
        [
            "empty",
            "unknown-status",
            "missing-grade",
            "invalid-grade",
            "missing-status",
            "no-current-answer",
            "no-native-history",
        ],
    )
    def test_invalid_answer_shape_is_not_admitted(self, scored_record, tmp_path, change):
        payload = scored_record.model_dump()
        if change == "empty":
            payload = {}
        elif change == "unknown-status":
            payload["status"] = "still-running"
        elif change == "missing-grade":
            del payload["grader_result"]
        elif change == "invalid-grade":
            payload["grader_result"]["score"] = "0.375"
        elif change == "missing-status":
            del payload["status"]
        elif change == "no-current-answer":
            payload["convo"]["assistant_text"] = ""
        else:
            payload["convo"]["messages"] = []
        path = tmp_path / "wrong-schema.json"
        path.write_text(json.dumps(payload))
        assert read_answer(path) is None

    def test_question_metadata_survives_terminal_storage(self, scored_record, tmp_path):
        path = tmp_path / "metadata.json"
        write_answer(scored_record, path)
        restored = read_answer(path)
        assert isinstance(restored, ScoredAnswer)
        assert restored.datum.model_dump() == scored_record.datum.model_dump()
        assert restored.datum.metadata == {"coast": "west", "survey": "autumn-43"}

    def test_native_history_survives_terminal_storage(self, scored_record, completed_history, tmp_path):
        path = tmp_path / "native-history.json"
        write_answer(scored_record, path)
        restored = read_answer(path)
        assert isinstance(restored, ScoredAnswer)
        assert isinstance(restored.convo, ResponsesHistory)
        assert restored.convo.model_dump() == completed_history.model_dump()
        assert restored.convo.request_messages() == completed_history.request_messages()
        assert restored.convo.assistant_text == "The copper tower."
