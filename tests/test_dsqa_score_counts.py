"""Official F1 arithmetic is unchanged across empty and fractional matches."""

from parallel_search_evals.benchmarks.dsqa import dsqa_score


def test_f1_preserves_reference_arithmetic_for_answer_counts():
    for expected in range(1, 13):
        for found in range(expected + 1):
            for extra in range(13):
                precision = found / (found + extra) if found + extra else 0.0
                recall = found / expected
                reference = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                grade = {
                    "Answer Correctness": {
                        "Explanation": "Count characterization",
                        "Correctness Details": {f"part-{i}": i < found for i in range(expected)},
                        "Excessive Answers": [f"extra-{i}" for i in range(extra)],
                    }
                }
                assert dsqa_score(grade) == (reference, found == expected and extra == 0)
