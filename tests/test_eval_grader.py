import pandas as pd

from evals.assessment_extraction.grader import grade_measures


def _df(rows):
    return pd.DataFrame(rows)


def test_perfect_match_scores_all_ones():
    expected = [
        {"battery": "WAIS-IV", "domain": "General Intellectual Functioning",
         "test": "Vocabulary", "metric": "raw score", "value": "42"},
        {"battery": "WAIS-IV", "domain": "General Intellectual Functioning",
         "test": "Vocabulary", "metric": "age scaled score", "value": "12"},
    ]
    extracted = _df(expected)

    grade = grade_measures(expected, extracted)

    assert grade["recall"] == 1.0
    assert grade["precision"] == 1.0
    assert grade["field_accuracy"] == 1.0
    assert grade["n_matched"] == 2


def test_missed_measure_lowers_recall_not_precision():
    expected = [
        {"battery": "WAIS-IV", "domain": "Memory", "test": "Vocabulary", "metric": "raw score", "value": "42"},
        {"battery": "WAIS-IV", "domain": "Memory", "test": "Digit Span", "metric": "raw score", "value": "26"},
    ]
    extracted = _df([expected[0]])

    grade = grade_measures(expected, extracted)

    assert grade["recall"] == 0.5
    assert grade["precision"] == 1.0


def test_hallucinated_measure_lowers_precision_not_recall():
    expected = [
        {"battery": "WAIS-IV", "domain": "Memory", "test": "Vocabulary", "metric": "raw score", "value": "42"},
    ]
    extracted = _df(
        [
            expected[0],
            {"battery": "WAIS-IV", "domain": "Memory", "test": "Made Up Test", "metric": "raw score", "value": "99"},
        ]
    )

    grade = grade_measures(expected, extracted)

    assert grade["recall"] == 1.0
    assert grade["precision"] == 0.5


def test_value_normalization_ignores_prorated_marker_and_trailing_zero():
    expected = [{"battery": "WAIS-IV", "domain": "d", "test": "Verbal", "metric": "sum of scaled scores", "value": "35*"}]
    extracted = _df([{"battery": "WAIS-IV", "domain": "d", "test": "Verbal", "metric": "sum of scaled scores", "value": "35"}])

    grade = grade_measures(expected, extracted)
    assert grade["recall"] == 1.0

    expected2 = [{"battery": "b", "domain": "d", "test": "IQ", "metric": "m", "value": "97"}]
    extracted2 = _df([{"battery": "b", "domain": "d", "test": "IQ", "metric": "m", "value": "97.0"}])
    assert grade_measures(expected2, extracted2)["recall"] == 1.0


def test_test_name_matching_is_case_and_whitespace_insensitive():
    expected = [{"battery": "b", "domain": "d", "test": "Full  scale IQ", "metric": "m", "value": "117"}]
    extracted = _df([{"battery": "b", "domain": "d", "test": "full scale iq", "metric": "m", "value": "117"}])

    assert grade_measures(expected, extracted)["recall"] == 1.0


def test_field_accuracy_only_counts_matched_rows_and_checks_domain_battery_metric():
    expected = [
        {"battery": "WAIS-IV", "domain": "Memory", "test": "Vocabulary", "metric": "raw score", "value": "42"},
    ]
    # matched on (test, value), but domain is wrong
    extracted = _df(
        [{"battery": "WAIS-IV", "domain": "Language", "test": "Vocabulary", "metric": "raw score", "value": "42"}]
    )

    grade = grade_measures(expected, extracted)
    assert grade["recall"] == 1.0
    assert grade["field_accuracy"] == 0.0


def test_duplicate_expected_rows_require_separate_matches():
    expected = [
        {"battery": "b", "domain": "d", "test": "Errors", "metric": "count", "value": "0"},
        {"battery": "b", "domain": "d", "test": "Errors", "metric": "count", "value": "0"},
    ]
    extracted_one = _df([expected[0]])
    grade_one = grade_measures(expected, extracted_one)
    assert grade_one["n_matched"] == 1
    assert grade_one["recall"] == 0.5

    extracted_two = _df(expected)
    grade_two = grade_measures(expected, extracted_two)
    assert grade_two["n_matched"] == 2
    assert grade_two["recall"] == 1.0


def test_empty_expected_and_empty_extraction_edge_cases():
    assert grade_measures([], pd.DataFrame())["recall"] == 1.0
    assert grade_measures([], pd.DataFrame())["precision"] == 1.0

    expected = [{"battery": "b", "domain": "d", "test": "t", "metric": "m", "value": "1"}]
    grade = grade_measures(expected, pd.DataFrame())
    assert grade["recall"] == 0.0
    assert grade["precision"] == 0.0
