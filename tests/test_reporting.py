"""Tests for reporting features: attempter profiling and financial impact."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import profile_attempters, financial_impact

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  -- {detail}")


def _make_result(attempter, determination, confidence=0.5, channels=None):
    """Build a minimal result dict for testing."""
    ch = channels or {}
    return {
        'attempter': attempter,
        'determination': determination,
        'confidence': confidence,
        'occupation': 'tester',
        'channel_details': {'channels': ch},
    }


def test_profile_empty():
    print("\n-- No attempter data returns empty --")
    result = profile_attempters([])
    check("empty input -> empty list", result == [])


def test_profile_below_min_submissions():
    print("\n-- Below min_submissions threshold --")
    results = [_make_result('alice', 'RED')]
    profiles = profile_attempters(results, min_submissions=2)
    check("single submission skipped", len(profiles) == 0)


def test_profile_ranks_by_flag_rate():
    print("\n-- Profiles ranked by flag rate descending --")
    results = [
        _make_result('alice', 'RED'),
        _make_result('alice', 'RED'),
        _make_result('alice', 'GREEN'),
        _make_result('bob', 'GREEN'),
        _make_result('bob', 'GREEN'),
        _make_result('bob', 'GREEN'),
        _make_result('carol', 'AMBER'),
        _make_result('carol', 'AMBER'),
    ]
    profiles = profile_attempters(results)
    check("3 profiles", len(profiles) == 3, f"got {len(profiles)}")
    if len(profiles) >= 2:
        check("carol first (100% flag rate)",
              profiles[0]['attempter'] == 'carol',
              f"got {profiles[0]['attempter']}")
        check("alice second (67% flag rate)",
              profiles[1]['attempter'] == 'alice',
              f"got {profiles[1]['attempter']}")
        check("bob last (0% flag rate)",
              profiles[2]['attempter'] == 'bob',
              f"got {profiles[2]['attempter']}")


def test_profile_fields():
    print("\n-- Profile dict has all fields --")
    results = [
        _make_result('alice', 'RED', 0.9),
        _make_result('alice', 'GREEN', 0.1),
    ]
    profiles = profile_attempters(results)
    check("profile returned", len(profiles) == 1)
    if profiles:
        p = profiles[0]
        check("has attempter", p['attempter'] == 'alice')
        check("total_submissions == 2", p['total_submissions'] == 2)
        check("flagged == 1", p['flagged'] == 1)
        check("flag_rate == 0.5", p['flag_rate'] == 0.5)
        check("red == 1", p['red'] == 1)
        check("green == 1", p['green'] == 1)
        check("mean_flagged_confidence == 0.9",
              p['mean_flagged_confidence'] == 0.9,
              f"got {p['mean_flagged_confidence']}")


def test_financial_zero_results():
    print("\n-- Financial impact with zero results --")
    impact = financial_impact([])
    check("total_submissions == 0", impact['total_submissions'] == 0)
    check("no division by zero", impact['flag_rate'] == 0.0)
    check("waste == 0", impact['waste_estimate'] == 0.0)


def test_financial_arithmetic():
    print("\n-- Financial impact arithmetic --")
    results = [
        _make_result('a', 'RED'),
        _make_result('a', 'RED'),
        _make_result('a', 'AMBER'),
        _make_result('a', 'GREEN'),
        _make_result('a', 'GREEN'),
    ]
    impact = financial_impact(results, cost_per_prompt=100.0)
    check("total == 5", impact['total_submissions'] == 5)
    check("total_spend == 500", impact['total_spend'] == 500.0)
    check("flagged == 3", impact['flagged_count'] == 3)
    check("flag_rate == 0.6", impact['flag_rate'] == 0.6)
    check("waste == 300", impact['waste_estimate'] == 300.0)
    check("clean_count == 2", impact['clean_count'] == 2)
    check("clean_yield == 0.4", impact['clean_yield'] == 0.4)
    check("annual_waste == 1200", impact['projected_annual_waste'] == 1200.0)
    check("annual_savings == 720", impact['projected_annual_savings_60pct'] == 720.0)


if __name__ == '__main__':
    print("=" * 70)
    print("  REPORTING TESTS")
    print("=" * 70)

    test_profile_empty()
    test_profile_below_min_submissions()
    test_profile_ranks_by_flag_rate()
    test_profile_fields()
    test_financial_zero_results()
    test_financial_arithmetic()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
