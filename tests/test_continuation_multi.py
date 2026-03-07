"""Tests for multi-truncation continuation analysis (FEAT 1, 2, 6)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import run_continuation_local_multi
from tests.conftest import AI_TEXT, HUMAN_TEXT

PASSED = 0
FAILED = 0

# Longer text for continuation analysis (needs >= 80 words)
LONG_AI_TEXT = (
    "This comprehensive analysis provides a thorough examination of the "
    "key factors that contribute to the overall effectiveness of the proposed "
    "framework. Furthermore, it is essential to note that the implementation "
    "of these strategies ensures alignment with best practices and industry "
    "standards. To address this challenge, we must consider multiple perspectives "
    "and leverage data-driven insights to achieve optimal outcomes. Additionally, "
    "this approach demonstrates the critical importance of systematic evaluation "
    "and evidence-based decision making in the modern landscape. The methodology "
    "presented herein encompasses a multi-dimensional assessment framework that "
    "integrates both quantitative metrics and qualitative indicators to provide "
    "a holistic understanding of the organizational dynamics at play."
)


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  -- {detail}")


def test_short_text_graceful():
    print("\n-- Short text returns gracefully --")
    result = run_continuation_local_multi("Hello world.")
    proxy = result.get('proxy_features', {})
    check("Returns dict", isinstance(result, dict))
    check("Has proxy_features", 'proxy_features' in result)
    check("composite_stability present", 'composite_stability' in proxy,
          f"keys: {list(proxy.keys())}")
    check("composite_variance present", 'composite_variance' in proxy)


def test_backward_compat_fields():
    print("\n-- Backward compatibility: all existing fields present --")
    result = run_continuation_local_multi(LONG_AI_TEXT)
    check("Has determination", 'determination' in result)
    check("Has confidence", 'confidence' in result)
    check("Has bscore", 'bscore' in result)
    proxy = result.get('proxy_features', {})
    check("Has composite", 'composite' in proxy)
    check("Has ncd", 'ncd' in proxy)
    check("Has internal_overlap", 'internal_overlap' in proxy)
    check("Has cond_surprisal", 'cond_surprisal' in proxy)


def test_multi_truncation_fields():
    print("\n-- Multi-truncation stability fields --")
    result = run_continuation_local_multi(LONG_AI_TEXT)
    proxy = result.get('proxy_features', {})
    check("multi_composites present", 'multi_composites' in proxy,
          f"keys: {list(proxy.keys())}")
    check("multi_composites has 3 entries",
          len(proxy.get('multi_composites', [])) == 3,
          f"got {len(proxy.get('multi_composites', []))}")
    check("composite_variance >= 0",
          proxy.get('composite_variance', -1) >= 0,
          f"got {proxy.get('composite_variance')}")
    check("composite_stability is float",
          isinstance(proxy.get('composite_stability', None), (int, float)),
          f"got type {type(proxy.get('composite_stability'))}")


def test_ncd_matrix_fields():
    print("\n-- NCD matrix fields (FEAT 6) --")
    result = run_continuation_local_multi(LONG_AI_TEXT)
    proxy = result.get('proxy_features', {})
    check("ncd_matrix_mean present", 'ncd_matrix_mean' in proxy)
    check("ncd_matrix_variance present", 'ncd_matrix_variance' in proxy)
    check("ncd_matrix_min present", 'ncd_matrix_min' in proxy)


def test_improvement_rate():
    print("\n-- Surprisal improvement rate (FEAT 2) --")
    result = run_continuation_local_multi(LONG_AI_TEXT)
    proxy = result.get('proxy_features', {})
    check("improvement_rate present", 'improvement_rate' in proxy,
          f"keys: {list(proxy.keys())}")
    check("surprisal_curve present", 'surprisal_curve' in proxy)


if __name__ == '__main__':
    print("=" * 70)
    print("  MULTI-TRUNCATION CONTINUATION TESTS")
    print("=" * 70)

    test_short_text_graceful()
    test_backward_compat_fields()
    test_multi_truncation_fields()
    test_ncd_matrix_fields()
    test_improvement_rate()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
