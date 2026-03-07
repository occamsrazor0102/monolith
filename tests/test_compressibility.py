"""Tests for compressibility features (FEAT 5, 7)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import run_self_similarity, run_perplexity, HAS_PERPLEXITY
from tests.conftest import AI_TEXT, HUMAN_TEXT

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


# Long text for NSSI (needs >= 200 words)
LONG_TEXT = (
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
    "a holistic understanding of the organizational dynamics at play. Furthermore, "
    "the strategic implications of these findings suggest that a paradigm shift "
    "is necessary to accommodate the evolving requirements of contemporary "
    "stakeholder engagement models. The rigorous analytical procedures employed "
    "throughout this investigation have yielded robust and reproducible results "
    "that underscore the validity of our theoretical propositions. In conclusion, "
    "this examination reveals that the intersection of technological innovation "
    "and methodological rigor creates unprecedented opportunities for advancing "
    "organizational performance and operational excellence across diverse sectors "
    "and institutional contexts within the broader framework of global competition. "
    "Moreover, the comprehensive nature of this endeavor necessitates a thorough "
    "understanding of the underlying mechanisms that drive stakeholder value creation "
    "and sustainable organizational development in an increasingly complex environment."
)


def test_s13_fields_present():
    print("\n-- s13 structural compression delta fields --")
    # Short text: fields present with defaults
    result_short = run_self_similarity(AI_TEXT)
    check("short text: shuffled_comp_ratio present",
          'shuffled_comp_ratio' in result_short,
          f"keys: {sorted(result_short.keys())}")
    check("short text: structural_compression_delta present",
          'structural_compression_delta' in result_short)

    # Long text: s13 fully computed
    result = run_self_similarity(LONG_TEXT)
    check("long text: shuffled_comp_ratio present",
          'shuffled_comp_ratio' in result)
    check("long text: shuffled_comp_ratio > 0",
          result.get('shuffled_comp_ratio', 0) > 0,
          f"got {result.get('shuffled_comp_ratio')}")


def test_s13_deterministic():
    print("\n-- s13 is deterministic (seed=42) --")
    r1 = run_self_similarity(LONG_TEXT)
    r2 = run_self_similarity(LONG_TEXT)
    check("shuffled_comp_ratio stable across calls",
          r1['shuffled_comp_ratio'] == r2['shuffled_comp_ratio'],
          f"r1={r1['shuffled_comp_ratio']}, r2={r2['shuffled_comp_ratio']}")


def test_compression_perplexity_fields():
    print("\n-- Compression-perplexity divergence fields (FEAT 7) --")
    result = run_perplexity(AI_TEXT)
    check("comp_ratio present", 'comp_ratio' in result,
          f"keys: {sorted(result.keys())}")
    check("zlib_normalized_ppl present", 'zlib_normalized_ppl' in result)
    check("comp_ppl_ratio present", 'comp_ppl_ratio' in result)
    check("_token_losses present", '_token_losses' in result)

    if HAS_PERPLEXITY:
        check("comp_ratio > 0 when model available",
              result['comp_ratio'] > 0,
              f"got {result['comp_ratio']}")
        check("_token_losses is list",
              isinstance(result['_token_losses'], list))
    else:
        print("  (transformers/torch not installed -- skipping model tests)")
        check("comp_ratio == 0 when unavailable",
              result['comp_ratio'] == 0.0)
        check("_token_losses is empty list",
              result['_token_losses'] == [])


def test_short_text_compression_fields():
    print("\n-- Short text: compression-perplexity fields default --")
    result = run_perplexity("Hello world.")
    check("comp_ratio == 0", result['comp_ratio'] == 0.0,
          f"got {result['comp_ratio']}")
    check("zlib_normalized_ppl == 0", result['zlib_normalized_ppl'] == 0.0,
          f"got {result['zlib_normalized_ppl']}")
    check("_token_losses is empty", result['_token_losses'] == [],
          f"got {result['_token_losses']}")


if __name__ == '__main__':
    print("=" * 70)
    print("  COMPRESSIBILITY TESTS")
    print("=" * 70)

    test_s13_fields_present()
    test_s13_deterministic()
    test_compression_perplexity_fields()
    test_short_text_compression_fields()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
