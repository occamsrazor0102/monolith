"""Tests for individual analyzer modules."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import HAS_SEMANTIC, HAS_PERPLEXITY
from llm_detector_monolith import run_semantic_resonance
from llm_detector_monolith import run_perplexity
from llm_detector_monolith import run_token_cohesiveness, score_surprisal_windows
from tests.conftest import AI_TEXT, HUMAN_TEXT, CLINICAL_TEXT

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


def test_semantic_resonance():
    print("\n-- SEMANTIC RESONANCE --")

    short = "Hello world."
    r_short = run_semantic_resonance(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_SEMANTIC:
        r_ai = run_semantic_resonance(AI_TEXT)
        check("AI text: semantic_ai_score > 0", r_ai['semantic_ai_score'] > 0,
              f"got {r_ai['semantic_ai_score']}")
        check("AI text: semantic_delta > 0", r_ai['semantic_delta'] > 0,
              f"got {r_ai['semantic_delta']}")
        check("AI text: has determination", r_ai['determination'] is not None,
              f"got {r_ai['determination']}, delta={r_ai['semantic_delta']}")

        r_human = run_semantic_resonance(HUMAN_TEXT)
        check("Human text: lower ai_score", r_human['semantic_ai_score'] < r_ai['semantic_ai_score'],
              f"human={r_human['semantic_ai_score']}, ai={r_ai['semantic_ai_score']}")
    else:
        print("  (sentence-transformers not installed -- skipping model tests)")
        check("Unavailable: ai_score=0", r_short['semantic_ai_score'] == 0.0)


def test_perplexity():
    print("\n-- PERPLEXITY SCORING --")

    short = "Hello world."
    r_short = run_perplexity(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_PERPLEXITY:
        r_normal = run_perplexity(CLINICAL_TEXT)
        check("Normal text: perplexity > 0", r_normal['perplexity'] > 0,
              f"got {r_normal['perplexity']}")
        check("Normal text: has reason", len(r_normal.get('reason', '')) > 0)
    else:
        print("  (transformers/torch not installed -- skipping model tests)")
        check("Unavailable: perplexity=0", r_short['perplexity'] == 0.0)


def test_perplexity_diveye_fields():
    print("\n-- PERPLEXITY DIVEYE FIELDS --")

    short = "Hello world."
    r_short = run_perplexity(short)
    check("Short text: surprisal_variance present",
          'surprisal_variance' in r_short)
    check("Short text: volatility_decay present",
          'volatility_decay' in r_short)
    check("Short text: volatility_decay == 1.0",
          r_short['volatility_decay'] == 1.0,
          f"got {r_short['volatility_decay']}")
    check("Short text: n_tokens present",
          'n_tokens' in r_short)
    check("Short text: n_tokens == 0",
          r_short['n_tokens'] == 0,
          f"got {r_short['n_tokens']}")

    if HAS_PERPLEXITY:
        r_ai = run_perplexity(AI_TEXT)
        check("AI text: surprisal_variance > 0",
              r_ai['surprisal_variance'] > 0,
              f"got {r_ai['surprisal_variance']}")
        check("AI text: volatility_decay > 0",
              r_ai['volatility_decay'] > 0,
              f"got {r_ai['volatility_decay']}")
        check("AI text: n_tokens > 0",
              r_ai.get('n_tokens', 0) > 0)
    else:
        print("  (transformers/torch not installed -- skipping DivEye model tests)")


def test_feature_flags():
    print("\n-- FEATURE AVAILABILITY FLAGS --")
    check("HAS_SEMANTIC is bool", isinstance(HAS_SEMANTIC, bool))
    check("HAS_PERPLEXITY is bool", isinstance(HAS_PERPLEXITY, bool))
    print(f"    HAS_SEMANTIC={HAS_SEMANTIC}, HAS_PERPLEXITY={HAS_PERPLEXITY}")


def test_tocsin():
    print("\n-- TOKEN COHESIVENESS (TOCSIN) --")
    short = "Hello world."
    r_short = run_token_cohesiveness(short)
    check("Short text: cohesiveness == 0", r_short['cohesiveness'] == 0.0)
    check("Short text: determination is None", r_short['determination'] is None)
    check("Short text: n_rounds == 0", r_short['n_rounds'] == 0,
          f"got {r_short['n_rounds']}")

    if HAS_SEMANTIC:
        r_ai = run_token_cohesiveness(AI_TEXT)
        check("AI text: cohesiveness > 0", r_ai['cohesiveness'] > 0,
              f"got {r_ai['cohesiveness']}")
        check("AI text: n_rounds == 10", r_ai['n_rounds'] == 10,
              f"got {r_ai['n_rounds']}")
        check("AI text: cohesiveness_std present",
              'cohesiveness_std' in r_ai)
    else:
        print("  (sentence-transformers not installed -- skipping TOCSIN model tests)")
        check("Unavailable: cohesiveness == 0", r_short['cohesiveness'] == 0.0)


def test_surprisal_trajectory():
    print("\n-- SURPRISAL TRAJECTORY --")
    # Empty input
    r_empty = score_surprisal_windows([])
    check("Empty: trajectory_cv == 0", r_empty['surprisal_trajectory_cv'] == 0.0)
    check("Empty: stationarity == 0", r_empty['surprisal_stationarity'] == 0.0)
    check("Empty: n_windows == 0", r_empty['n_surprisal_windows'] == 0)

    # Too short
    r_short = score_surprisal_windows([0.5] * 10)
    check("Short: n_windows == 0", r_short['n_surprisal_windows'] == 0)

    # Uniform losses → high stationarity
    uniform = [1.0] * 256
    r_uniform = score_surprisal_windows(uniform)
    check("Uniform: n_windows > 0", r_uniform['n_surprisal_windows'] > 0,
          f"got {r_uniform['n_surprisal_windows']}")
    check("Uniform: trajectory_cv == 0 (constant)",
          r_uniform['surprisal_trajectory_cv'] == 0.0,
          f"got {r_uniform['surprisal_trajectory_cv']}")
    check("Uniform: stationarity == 1.0",
          r_uniform['surprisal_stationarity'] == 1.0,
          f"got {r_uniform['surprisal_stationarity']}")


if __name__ == '__main__':
    print("=" * 70)
    print("Analyzer Tests")
    print("=" * 70)

    test_feature_flags()
    test_semantic_resonance()
    test_perplexity()
    test_perplexity_diveye_fields()
    test_tocsin()
    test_surprisal_trajectory()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
