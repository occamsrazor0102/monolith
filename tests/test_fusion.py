"""Tests for the fusion/determine module and channel scoring."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import determine
from llm_detector_monolith import score_stylometric

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


def test_stylometry_integration():
    print("\n-- STYLOMETRY INTEGRATION (semantic + perplexity) --")

    ch_none = score_stylometric(0, None, semantic=None, ppl=None)
    check("No signals -> GREEN", ch_none.severity == 'GREEN')

    l30_r = {'determination': 'RED', 'nssi_score': 0.8, 'nssi_signals': 7, 'confidence': 0.85}
    ch_r = score_stylometric(0, l30_r, semantic=None, ppl=None)
    check("NSSI RED still works", ch_r.severity == 'RED')

    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    ch_sem = score_stylometric(0, None, semantic=l28_amber, ppl=None)
    check("Semantic AMBER alone -> AMBER", ch_sem.severity == 'AMBER',
          f"got {ch_sem.severity}")
    check("Semantic in sub_signals", 'semantic_delta' in ch_sem.sub_signals)

    ppl_yellow = {
        'determination': 'YELLOW', 'perplexity': 22.0, 'confidence': 0.30,
    }
    ch_ppl = score_stylometric(0, None, semantic=None, ppl=ppl_yellow)
    check("PPL YELLOW alone -> YELLOW", ch_ppl.severity == 'YELLOW',
          f"got {ch_ppl.severity}")
    check("Perplexity in sub_signals", 'perplexity' in ch_ppl.sub_signals)

    ch_boost = score_stylometric(0, l30_r, semantic=l28_amber, ppl=None)
    check("NSSI+Semantic boost > NSSI alone", ch_boost.score > ch_r.score,
          f"boost={ch_boost.score}, alone={ch_r.score}")


def test_determine_with_new_signals():
    print("\n-- DETERMINE WITH NEW SIGNALS --")

    l25_low = {'composite': 0.05, 'framing_completeness': 0}
    l26_none = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                'spec_score': 0, 'contractions': 5, 'hedges': 3}
    l27_none = {'idi': 2.0}

    det, _, _, _ = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                             mode='generic_aigt', semantic=None, ppl=None)
    check("No new signals -> GREEN", det == 'GREEN', f"got {det}")

    l28_amber = {
        'determination': 'AMBER', 'semantic_ai_mean': 0.70,
        'semantic_delta': 0.20, 'confidence': 0.55,
    }
    det2, _, _, cd = determine(0, 'NONE', l25_low, l26_none, l27_none, 300,
                                mode='generic_aigt', semantic=l28_amber, ppl=None)
    check("Semantic AMBER -> AMBER in generic_aigt",
          det2 in ('AMBER', 'RED'), f"got {det2}")

    check("4 channels in details", len(cd.get('channels', {})) == 4,
          f"got {len(cd.get('channels', {}))}")


if __name__ == '__main__':
    print("=" * 70)
    print("Fusion / Channel Scoring Tests")
    print("=" * 70)

    test_stylometry_integration()
    test_determine_with_new_signals()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
