"""Tests for Chain-of-Thought leakage detection in preamble layer."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import run_preamble

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


def test_think_tags():
    print("\n-- Think tag detection --")
    score, sev, hits, _spans = run_preamble(
        "<think>\nThe user wants a pharmacist task...\n</think>\n"
        "You are a board-certified pharmacist."
    )
    check("<think> tag -> CRITICAL", sev == 'CRITICAL', f"got {sev}")
    check("score == 0.99", score == 0.99, f"got {score}")
    hit_names = [h[0] for h in hits]
    check("cot_leakage in hits", 'cot_leakage' in hit_names, f"got {hit_names}")


def test_reasoning_tags():
    print("\n-- Reasoning tag detection --")
    score, sev, hits, _spans = run_preamble(
        "<reasoning>I need to design a complex scenario</reasoning>\n"
        "You are a senior data analyst."
    )
    check("<reasoning> tag -> CRITICAL", sev == 'CRITICAL', f"got {sev}")


def test_think_tag_mid_document():
    print("\n-- Think tag mid-document --")
    score, sev, hits, _spans = run_preamble(
        "You are a pharmacist. <think>Let me design this carefully.</think> "
        "Review the following prescription orders."
    )
    check("Mid-doc <think> -> CRITICAL", sev == 'CRITICAL', f"got {sev}")


def test_self_correction():
    print("\n-- Self-correction phrases --")
    score, sev, hits, _spans = run_preamble(
        "Wait, actually let me rethink the constraints for this task. "
        "You must process each CSV row and validate all fields."
    )
    check("Self-correction -> HIGH", sev == 'HIGH', f"got {sev}")
    check("score == 0.75", score == 0.75, f"got {score}")


def test_hmm_self_correction():
    print("\n-- Hmm self-correction --")
    score, sev, hits, _spans = run_preamble(
        "Hmm, let me reconsider the approach here. "
        "The task should require the analyst to cross-reference two datasets."
    )
    check("'Hmm, let me' -> HIGH", sev == 'HIGH', f"got {sev}")


def test_final_answer():
    print("\n-- Final answer phrase --")
    score, sev, hits, _spans = run_preamble(
        "My final answer is the following task prompt: "
        "You are a compliance officer reviewing internal audit reports."
    )
    check("'My final answer is' -> HIGH", sev == 'HIGH', f"got {sev}")


def test_step_numbering():
    print("\n-- Step numbering (MEDIUM) --")
    score, sev, hits, _spans = run_preamble(
        "Step 1: Review the patient chart.\n"
        "Step 2: Identify medication interactions.\n"
        "Step 3: Document findings in the report."
    )
    check("Step numbering -> MEDIUM at most", sev in ('NONE', 'MEDIUM'),
          f"got {sev}")


def test_no_false_positive_think():
    print("\n-- No false positive on 'think carefully' --")
    score, sev, hits, _spans = run_preamble(
        "Think carefully about edge cases when reviewing patient records. "
        "You are a clinical pharmacist reviewing medication orders."
    )
    check("'Think carefully' -> no CoT hit", sev == 'NONE', f"got {sev}")


def test_no_false_positive_plain_task():
    print("\n-- No false positive on plain task prompt --")
    score, sev, hits, _spans = run_preamble(
        "You are a data analyst. Use the attached CSV to create a summary report. "
        "Include charts for quarterly revenue trends."
    )
    check("Plain task -> NONE", sev == 'NONE', f"got {sev}")


if __name__ == '__main__':
    print("=" * 70)
    print("  COT LEAKAGE PREAMBLE TESTS")
    print("=" * 70)

    test_think_tags()
    test_reasoning_tags()
    test_think_tag_mid_document()
    test_self_correction()
    test_hmm_self_correction()
    test_final_answer()
    test_step_numbering()
    test_no_false_positive_think()
    test_no_false_positive_plain_task()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
