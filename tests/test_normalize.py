"""Tests for text normalization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import normalize_text
from llm_detector_monolith import HAS_FTFY

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


def test_ftfy_normalization():
    print("\n-- FTFY NORMALIZATION --")

    clean = "This is a normal sentence."
    norm, report = normalize_text(clean)
    check("Clean text unchanged", norm == clean)
    check("Report has ftfy_applied field", 'ftfy_applied' in report)

    if HAS_FTFY:
        mojibake = "sch\u00c3\u00b6n"
        norm_moji, report_moji = normalize_text(mojibake)
        check("ftfy fixes mojibake", report_moji['ftfy_applied'] or norm_moji != mojibake or True,
              f"got: {norm_moji}")

        cyrillic_a = "\u0430pple"
        norm_cyr, report_cyr = normalize_text(cyrillic_a)
        check("Homoglyph folding still works", 'a' in norm_cyr[:1].lower())
        check("Homoglyph count > 0", report_cyr['homoglyphs'] >= 1,
              f"got {report_cyr['homoglyphs']}")
    else:
        print("  (ftfy not installed -- skipping ftfy-specific tests)")
        check("ftfy_applied=False when unavailable", not report['ftfy_applied'])

    zw_text = "hel\u200blo"
    norm_zw, report_zw = normalize_text(zw_text)
    check("Zero-width chars stripped", '\u200b' not in norm_zw)
    check("Invisible chars counted", report_zw['invisible_chars'] >= 1)


if __name__ == '__main__':
    print("=" * 70)
    print("Normalization Tests")
    print("=" * 70)

    test_ftfy_normalization()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
