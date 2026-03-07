"""Tests for lexicon pack scoring and pipeline integration."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import (
    PACK_REGISTRY, score_pack, score_packs,
    get_packs_for_layer, get_packs_for_mode,
    get_total_constraint_score, get_total_schema_score,
    get_total_exec_spec_score, get_category_score,
)
from llm_detector_monolith import (
    run_prompt_signature_enhanced,
    run_voice_dissonance_enhanced,
    run_instruction_density_enhanced,
)
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


CONSTRAINT_TEXT = (
    "You MUST process each CSV row. If the field is null, leave blank and "
    "flag as MISSING. Each record SHALL contain a valid patient_id. "
    "Do not include any fields that are not REQUIRED. "
    "At least 3 columns must be present. "
    "The output MUST NOT exceed 500 rows."
)

SCHEMA_TEXT = (
    "The JSON schema for the response body must include: patient_id (string, required), "
    "diagnosis_code (enum: ['A01', 'B02']), risk_score (number, nullable). "
    "The API endpoint accepts a POST request with Content-Type application/json. "
    "Response format: JSON with header row. Output as pipe-delimited CSV."
)

GHERKIN_TEXT = (
    "Feature: Patient data processing\n"
    "  Scenario: Valid input file\n"
    "    Given the input file has a header row\n"
    "    When processing each record\n"
    "    Then validate all required fields\n"
    "    And mark invalid rows\n"
    "  Examples:\n"
    "    | patient_id | status |\n"
)


def test_registry_completeness():
    print("\n-- Registry completeness --")
    check("PACK_REGISTRY has 16 packs", len(PACK_REGISTRY) == 16,
          f"got {len(PACK_REGISTRY)}")
    expected = {
        'obligation', 'prohibition', 'recommendation', 'conditional',
        'cardinality', 'state', 'schema_json', 'schema_types',
        'data_fields', 'tabular', 'gherkin', 'rubric', 'acceptance',
        'task_verbs', 'value_domain', 'format_markup',
    }
    check("All expected pack names present", set(PACK_REGISTRY.keys()) == expected,
          f"missing={expected - set(PACK_REGISTRY.keys())}, extra={set(PACK_REGISTRY.keys()) - expected}")


def test_score_pack_obligation():
    print("\n-- score_pack obligation --")
    result = score_pack(CONSTRAINT_TEXT, 'obligation', n_sentences=5)
    check("obligation uppercase_hits > 0", result.uppercase_hits > 0,
          f"got {result.uppercase_hits}")
    check("obligation has hits (raw or uppercase)", result.raw_hits > 0 or result.uppercase_hits > 0,
          f"raw_hits={result.raw_hits}, uppercase_hits={result.uppercase_hits}")
    check("obligation capped_score <= family_cap",
          result.capped_score <= PACK_REGISTRY['obligation'].family_cap + 0.001,
          f"got {result.capped_score}, cap={PACK_REGISTRY['obligation'].family_cap}")


def test_score_pack_empty():
    print("\n-- score_pack on empty text --")
    result = score_pack("", 'obligation', n_sentences=1)
    check("empty text obligation hits == 0", result.raw_hits == 0,
          f"got {result.raw_hits}")
    check("empty text obligation capped_score == 0", result.capped_score == 0.0,
          f"got {result.capped_score}")


def test_layer_assignments():
    print("\n-- Layer assignments --")
    ps_packs = get_packs_for_layer('prompt_signature')
    check("prompt_signature layer has constraint + exec_spec packs",
          'obligation' in ps_packs and 'gherkin' in ps_packs,
          f"got {ps_packs}")
    vd_packs = get_packs_for_layer('voice_dissonance')
    check("voice_dissonance layer has schema + format packs",
          'schema_json' in vd_packs and 'format_markup' in vd_packs,
          f"got {vd_packs}")
    idi_packs = get_packs_for_layer('instruction_density')
    check("instruction_density layer has task_verbs + value_domain",
          'task_verbs' in idi_packs and 'value_domain' in idi_packs,
          f"got {idi_packs}")


def test_mode_filtering():
    print("\n-- Mode filtering --")
    tp_packs = get_packs_for_mode('task_prompt')
    check("task_prompt mode includes obligation", 'obligation' in tp_packs)
    check("task_prompt mode includes all 16 packs (all are task_prompt or both)",
          len(tp_packs) == 16, f"got {len(tp_packs)}")


def test_category_aggregation():
    print("\n-- Category aggregation --")
    scores = score_packs(CONSTRAINT_TEXT, n_sentences=5)
    total_constraint = get_total_constraint_score(scores)
    check("constraint score > 0 for constraint-heavy text", total_constraint > 0,
          f"got {total_constraint}")

    schema_scores = score_packs(SCHEMA_TEXT, n_sentences=3)
    total_schema = get_total_schema_score(schema_scores)
    check("schema score > 0 for schema-heavy text", total_schema > 0,
          f"got {total_schema}")


def test_family_caps():
    print("\n-- Family caps --")
    scores = score_packs(CONSTRAINT_TEXT, n_sentences=5)
    for name, ps in scores.items():
        pack = PACK_REGISTRY[name]
        check(f"{name} capped_score <= family_cap ({pack.family_cap})",
              ps.capped_score <= pack.family_cap + 0.001,
              f"got {ps.capped_score}")


def test_enhanced_prompt_signature():
    print("\n-- Enhanced prompt signature --")
    result = run_prompt_signature_enhanced(CONSTRAINT_TEXT)
    check("pack_boost > 0 for constraint-heavy text",
          result.get('pack_boost', 0) > 0,
          f"got {result.get('pack_boost')}")
    check("pack_constraint_score present",
          'pack_constraint_score' in result)
    check("composite >= legacy_composite",
          result.get('composite', 0) >= result.get('legacy_composite', 0),
          f"composite={result.get('composite')}, legacy={result.get('legacy_composite')}")

    human_result = run_prompt_signature_enhanced(HUMAN_TEXT)
    check("pack_boost == 0 or near 0 for human text",
          human_result.get('pack_boost', 0) <= 0.05,
          f"got {human_result.get('pack_boost')}")


def test_enhanced_voice_dissonance():
    print("\n-- Enhanced voice dissonance --")
    result = run_voice_dissonance_enhanced(SCHEMA_TEXT)
    check("pack_schema_score > 0 for schema-heavy text",
          result.get('pack_schema_score', 0) > 0,
          f"got {result.get('pack_schema_score')}")
    check("enhanced spec_score >= legacy spec",
          result.get('spec_score', 0) >= result.get('legacy_spec_score', 0),
          f"spec={result.get('spec_score')}, legacy={result.get('legacy_spec_score')}")


def test_enhanced_instruction_density():
    print("\n-- Enhanced instruction density --")
    result_paired = run_instruction_density_enhanced(
        CONSTRAINT_TEXT, constraint_active=True, schema_active=False)
    result_unpaired = run_instruction_density_enhanced(
        CONSTRAINT_TEXT, constraint_active=False, schema_active=False)
    check("paired has higher or equal weight than unpaired",
          result_paired.get('pack_tv_weight', 0) >= result_unpaired.get('pack_tv_weight', 0),
          f"paired={result_paired.get('pack_tv_weight')}, unpaired={result_unpaired.get('pack_tv_weight')}")


if __name__ == '__main__':
    print("=" * 70)
    print("  LEXICON PACK TESTS")
    print("=" * 70)

    test_registry_completeness()
    test_score_pack_obligation()
    test_score_pack_empty()
    test_layer_assignments()
    test_mode_filtering()
    test_category_aggregation()
    test_family_caps()
    test_enhanced_prompt_signature()
    test_enhanced_voice_dissonance()
    test_enhanced_instruction_density()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
