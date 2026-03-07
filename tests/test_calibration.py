"""Tests for calibration save/load/apply cycle and p-value monotonicity."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import (
    calibrate_from_baselines, save_calibration, load_calibration,
    apply_calibration,
)

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


def _write_jsonl(path, records):
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


def test_insufficient_data():
    print("\n-- Insufficient data returns None --")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(10):
            f.write(json.dumps({'ground_truth': 'human', 'confidence': 0.3}) + '\n')
        path = f.name
    try:
        result = calibrate_from_baselines(path)
        check("calibrate with < 20 records returns None", result is None,
              f"got {result}")
    finally:
        os.unlink(path)


def test_sufficient_data():
    print("\n-- Sufficient data returns valid table --")
    records = [{'ground_truth': 'human', 'confidence': i / 100.0,
                'domain': 'test', 'length_bin': 'medium'}
               for i in range(30)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
        path = f.name
    try:
        result = calibrate_from_baselines(path)
        check("calibrate with 30 records returns dict", result is not None and isinstance(result, dict))
        check("result has 'global' key", 'global' in result)
        check("result has 'n_calibration' == 30", result.get('n_calibration') == 30,
              f"got {result.get('n_calibration')}")
        check("global has 3 alpha thresholds", len(result.get('global', {})) == 3,
              f"got {len(result.get('global', {}))}")
    finally:
        os.unlink(path)


def test_save_load_roundtrip():
    print("\n-- Save/load round-trip --")
    records = [{'ground_truth': 'human', 'confidence': i / 50.0,
                'domain': 'clinical', 'length_bin': 'short'}
               for i in range(25)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
        jsonl_path = f.name

    cal_path = jsonl_path.replace('.jsonl', '_cal.json')
    try:
        cal = calibrate_from_baselines(jsonl_path)
        check("calibration built successfully", cal is not None)
        if cal is None:
            return

        save_calibration(cal, cal_path)
        check("calibration file created", os.path.exists(cal_path))

        loaded = load_calibration(cal_path)
        check("loaded has same n_calibration",
              loaded['n_calibration'] == cal['n_calibration'],
              f"loaded={loaded['n_calibration']}, orig={cal['n_calibration']}")

        for alpha in [0.01, 0.05, 0.10]:
            orig_val = cal['global'].get(alpha, -1)
            loaded_val = loaded['global'].get(alpha, -2)
            check(f"global alpha={alpha} round-trips",
                  abs(orig_val - loaded_val) < 0.0001,
                  f"orig={orig_val}, loaded={loaded_val}")
    finally:
        os.unlink(jsonl_path)
        if os.path.exists(cal_path):
            os.unlink(cal_path)


def test_apply_without_cal_table():
    print("\n-- Apply calibration without cal_table --")
    result = apply_calibration(0.75, None)
    check("raw unchanged", result['calibrated_confidence'] == 0.75,
          f"got {result['calibrated_confidence']}")
    check("confidence_quantile is None", result['confidence_quantile'] is None)
    check("stratum is uncalibrated", result['stratum_used'] == 'uncalibrated')


def test_apply_with_cal_table():
    print("\n-- Apply calibration with cal_table --")
    cal_table = {
        'global': {0.01: 0.1, 0.05: 0.3, 0.10: 0.5},
        'strata': {},
        'n_calibration': 50,
    }
    result = apply_calibration(0.75, cal_table)
    check("calibrated_confidence is a number",
          isinstance(result['calibrated_confidence'], (int, float)))
    check("confidence_quantile is a number", isinstance(result['confidence_quantile'], (int, float)))


def test_stratum_fallback():
    print("\n-- Stratum fallback to global --")
    cal_table = {
        'global': {0.01: 0.1, 0.05: 0.3, 0.10: 0.5},
        'strata': {('clinical', 'short'): {0.01: 0.05, 0.05: 0.2, 0.10: 0.4}},
        'n_calibration': 50,
    }
    result_stratum = apply_calibration(0.75, cal_table, domain='clinical', length_bin='short')
    check("uses stratum when found", result_stratum['stratum_used'] == 'clinical_short',
          f"got {result_stratum['stratum_used']}")

    result_fallback = apply_calibration(0.75, cal_table, domain='unknown_domain', length_bin='long')
    check("falls back to global when stratum not found",
          result_fallback['stratum_used'] == 'global',
          f"got {result_fallback['stratum_used']}")


def test_pvalue_monotonicity():
    print("\n-- confidence_quantile monotonicity --")
    cal_table = {
        'global': {0.01: 0.10, 0.05: 0.30, 0.10: 0.50},
        'strata': {},
        'n_calibration': 100,
    }
    # As confidence increases, nc_score decreases, quantile should increase
    confidences = [0.40, 0.60, 0.75, 0.85, 0.95]
    quantiles = []
    for conf in confidences:
        result = apply_calibration(conf, cal_table)
        quantiles.append(result['confidence_quantile'])

    check("confidence_quantiles are monotonically non-decreasing as confidence increases",
          all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles) - 1)),
          f"quantiles={quantiles} for confidences={confidences}")


if __name__ == '__main__':
    print("=" * 70)
    print("  CALIBRATION TESTS")
    print("=" * 70)

    test_insufficient_data()
    test_sufficient_data()
    test_save_load_roundtrip()
    test_apply_without_cal_table()
    test_apply_with_cal_table()
    test_stratum_fallback()
    test_pvalue_monotonicity()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
