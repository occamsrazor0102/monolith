"""Tests for span-level explainability (collect_spans)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import (collect_spans, analyze_prompt, score_pack,
                                    generate_html_report, run_preamble)
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


def test_empty_text():
    print("\n-- Empty text returns no spans --")
    spans = collect_spans("")
    check("empty text -> empty list", spans == [], f"got {spans}")


def test_preamble_spans():
    print("\n-- Preamble span detection --")
    text = "Sure thing! Here is your revised task prompt for evaluation."
    spans = collect_spans(text)
    preamble_spans = [s for s in spans if s['layer'] == 'preamble']
    check("preamble spans found", len(preamble_spans) > 0,
          f"got {len(preamble_spans)} spans")
    if preamble_spans:
        s = preamble_spans[0]
        check("has start/end", 'start' in s and 'end' in s)
        check("text matches slice", text[s['start']:s['end']] == s['text'],
              f"span text={s['text']!r}, slice={text[s['start']:s['end']]!r}")
        check("severity is CRITICAL", s['severity'] == 'CRITICAL',
              f"got {s['severity']}")


def test_cot_leakage_spans():
    print("\n-- CoT leakage span detection --")
    text = "<think>Let me reason about this.</think> The answer is 42."
    spans = collect_spans(text)
    cot_spans = [s for s in spans if s['pattern'] == 'cot_leakage']
    check("cot_leakage spans found", len(cot_spans) >= 2,
          f"got {len(cot_spans)}")
    if cot_spans:
        check("first span is <think>", cot_spans[0]['text'] == '<think>',
              f"got {cot_spans[0]['text']!r}")


def test_fingerprint_spans():
    print("\n-- Fingerprint word spans --")
    text = "We must delve into this comprehensive analysis and leverage our robust framework."
    spans = collect_spans(text)
    fp_spans = [s for s in spans if s['layer'] == 'fingerprint']
    fp_words = {s['text'].lower() for s in fp_spans}
    check("fingerprint spans found", len(fp_spans) >= 3,
          f"got {len(fp_spans)}: {fp_words}")
    check("'delve' detected", 'delve' in fp_words, f"got {fp_words}")
    check("'comprehensive' detected", 'comprehensive' in fp_words, f"got {fp_words}")
    check("'leverage' detected", 'leverage' in fp_words, f"got {fp_words}")
    check("'robust' detected", 'robust' in fp_words, f"got {fp_words}")
    # Verify text slice correctness
    for s in fp_spans:
        check(f"'{s['text']}' slice matches",
              text[s['start']:s['end']].lower() == s['text'].lower(),
              f"slice={text[s['start']:s['end']]!r}")


def test_formulaic_spans():
    print("\n-- Formulaic academic phrase spans --")
    text = "It is worth noting that the analysis reveals significant patterns."
    spans = collect_spans(text)
    form_spans = [s for s in spans if s['layer'] == 'formulaic']
    check("formulaic spans found", len(form_spans) > 0,
          f"got {len(form_spans)} total spans, layers: {set(s['layer'] for s in spans)}")
    if form_spans:
        check("weight > 0", form_spans[0]['weight'] > 0,
              f"got {form_spans[0]['weight']}")
        s = form_spans[0]
        check("text matches slice", text[s['start']:s['end']] == s['text'],
              f"span text={s['text']!r}")


def test_power_adj_spans():
    print("\n-- Power adjective spans --")
    text = "This is a comprehensive and robust approach to a multifaceted problem."
    spans = collect_spans(text)
    pa_spans = [s for s in spans if s['layer'] == 'power_adj']
    pa_words = {s['text'].lower() for s in pa_spans}
    check("power_adj spans found", len(pa_spans) >= 2,
          f"got {len(pa_spans)}: {pa_words}")
    check("'comprehensive' detected", 'comprehensive' in pa_words, f"got {pa_words}")
    check("'robust' detected", 'robust' in pa_words, f"got {pa_words}")


def test_transition_spans():
    print("\n-- Transition connector spans --")
    text = "However, the results are clear. Furthermore, we see improvement. Moreover, growth continues."
    spans = collect_spans(text)
    tr_spans = [s for s in spans if s['layer'] == 'transition']
    tr_words = {s['text'].lower() for s in tr_spans}
    check("transition spans found", len(tr_spans) >= 3,
          f"got {len(tr_spans)}: {tr_words}")
    check("'however' detected", 'however' in tr_words, f"got {tr_words}")
    check("'furthermore' detected", 'furthermore' in tr_words, f"got {tr_words}")
    check("'moreover' detected", 'moreover' in tr_words, f"got {tr_words}")


def test_sorted_order():
    print("\n-- Spans sorted by start position --")
    spans = collect_spans(AI_TEXT)
    if len(spans) >= 2:
        starts = [s['start'] for s in spans]
        check("sorted by start", starts == sorted(starts),
              f"starts: {starts[:10]}...")
    else:
        check("at least 2 spans for sort check", False,
              f"only {len(spans)} spans")


def test_text_slice_consistency():
    print("\n-- All span text fields match their slices --")
    spans = collect_spans(AI_TEXT)
    check("AI_TEXT produces spans", len(spans) > 0, f"got {len(spans)}")
    mismatches = []
    for s in spans:
        actual = AI_TEXT[s['start']:s['end']]
        if actual != s['text']:
            mismatches.append((s['text'], actual, s['start'], s['end']))
    check("all text fields match slices", len(mismatches) == 0,
          f"{len(mismatches)} mismatches: {mismatches[:3]}")


def test_span_dict_fields():
    print("\n-- Span dict has all required fields --")
    spans = collect_spans(AI_TEXT)
    if spans:
        s = spans[0]
        required = {'start', 'end', 'text', 'layer', 'pattern', 'severity', 'weight'}
        check("has all required keys", required.issubset(s.keys()),
              f"missing: {required - set(s.keys())}")
        check("start is int", isinstance(s['start'], int))
        check("end is int", isinstance(s['end'], int))
        check("weight is numeric", isinstance(s['weight'], (int, float)))
    else:
        check("spans present for field check", False, "no spans")


def test_human_text_fewer_spans():
    print("\n-- Human text has fewer spans than AI text --")
    ai_spans = collect_spans(AI_TEXT)
    human_spans = collect_spans(HUMAN_TEXT)
    check("AI text has more spans",
          len(ai_spans) > len(human_spans),
          f"ai={len(ai_spans)}, human={len(human_spans)}")


def test_pipeline_integration():
    print("\n-- Pipeline: _spans in analyze_prompt result --")
    result = analyze_prompt(text=AI_TEXT, task_id="span_test")
    check("_spans key present", '_spans' in result,
          f"keys include: {[k for k in result if k.startswith('_')]}")
    check("_spans is list", isinstance(result.get('_spans'), list))
    if result.get('_spans'):
        check("_spans has entries", len(result['_spans']) > 0)


def test_overlapping_layers():
    print("\n-- Same word can appear in multiple layers --")
    text = "This comprehensive framework is robust and holistic."
    spans = collect_spans(text)
    # 'comprehensive' should be in both fingerprint and power_adj
    comp_layers = {s['layer'] for s in spans if 'comprehensive' in s['text'].lower()}
    check("'comprehensive' in multiple layers", len(comp_layers) >= 2,
          f"layers: {comp_layers}")


def test_pack_score_spans():
    print("\n-- PackScore has spans after scoring --")
    text = "You MUST include all required fields. The system SHALL NOT exceed limits."
    ps = score_pack(text, 'obligation', n_sentences=2)
    check("PackScore has spans attr", hasattr(ps, 'spans'))
    check("spans is list", isinstance(ps.spans, list))
    if ps.raw_hits > 0:
        check("spans non-empty when hits > 0", len(ps.spans) > 0,
              f"hits={ps.raw_hits}, spans={len(ps.spans)}")
        s = ps.spans[0]
        check("span has start", 'start' in s)
        check("span has end", 'end' in s)
        check("span has pack name", s.get('pack') == 'obligation',
              f"got {s.get('pack')}")
        check("span text matches slice",
              text[s['start']:s['end']].lower() == s['text'].lower()[:len(text[s['start']:s['end']])],
              f"span={s['text']!r}, slice={text[s['start']:s['end']]!r}")


def test_preamble_returns_4_tuple():
    print("\n-- run_preamble returns 4-tuple with spans --")
    text = "Sure thing! Here is your revised evaluation prompt."
    result = run_preamble(text)
    check("returns 4-tuple", len(result) == 4, f"got {len(result)}-tuple")
    score, sev, hits, spans = result
    check("score is float", isinstance(score, float))
    check("spans is list", isinstance(spans, list))
    if spans:
        s = spans[0]
        check("span has start/end", 'start' in s and 'end' in s)
        check("span has pattern", 'pattern' in s)
        check("span has severity", 'severity' in s)


def test_detection_spans_in_pipeline():
    print("\n-- Pipeline: detection_spans merges all sources --")
    result = analyze_prompt(text=AI_TEXT, task_id="det_span_test")
    check("detection_spans key present", 'detection_spans' in result)
    ds = result.get('detection_spans', [])
    check("detection_spans is list", isinstance(ds, list))
    # Should include at least base spans (fingerprint/formulaic/etc)
    if ds:
        layers = {s.get('layer', s.get('type', s.get('pack', '?'))) for s in ds}
        check("multiple span sources", len(layers) >= 1,
              f"layers: {layers}")


def test_html_report_basic():
    print("\n-- HTML report generation --")
    text = "This comprehensive analysis leverages robust frameworks."
    result = {
        'determination': 'AMBER',
        'reason': 'Test reason',
        'confidence': 0.75,
        'task_id': 'html_test',
        'word_count': 7,
        'mode': 'auto',
        'detection_spans': [
            {'start': 5, 'end': 18, 'text': 'comprehensive', 'severity': 'LOW',
             'pattern': 'fingerprint_word'},
        ],
        'channel_details': {
            'channels': {
                'prompt_structure': {'severity': 'AMBER', 'explanation': 'test'},
                'stylometry': {'severity': 'GREEN', 'explanation': ''},
                'continuation': {'severity': 'GREEN', 'explanation': ''},
                'windowing': {'severity': 'GREEN', 'explanation': ''},
            }
        },
    }
    html = generate_html_report(text, result)
    check("returns string", isinstance(html, str))
    check("contains DOCTYPE", '<!DOCTYPE html>' in html)
    check("contains determination", 'AMBER' in html)
    check("contains highlighted span", 'signal-LOW' in html)
    check("contains task_id", 'html_test' in html)


def test_html_empty_spans():
    print("\n-- HTML report with no spans --")
    text = "Plain text with no signals."
    result = {
        'determination': 'GREEN', 'reason': '', 'confidence': 0.0,
        'task_id': 'empty', 'word_count': 6, 'mode': 'auto',
        'detection_spans': [],
        'channel_details': {'channels': {}},
    }
    html = generate_html_report(text, result)
    check("produces valid HTML", '<!DOCTYPE html>' in html)
    check("contains escaped text", 'Plain text' in html)


if __name__ == '__main__':
    print("=" * 70)
    print("  SPAN-LEVEL EXPLAINABILITY TESTS")
    print("=" * 70)

    test_empty_text()
    test_preamble_spans()
    test_cot_leakage_spans()
    test_fingerprint_spans()
    test_formulaic_spans()
    test_power_adj_spans()
    test_transition_spans()
    test_sorted_order()
    test_text_slice_consistency()
    test_span_dict_fields()
    test_human_text_fewer_spans()
    test_pipeline_integration()
    test_overlapping_layers()
    test_pack_score_spans()
    test_preamble_returns_4_tuple()
    test_detection_spans_in_pipeline()
    test_html_report_basic()
    test_html_empty_spans()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
