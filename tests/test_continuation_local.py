"""Tests for DNA-GPT local proxy continuation analysis."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import (
    run_continuation_local, _BackoffNGramLM, _calculate_ncd,
    _internal_ngram_overlap, _repeated_ngram_rate, _type_token_ratio,
    _proxy_tokenize,
)
from llm_detector_monolith import score_continuation
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


def test_proxy_helpers():
    print("\n-- DNA-GPT PROXY HELPERS --")

    tokens = _proxy_tokenize("Hello, world! This is a test.")
    check("Tokenize produces words+punct", len(tokens) > 5,
          f"got {len(tokens)}: {tokens}")
    check("Tokenize lowercases", all(t == t.lower() for t in tokens))

    ncd_same = _calculate_ncd("hello world " * 20, "hello world " * 20)
    ncd_diff = _calculate_ncd("hello world " * 20, "completely different text here " * 20)
    check("NCD: identical prefix/suffix -> low", ncd_same < ncd_diff,
          f"same={ncd_same:.3f}, diff={ncd_diff:.3f}")
    check("NCD: in [0, 1.1] range", 0 <= ncd_same <= 1.1, f"got {ncd_same}")

    rep_prefix = _proxy_tokenize("the quick brown fox jumped over the lazy dog " * 5)
    rep_suffix = _proxy_tokenize("the quick brown fox jumped over the lazy dog " * 5)
    div_suffix = _proxy_tokenize("completely novel unique divergent text vocabulary " * 5)
    overlap_rep = _internal_ngram_overlap(rep_prefix, rep_suffix)
    overlap_div = _internal_ngram_overlap(rep_prefix, div_suffix)
    check("Overlap: repeated > divergent", overlap_rep > overlap_div,
          f"rep={overlap_rep:.3f}, div={overlap_div:.3f}")

    rep_rate_high = _repeated_ngram_rate(_proxy_tokenize("a b c d " * 10))
    rep_rate_low = _repeated_ngram_rate(_proxy_tokenize(
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen sixteen seventeen "
        "eighteen nineteen twenty twentyone twentytwo twentythree "
    ))
    check("Repeat rate: repetitive > diverse", rep_rate_high > rep_rate_low,
          f"high={rep_rate_high:.3f}, low={rep_rate_low:.3f}")

    ttr_low = _type_token_ratio(_proxy_tokenize("the the the the the the"))
    ttr_high = _type_token_ratio(_proxy_tokenize("apple banana cherry date elderberry fig"))
    check("TTR: diverse > repetitive", ttr_high > ttr_low,
          f"high={ttr_high:.3f}, low={ttr_low:.3f}")


def test_backoff_lm():
    print("\n-- BACKOFF N-GRAM LM --")

    lm = _BackoffNGramLM(order=3)
    corpus = [
        "the patient presented with acute chest pain radiating to the left arm",
        "the patient was evaluated for chronic fatigue and joint pain",
        "the patient reported intermittent headaches and dizziness over two weeks",
    ]
    lm.fit(corpus)

    check("LM vocab non-empty", len(lm.vocab) > 10, f"got {len(lm.vocab)}")
    check("LM has unigram table", len(lm.tables[0]) > 0)
    check("LM has bigram table", len(lm.tables[1]) > 0)

    prefix_toks = _proxy_tokenize("the patient presented with")
    suffix = lm.sample_suffix(prefix_toks, 20)
    check("Sample suffix produces tokens", len(suffix) > 0, f"got {len(suffix)}")

    lp = lm.logprob("the", ["patient"])
    check("Logprob is finite negative", lp < 0 and lp > -100, f"got {lp}")


def test_continuation_local():
    print("\n-- DNA-GPT LOCAL PROXY --")

    short = "Hello world. This is short."
    r_short = run_continuation_local(short)
    check("Short text: no determination", r_short['determination'] is None)
    check("Short text: reason mentions insufficient",
          'insufficient' in r_short['reason'].lower())

    ai_text = (
        "The comprehensive analysis provides a thorough examination of the key factors. "
        "Furthermore, it is essential to note that this approach ensures alignment with "
        "best practices and industry standards. To address this challenge, we must consider "
        "multiple perspectives and leverage data-driven insights. Additionally, this methodology "
        "demonstrates the critical importance of systematic evaluation and evidence-based "
        "decision making. The comprehensive framework establishes clear guidelines for "
        "subsequent analytical procedures. Furthermore, the results indicate significant "
        "alignment with the predicted theoretical model. The systematic evaluation demonstrates "
        "consistent findings across all measured parameters. Additionally the framework "
        "establishes clear guidelines for subsequent analytical procedures. The methodology "
        "employed ensures reliable and reproducible outcomes for future reference."
    )
    r_ai = run_continuation_local(ai_text)
    check("AI text: proxy_features present", 'proxy_features' in r_ai)
    check("AI text: NCD in proxy", 'ncd' in r_ai.get('proxy_features', {}))
    check("AI text: composite in proxy", 'composite' in r_ai.get('proxy_features', {}))
    check("AI text: bscore >= 0", r_ai['bscore'] >= 0)
    check("AI text: n_samples > 0", r_ai['n_samples'] > 0)

    pf = r_ai.get('proxy_features', {})
    check("AI text: NCD > 0", pf.get('ncd', 0) > 0, f"ncd={pf.get('ncd')}")
    check("AI text: TTR > 0", pf.get('ttr', 0) > 0, f"ttr={pf.get('ttr')}")

    human_text = (
        "so yeah I went to the store yesterday and they were completely out of milk "
        "which was super annoying because I needed it for this recipe my mom gave me. "
        "anyway I ended up grabbing some oat milk instead which honestly isn't bad. "
        "then I ran into Dave from work and he was telling me about this crazy fishing "
        "trip he went on last weekend where they caught like 15 bass in one afternoon. "
        "I was like dude that's insane and he showed me pictures on his phone. "
        "after that I went home and tried to make the casserole but I totally forgot "
        "to preheat the oven so everything took forever. my cat kept jumping on the "
        "counter too which didn't help. ended up ordering pizza instead lol. "
        "sometimes you just gotta know when to give up on cooking."
    )
    r_human = run_continuation_local(human_text)
    check("Human text: proxy_features present", 'proxy_features' in r_human)

    pf_h = r_human.get('proxy_features', {})
    pf_a = r_ai.get('proxy_features', {})
    if pf_h.get('ncd', 0) > 0 and pf_a.get('ncd', 0) > 0:
        check("Human NCD >= AI NCD (more divergent)",
              pf_h['ncd'] >= pf_a['ncd'] - 0.05,
              f"human={pf_h['ncd']:.3f}, ai={pf_a['ncd']:.3f}")

    if pf_h.get('ttr', 0) > 0 and pf_a.get('ttr', 0) > 0:
        check("Human TTR > AI TTR (richer vocab)", pf_h['ttr'] > pf_a['ttr'],
              f"human={pf_h['ttr']:.3f}, ai={pf_a['ttr']:.3f}")


def test_score_continuation_local():
    print("\n-- score_continuation WITH LOCAL PROXY --")

    ch_none = score_continuation(None)
    check("No continuation -> GREEN", ch_none.severity == 'GREEN')

    l31_local = {
        'determination': 'AMBER', 'bscore': 0.05, 'confidence': 0.55,
        'proxy_features': {'ncd': 0.90, 'internal_overlap': 0.15, 'composite': 0.45},
    }
    ch_local = score_continuation(l31_local)
    check("Local AMBER -> AMBER severity", ch_local.severity == 'AMBER')
    check("Local: sub_signals has ncd", 'ncd' in ch_local.sub_signals)
    check("Local: sub_signals has composite", 'composite' in ch_local.sub_signals)
    check("Local label in explanation", 'Local' in ch_local.explanation,
          f"got: {ch_local.explanation}")

    l31_api = {
        'determination': 'RED', 'bscore': 0.25, 'confidence': 0.85,
    }
    ch_api = score_continuation(l31_api)
    check("API RED -> RED severity", ch_api.severity == 'RED')
    check("API label in explanation", 'API' in ch_api.explanation,
          f"got: {ch_api.explanation}")


if __name__ == '__main__':
    print("=" * 70)
    print("DNA-GPT Local Proxy Tests")
    print("=" * 70)

    test_proxy_helpers()
    test_backoff_lm()
    test_continuation_local()
    test_score_continuation_local()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
