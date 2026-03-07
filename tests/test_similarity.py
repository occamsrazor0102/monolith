"""Tests for enhanced cross-submission similarity (FEAT 11-15)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_detector_monolith import (
    _word_shingles, _jaccard, _structural_similarity, _adaptive_thresholds,
    _minhash_signature, _minhash_jaccard, _load_minhash_store, _store_minhash,
    _factor_instructions, analyze_similarity, apply_similarity_feedback,
)


def _make_result(task_id, attempter='A', occupation='occ1', determination='GREEN',
                 reason='', **extra):
    r = {
        'task_id': task_id,
        'attempter': attempter,
        'occupation': occupation,
        'determination': determination,
        'confidence': 0.5,
        'reason': reason,
    }
    r.update(extra)
    return r


# === FEAT 11: Adaptive Thresholds ===

class TestAdaptiveThresholds(unittest.TestCase):

    def test_small_group_falls_back(self):
        """Groups with < 4 results use flat thresholds."""
        group = [_make_result(f't{i}', attempter=f'a{i}') for i in range(3)]
        cache = {f't{i}': _word_shingles(f'word{i} text here') for i in range(3)}
        jt, st = _adaptive_thresholds(group, cache, 0.40, 0.90)
        self.assertEqual(jt, 0.40)
        self.assertEqual(st, 0.90)

    def test_adaptive_computation(self):
        """Verify adaptive thresholds are computed from group distribution."""
        # Create 5 results with different texts
        texts = {
            't0': 'the quick brown fox jumps over the lazy dog',
            't1': 'a fast red fox leaps across the sleepy hound',
            't2': 'the slow green cat sits under the active bird',
            't3': 'the bright blue fish swims through the deep ocean',
            't4': 'the large gray wolf runs along the mountain trail',
        }
        group = [_make_result(f't{i}', attempter=f'a{i}') for i in range(5)]
        cache = {tid: _word_shingles(txt) for tid, txt in texts.items()}
        jt, st = _adaptive_thresholds(group, cache, 0.40, 0.90)
        # Adaptive thresholds should be <= flat thresholds (clamped)
        self.assertLessEqual(jt, 0.40)
        self.assertLessEqual(st, 0.90)
        # And >= minimum floors
        self.assertGreaterEqual(jt, 0.15)
        self.assertGreaterEqual(st, 0.50)

    def test_adaptive_never_exceeds_user_threshold(self):
        """Adaptive thresholds clamped to user-specified values."""
        # Identical texts → high Jaccard → high adaptive threshold
        group = [_make_result(f't{i}', attempter=f'a{i}') for i in range(5)]
        cache = {f't{i}': _word_shingles('identical text for all submissions here') for i in range(5)}
        jt, st = _adaptive_thresholds(group, cache, 0.30, 0.80)
        self.assertLessEqual(jt, 0.30)
        self.assertLessEqual(st, 0.80)


# === FEAT 12: Semantic Similarity ===

class TestSemanticSimilarity(unittest.TestCase):

    def test_semantic_disabled_returns_zero(self):
        """When semantic=False, semantic field is 0.0 in flagged pairs."""
        texts = {
            't0': 'the quick brown fox jumps over the lazy dog ' * 5,
            't1': 'the quick brown fox jumps over the lazy dog ' * 5,
        }
        results = [
            _make_result('t0', attempter='alice', occupation='eng'),
            _make_result('t1', attempter='bob', occupation='eng'),
        ]
        pairs = analyze_similarity(results, texts, jaccard_threshold=0.10, semantic=False)
        for p in pairs:
            self.assertEqual(p['semantic'], 0.0)

    def test_semantic_field_present_in_pairs(self):
        """Flagged pairs always include a 'semantic' key."""
        texts = {
            't0': 'the quick brown fox jumps over the lazy dog ' * 5,
            't1': 'the quick brown fox jumps over the lazy dog ' * 5,
        }
        results = [
            _make_result('t0', attempter='alice', occupation='eng'),
            _make_result('t1', attempter='bob', occupation='eng'),
        ]
        pairs = analyze_similarity(results, texts, jaccard_threshold=0.10)
        self.assertTrue(len(pairs) > 0)
        for p in pairs:
            self.assertIn('semantic', p)


# === FEAT 13: Similarity Determination Feedback ===

class TestSimilarityFeedback(unittest.TestCase):

    def test_upgrade_yellow_to_amber(self):
        """YELLOW paired with RED → YELLOW upgraded to AMBER."""
        results = [
            _make_result('t0', determination='YELLOW', reason='low signal'),
            _make_result('t1', determination='RED', reason='strong signal'),
        ]
        pairs = [{
            'id_a': 't0', 'id_b': 't1',
            'flag_type': 'text',
            'jaccard': 0.50, 'semantic': 0.0,
        }]
        n = apply_similarity_feedback(results, pairs)
        self.assertEqual(n, 1)
        self.assertEqual(results[0]['determination'], 'AMBER')
        self.assertTrue(results[0].get('similarity_upgraded'))

    def test_no_downgrade(self):
        """GREEN result never downgraded."""
        results = [
            _make_result('t0', determination='GREEN', reason='clean'),
            _make_result('t1', determination='RED', reason='flagged'),
        ]
        pairs = [{
            'id_a': 't0', 'id_b': 't1',
            'flag_type': 'text',
            'jaccard': 0.50, 'semantic': 0.0,
        }]
        n = apply_similarity_feedback(results, pairs)
        self.assertEqual(n, 0)
        self.assertEqual(results[0]['determination'], 'GREEN')

    def test_both_yellow_high_semantic(self):
        """Both YELLOW + semantic >= 0.90 → both upgraded to AMBER."""
        results = [
            _make_result('t0', determination='YELLOW', reason='mild'),
            _make_result('t1', determination='YELLOW', reason='mild'),
        ]
        pairs = [{
            'id_a': 't0', 'id_b': 't1',
            'flag_type': 'semantic',
            'jaccard': 0.15, 'semantic': 0.92,
        }]
        n = apply_similarity_feedback(results, pairs)
        self.assertEqual(n, 2)
        self.assertEqual(results[0]['determination'], 'AMBER')
        self.assertEqual(results[1]['determination'], 'AMBER')

    def test_no_action_on_structural_only(self):
        """Structural-only flag_type doesn't trigger feedback."""
        results = [
            _make_result('t0', determination='YELLOW'),
            _make_result('t1', determination='RED'),
        ]
        pairs = [{
            'id_a': 't0', 'id_b': 't1',
            'flag_type': 'structural',
            'jaccard': 0.10, 'semantic': 0.0,
        }]
        n = apply_similarity_feedback(results, pairs)
        self.assertEqual(n, 0)
        self.assertEqual(results[0]['determination'], 'YELLOW')


# === FEAT 14: Cross-Batch MinHash Store ===

class TestMinHash(unittest.TestCase):

    def test_minhash_deterministic(self):
        """Same shingles → same signature."""
        shingles = _word_shingles('the quick brown fox jumps over')
        sig1 = _minhash_signature(shingles)
        sig2 = _minhash_signature(shingles)
        self.assertEqual(sig1, sig2)

    def test_minhash_jaccard_identical(self):
        """Identical signatures → estimated Jaccard ~1.0."""
        shingles = _word_shingles('the quick brown fox jumps over the lazy dog')
        sig = _minhash_signature(shingles)
        self.assertAlmostEqual(_minhash_jaccard(sig, sig), 1.0)

    def test_minhash_jaccard_different(self):
        """Very different texts → low estimated Jaccard."""
        s1 = _minhash_signature(_word_shingles('alpha beta gamma delta epsilon'))
        s2 = _minhash_signature(_word_shingles('one two three four five six seven'))
        self.assertLess(_minhash_jaccard(s1, s2), 0.3)

    def test_minhash_store_roundtrip(self):
        """Write and read JSONL store, verify contents."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name

        try:
            entries = [
                {'task_id': 't1', 'attempter': 'alice', 'minhash': [1, 2, 3]},
                {'task_id': 't2', 'attempter': 'bob', 'minhash': [4, 5, 6]},
            ]
            _store_minhash(entries, path)
            loaded = _load_minhash_store(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]['task_id'], 't1')
            self.assertEqual(loaded[1]['minhash'], [4, 5, 6])
        finally:
            os.unlink(path)

    def test_load_nonexistent_store(self):
        """Loading from nonexistent path returns empty list."""
        loaded = _load_minhash_store('/tmp/does_not_exist_test_minhash.jsonl')
        self.assertEqual(loaded, [])

    def test_cross_batch_detection(self):
        """Store batch 1, run batch 2 — detect similar pair across batches."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            store_path = f.name

        try:
            shared_text = 'the quick brown fox jumps over the lazy dog in the park ' * 3
            text_batch1 = {'t1': shared_text}
            text_batch2 = {'t2': shared_text + ' today'}

            # Batch 1
            r1 = [_make_result('t1', attempter='alice', occupation='eng')]
            analyze_similarity(r1, text_batch1, similarity_store_path=store_path)

            # Batch 2 should detect cross-batch match
            r2 = [_make_result('t2', attempter='bob', occupation='eng')]
            pairs = analyze_similarity(r2, text_batch2, jaccard_threshold=0.10,
                                       similarity_store_path=store_path)
            cross = [p for p in pairs if p['flag_type'] == 'cross-batch']
            self.assertTrue(len(cross) > 0)
        finally:
            os.unlink(store_path)


# === FEAT 15: Instruction Template Factoring ===

class TestInstructionFactoring(unittest.TestCase):

    def test_factor_removes_shared_shingles(self):
        """Instruction shingles are subtracted from text shingles."""
        text_shingles = _word_shingles('please write a detailed report about the topic')
        instr_shingles = _word_shingles('please write a detailed report')
        factored = _factor_instructions(text_shingles, instr_shingles)
        # Factored should have fewer shingles
        self.assertLess(len(factored), len(text_shingles))

    def test_factor_none_instructions(self):
        """None instructions returns original shingles unchanged."""
        shingles = _word_shingles('some text here for testing')
        factored = _factor_instructions(shingles, None)
        self.assertEqual(factored, shingles)

    def test_factored_jaccard_lower(self):
        """Jaccard drops when shared instructions are factored out."""
        instructions = 'write a comprehensive analysis of the following submission'
        text_a = instructions + ' the candidate demonstrated strong leadership skills'
        text_b = instructions + ' the applicant showed excellent technical abilities'

        instr_shingles = _word_shingles(instructions)

        # Without factoring
        jac_raw = _jaccard(_word_shingles(text_a), _word_shingles(text_b))

        # With factoring
        sa = _factor_instructions(_word_shingles(text_a), instr_shingles)
        sb = _factor_instructions(_word_shingles(text_b), instr_shingles)
        jac_factored = _jaccard(sa, sb)

        self.assertLess(jac_factored, jac_raw)

    def test_cli_instructions_integration(self):
        """analyze_similarity with instruction_shingles modifies shingle cache."""
        instructions = 'please evaluate the following submission carefully'
        instr_shingles = _word_shingles(instructions)

        text_a = instructions + ' great work on the project deliverable results'
        text_b = instructions + ' excellent effort on the assignment output findings'

        texts = {'t0': text_a, 't1': text_b}
        results = [
            _make_result('t0', attempter='alice', occupation='eng'),
            _make_result('t1', attempter='bob', occupation='eng'),
        ]

        # Without instruction factoring
        pairs_raw = analyze_similarity(results, texts, jaccard_threshold=0.05)
        # With instruction factoring
        pairs_factored = analyze_similarity(results, texts, jaccard_threshold=0.05,
                                            instruction_shingles=instr_shingles)

        # If both produce pairs, factored should have lower Jaccard
        if pairs_raw and pairs_factored:
            self.assertLess(pairs_factored[0]['jaccard'], pairs_raw[0]['jaccard'])


# === Integration ===

class TestAnalyzeSimilarityIntegration(unittest.TestCase):

    def test_adaptive_thresholds_in_pair(self):
        """Flagged pairs include adaptive threshold fields."""
        texts = {
            't0': 'the quick brown fox jumps over the lazy dog ' * 5,
            't1': 'the quick brown fox jumps over the lazy dog ' * 5,
        }
        results = [
            _make_result('t0', attempter='alice', occupation='eng'),
            _make_result('t1', attempter='bob', occupation='eng'),
        ]
        pairs = analyze_similarity(results, texts, jaccard_threshold=0.10)
        self.assertTrue(len(pairs) > 0)
        self.assertIn('adaptive_jac_threshold', pairs[0])
        self.assertIn('adaptive_struct_threshold', pairs[0])

    def test_empty_results(self):
        """No results produces no pairs."""
        pairs = analyze_similarity([], {})
        self.assertEqual(pairs, [])

    def test_same_attempter_skipped(self):
        """Pairs with same attempter are skipped."""
        texts = {
            't0': 'identical text for this test case ' * 5,
            't1': 'identical text for this test case ' * 5,
        }
        results = [
            _make_result('t0', attempter='alice', occupation='eng'),
            _make_result('t1', attempter='alice', occupation='eng'),
        ]
        pairs = analyze_similarity(results, texts, jaccard_threshold=0.10)
        self.assertEqual(len(pairs), 0)


if __name__ == '__main__':
    unittest.main()
