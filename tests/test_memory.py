"""Tests for BEET Historical Memory Store."""
import json
import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_detector_monolith import MemoryStore, _print_attempter_history


def _make_result(task_id, attempter='worker_1', occupation='pharmacist',
                 determination='GREEN', confidence=0.5, **extra):
    r = {
        'task_id': task_id,
        'attempter': attempter,
        'occupation': occupation,
        'determination': determination,
        'confidence': confidence,
        'reason': '',
        'word_count': 200,
        'preamble_score': 0.0,
        'prompt_signature_composite': 0.3,
        'prompt_signature_cfd': 0.2,
        'instruction_density_idi': 5.0,
        'voice_dissonance_vsd': 10.0,
        'voice_dissonance_spec_score': 3.0,
        'self_similarity_nssi_score': 0.0,
        'audit_trail': {'pipeline_version': 'v0.65'},
    }
    r.update(extra)
    return r


class TestMemoryStoreInit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_dir = os.path.join(self.tmpdir, 'test_beet')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_creates_directory(self):
        store = MemoryStore(self.store_dir)
        self.assertTrue(os.path.isdir(self.store_dir))
        self.assertTrue(os.path.isdir(os.path.join(self.store_dir, 'calibration_history')))

    def test_init_creates_config(self):
        store = MemoryStore(self.store_dir)
        # Config is in memory; saved to disk on _save_config()
        self.assertEqual(store._config['total_submissions'], 0)
        self.assertEqual(store._config['total_batches'], 0)
        self.assertEqual(store._config['version'], '0.65')
        # After save, file exists
        store._save_config()
        self.assertTrue(store.config_path.exists())

    def test_init_idempotent(self):
        store1 = MemoryStore(self.store_dir)
        store1._config['total_submissions'] = 42
        store1._save_config()

        store2 = MemoryStore(self.store_dir)
        self.assertEqual(store2._config['total_submissions'], 42)


class TestRecordBatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_dir = os.path.join(self.tmpdir, 'test_beet')
        self.store = MemoryStore(self.store_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_record_writes_submissions(self):
        results = [_make_result('t1', attempter='alice', occupation='eng')]
        text_map = {'t1': 'some test text for this task'}
        self.store.record_batch(results, text_map, batch_id='b1')

        lines = self.store.submissions_path.read_text().strip().split('\n')
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec['task_id'], 't1')
        self.assertEqual(rec['batch_id'], 'b1')
        self.assertIn('length_bin', rec)

    def test_record_writes_fingerprints(self):
        results = [_make_result('t1', attempter='alice')]
        text_map = {'t1': 'the quick brown fox jumps over the lazy dog'}
        self.store.record_batch(results, text_map, batch_id='b1')

        lines = self.store.fingerprints_path.read_text().strip().split('\n')
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec['task_id'], 't1')
        self.assertIn('minhash_128', rec)
        self.assertEqual(len(rec['minhash_128']), 128)
        self.assertIn('structural_vec', rec)

    def test_record_updates_config(self):
        results = [_make_result('t1'), _make_result('t2')]
        text_map = {'t1': 'text one', 't2': 'text two'}
        self.store.record_batch(results, text_map, batch_id='b1')

        config = json.loads(self.store.config_path.read_text())
        self.assertEqual(config['total_submissions'], 2)
        self.assertEqual(config['total_batches'], 1)

    def test_record_multiple_batches(self):
        r1 = [_make_result('t1', attempter='alice')]
        r2 = [_make_result('t2', attempter='bob')]
        self.store.record_batch(r1, {'t1': 'first batch text'}, batch_id='b1')
        self.store.record_batch(r2, {'t2': 'second batch text'}, batch_id='b2')

        lines = self.store.submissions_path.read_text().strip().split('\n')
        self.assertEqual(len(lines), 2)

        config = json.loads(self.store.config_path.read_text())
        self.assertEqual(config['total_submissions'], 2)
        self.assertEqual(config['total_batches'], 2)

    def test_length_bin_assignment(self):
        short = _make_result('t1', word_count=50)
        medium = _make_result('t2', word_count=200)
        long_ = _make_result('t3', word_count=500)
        very_long = _make_result('t4', word_count=1000)
        results = [short, medium, long_, very_long]
        text_map = {f't{i}': f'text {i}' for i in range(1, 5)}
        self.store.record_batch(results, text_map, batch_id='b1')

        lines = self.store.submissions_path.read_text().strip().split('\n')
        bins = [json.loads(l)['length_bin'] for l in lines]
        self.assertEqual(bins, ['short', 'medium', 'long', 'very_long'])


class TestAttempterProfiles(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmpdir, 'beet'))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_attempter_profile_created(self):
        results = [_make_result('t1', attempter='alice', determination='RED')]
        self.store.record_batch(results, {'t1': 'text'}, batch_id='b1')

        profiles = self.store._load_attempter_profiles()
        self.assertIn('alice', profiles)
        self.assertEqual(profiles['alice']['total_submissions'], 1)

    def test_attempter_profile_updated(self):
        r1 = [_make_result('t1', attempter='alice', determination='RED')]
        r2 = [_make_result('t2', attempter='alice', determination='GREEN')]
        self.store.record_batch(r1, {'t1': 'text 1'}, batch_id='b1')
        self.store.record_batch(r2, {'t2': 'text 2'}, batch_id='b2')

        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['total_submissions'], 2)
        self.assertEqual(profiles['alice']['determinations']['RED'], 1)
        self.assertEqual(profiles['alice']['determinations']['GREEN'], 1)
        self.assertEqual(len(profiles['alice']['batches']), 2)

    def test_risk_tier_normal(self):
        # All GREEN → flag_rate = 0 → NORMAL
        results = [_make_result(f't{i}', attempter='alice', determination='GREEN')
                    for i in range(10)]
        self.store.record_batch(results, {f't{i}': 'text' for i in range(10)})
        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['risk_tier'], 'NORMAL')

    def test_risk_tier_elevated(self):
        # 2 RED / 10 total = 0.20 → ELEVATED
        results = [_make_result(f't{i}', attempter='alice',
                                determination='RED' if i < 2 else 'GREEN')
                    for i in range(10)]
        self.store.record_batch(results, {f't{i}': 'text' for i in range(10)})
        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['risk_tier'], 'ELEVATED')

    def test_risk_tier_high(self):
        # 4 RED / 10 total = 0.40 → HIGH
        results = [_make_result(f't{i}', attempter='alice',
                                determination='RED' if i < 4 else 'GREEN')
                    for i in range(10)]
        self.store.record_batch(results, {f't{i}': 'text' for i in range(10)})
        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['risk_tier'], 'HIGH')

    def test_risk_tier_critical(self):
        # flag_rate > 0.50 AND confirmed_ai > 0 → CRITICAL
        results = [_make_result(f't{i}', attempter='alice',
                                determination='RED' if i < 6 else 'GREEN')
                    for i in range(10)]
        self.store.record_batch(results, {f't{i}': f'text {i}' for i in range(10)})
        self.store.record_confirmation('t0', 'ai', verified_by='reviewer')
        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['risk_tier'], 'CRITICAL')


class TestQueries(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmpdir, 'beet'))
        results = [
            _make_result('t1', attempter='alice', occupation='eng', determination='RED'),
            _make_result('t2', attempter='alice', occupation='eng', determination='GREEN'),
            _make_result('t3', attempter='bob', occupation='eng', determination='GREEN'),
            _make_result('t4', attempter='charlie', occupation='pharm', determination='AMBER'),
        ]
        text_map = {f't{i}': f'text for task {i} goes here' for i in range(1, 5)}
        self.store.record_batch(results, text_map, batch_id='b1')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_get_attempter_history(self):
        history = self.store.get_attempter_history('alice')
        self.assertIsNotNone(history['profile'])
        self.assertEqual(len(history['submissions']), 2)
        self.assertEqual(history['confirmations'], [])

    def test_get_attempter_risk_report(self):
        report = self.store.get_attempter_risk_report(min_submissions=1)
        # Should be sorted by risk tier descending
        self.assertTrue(len(report) > 0)
        tiers = [p['risk_tier'] for p in report]
        tier_rank = {'CRITICAL': 4, 'HIGH': 3, 'ELEVATED': 2, 'NORMAL': 1}
        ranks = [tier_rank.get(t, 0) for t in tiers]
        self.assertEqual(ranks, sorted(ranks, reverse=True))

    def test_get_occupation_baselines(self):
        subs = self.store.get_occupation_baselines('eng')
        self.assertEqual(len(subs), 3)

    def test_pre_batch_context_unknown(self):
        ctx = self.store.pre_batch_context(attempter='unknown_person')
        self.assertEqual(ctx, {})

    def test_pre_batch_context_known(self):
        ctx = self.store.pre_batch_context(attempter='alice')
        self.assertIn('attempter_risk_tier', ctx)
        self.assertIn('attempter_flag_rate', ctx)


class TestCrossBatchSimilarity(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmpdir, 'beet'))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_cross_batch_detects_match(self):
        shared_text = 'the quick brown fox jumps over the lazy dog in the park ' * 5

        # Batch 1
        r1 = [_make_result('t1', attempter='alice', occupation='eng')]
        self.store.record_batch(r1, {'t1': shared_text}, batch_id='b1')

        # Batch 2 with nearly identical text
        r2 = [_make_result('t2', attempter='bob', occupation='eng')]
        flags = self.store.cross_batch_similarity(r2, {'t2': shared_text + ' today'})
        self.assertTrue(len(flags) > 0)
        self.assertEqual(flags[0]['current_id'], 't2')
        self.assertEqual(flags[0]['historical_id'], 't1')

    def test_cross_batch_skips_same_attempter(self):
        text = 'the quick brown fox jumps over the lazy dog ' * 5

        r1 = [_make_result('t1', attempter='alice', occupation='eng')]
        self.store.record_batch(r1, {'t1': text}, batch_id='b1')

        r2 = [_make_result('t2', attempter='alice', occupation='eng')]
        flags = self.store.cross_batch_similarity(r2, {'t2': text})
        self.assertEqual(len(flags), 0)

    def test_cross_batch_different_text_no_match(self):
        r1 = [_make_result('t1', attempter='alice', occupation='eng')]
        self.store.record_batch(r1, {'t1': 'alpha beta gamma delta epsilon ' * 5},
                                batch_id='b1')

        r2 = [_make_result('t2', attempter='bob', occupation='eng')]
        flags = self.store.cross_batch_similarity(
            r2, {'t2': 'one two three four five six seven ' * 5})
        self.assertEqual(len(flags), 0)


class TestConfirmation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmpdir, 'beet'))
        results = [_make_result('t1', attempter='alice', determination='AMBER')]
        self.store.record_batch(results, {'t1': 'text here'}, batch_id='b1')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_record_confirmation(self):
        self.store.record_confirmation('t1', 'ai', verified_by='reviewer_A',
                                        notes='clear template match')
        lines = self.store.confirmed_path.read_text().strip().split('\n')
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec['task_id'], 't1')
        self.assertEqual(rec['ground_truth'], 'ai')
        self.assertEqual(rec['verified_by'], 'reviewer_A')
        self.assertEqual(rec['pipeline_determination'], 'AMBER')

    def test_confirmation_updates_attempter(self):
        self.store.record_confirmation('t1', 'ai', verified_by='rev')
        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['confirmed_ai'], 1)

    def test_confirmation_upgrades_risk_tier(self):
        # alice has 1 AMBER / 1 total = 100% flag rate
        # Before confirmation: HIGH (flag_rate > 0.30)
        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['risk_tier'], 'HIGH')

        # After confirming AI: CRITICAL (confirmed_ai > 0 AND flag_rate > 0.50)
        self.store.record_confirmation('t1', 'ai', verified_by='rev')
        profiles = self.store._load_attempter_profiles()
        self.assertEqual(profiles['alice']['risk_tier'], 'CRITICAL')

    def test_config_total_confirmed(self):
        self.store.record_confirmation('t1', 'ai', verified_by='rev')
        config = json.loads(self.store.config_path.read_text())
        self.assertEqual(config['total_confirmed'], 1)


class TestCalibrationRebuild(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmpdir, 'beet'))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_rebuild_no_labels(self):
        result = self.store.rebuild_calibration()
        self.assertIsNone(result)

    def test_rebuild_snapshots_existing(self):
        # Write a dummy calibration file
        cal_data = {'global': {}, 'strata': {}, 'n_calibration': 5}
        with open(self.store.calibration_path, 'w') as f:
            json.dump(cal_data, f)

        # Record a batch + confirm enough human labels
        results = [_make_result(f't{i}', attempter=f'a{i}', confidence=0.1 + i*0.01)
                    for i in range(25)]
        text_map = {f't{i}': f'text {i}' for i in range(25)}
        self.store.record_batch(results, text_map, batch_id='b1')

        for i in range(25):
            self.store.record_confirmation(f't{i}', 'human', verified_by='rev')

        self.store.rebuild_calibration()

        # Should have a snapshot in calibration_history/
        history_dir = self.store.store_dir / 'calibration_history'
        snapshots = list(history_dir.glob('cal_*.json'))
        self.assertTrue(len(snapshots) >= 1)


class TestPrintAttempterHistory(unittest.TestCase):
    def test_no_profile(self):
        """Should print 'No history' without error."""
        _print_attempter_history({'profile': None, 'submissions': [], 'confirmations': []})

    def test_with_profile(self):
        """Should print without error."""
        history = {
            'profile': {
                'attempter': 'alice',
                'risk_tier': 'HIGH',
                'flag_rate': 0.5,
                'total_submissions': 10,
                'determinations': {'RED': 5, 'AMBER': 0, 'YELLOW': 0, 'GREEN': 5},
                'confirmed_ai': 1,
                'confirmed_human': 3,
                'occupations': ['eng'],
                'first_seen': '2026-01-01',
                'last_seen': '2026-03-01',
                'batches': ['b1', 'b2'],
            },
            'submissions': [
                {'task_id': 't1', 'determination': 'RED', 'confidence': 0.9, 'occupation': 'eng'},
            ],
            'confirmations': [
                {'task_id': 't1', 'ground_truth': 'ai', 'verified_by': 'reviewer'},
            ],
        }
        _print_attempter_history(history)  # Should not raise


if __name__ == '__main__':
    unittest.main()
