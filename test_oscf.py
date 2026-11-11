import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oscf_cv_mvp import (oct_mul, associator, create_synthetic_scene,
                          encode_scene, audit_scene, TemporalCache)


class OctonionTests(unittest.TestCase):
    def test_unit_multiplication_norm(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            a = rng.normal(size=8)
            b = rng.normal(size=8)
            self.assertAlmostEqual(np.linalg.norm(oct_mul(a, b)),
                                   np.linalg.norm(a)*np.linalg.norm(b), places=10)

    def test_nonassociative_and_alternative(self):
        a, b, c = np.eye(8)[1], np.eye(8)[2], np.eye(8)[4]
        # Some triples lie within a quaternion subalgebra; this one must not.
        self.assertGreater(np.linalg.norm(associator(a, b, c)), 0.1)
        self.assertLess(np.linalg.norm(associator(a, a, c)), 1e-12)
        self.assertLess(np.linalg.norm(associator(a, b, b)), 1e-12)


class SceneTests(unittest.TestCase):
    def test_synthetic_contract_satisfied(self):
        image, obs = create_synthetic_scene(Path('.'), floating_cup=False)
        r = audit_scene(obs, image.size, [('cup', 'table')])
        self.assertEqual(r['decision'], 'PASS')
        self.assertEqual(r['findings'][0]['octonion_features']['status'], 'MEASURED')
        self.assertAlmostEqual(r['findings'][0]['octonion_features']['associator_counterfactual_delta'],
                               0.0)

    def test_synthetic_contract_violation(self):
        image, obs = create_synthetic_scene(Path('.'), floating_cup=True)
        r = audit_scene(obs, image.size, [('cup', 'table')])
        self.assertEqual(r['decision'], 'WARN')
        self.assertGreater(r['findings'][0]['octonion_features']['associator_counterfactual_delta'], 0.0)

    def test_unconstrained_not_confused_with_pass(self):
        image, obs = create_synthetic_scene(Path('.'), floating_cup=True)
        r = audit_scene(obs, image.size, [])
        self.assertEqual(r['decision'], 'UNCONSTRAINED')

    def test_missing_depth_is_explicit(self):
        image, obs = create_synthetic_scene(Path('.'), floating_cup=False)
        codes = encode_scene(obs, image.size)
        self.assertFalse(codes['cup_0']['depth_valid'])
        self.assertEqual(codes['cup_0']['q'].shape, (8,))
        self.assertEqual(codes['cup_0']['q'][3], 0.0)

    def test_temporal_cache_requires_repeat_track_id(self):
        image, obs = create_synthetic_scene(Path('.'), floating_cup=False)
        cache = TemporalCache()
        self.assertEqual(cache.update(0, encode_scene(obs, image.size)), [])
        changes = cache.update(1, encode_scene(obs, image.size))
        self.assertEqual(len(changes), 3)
        self.assertEqual(changes[0]['xy_displacement_norm'], 0)


if __name__ == '__main__':
    unittest.main()
