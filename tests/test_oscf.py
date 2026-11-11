"""Regression tests for counterfactual projection of segmentation masks.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oscf_cv_mvp import (
    MaskProjectionError, Observation, audit_scene, encode_scene,
    project_onto_support, rule_associator_features, translate_mask,
)


class MaskProjectionTests(unittest.TestCase):
    def make_objects(self, support_top=42.0):
        mask = np.zeros((100, 120), dtype=bool)
        mask[20:39, 50:56] = True
        cup = Observation('cup0', 'cup', 0.95, (48, 18, 58, 42), mask=mask)
        table = Observation('table0', 'table', 0.95, (35, support_top, 85, support_top + 11))
        book = Observation('book0', 'book', 0.9, (70, 65, 90, 80))
        return cup, table, book

    def test_no_move_preserves_mask_and_zero_residual(self):
        cup, table, book = self.make_objects()
        projected = project_onto_support(cup, table)
        self.assertIs(projected.mask, cup.mask)
        self.assertEqual(projected.bbox, cup.bbox)
        scores = rule_associator_features([cup, table, book], (120, 100), cup, table)
        self.assertEqual(scores['status'], 'MEASURED')
        self.assertEqual(scores['associator_counterfactual_delta'], 0.0)
        self.assertEqual(scores['linear_8d_counterfactual_delta'], 0.0)

    def test_motion_translates_mask_without_changing_area(self):
        cup, table, book = self.make_objects(support_top=49)
        projected = project_onto_support(cup, table)
        self.assertEqual(projected.bbox, (48.0, 25.0, 58.0, 49.0))
        self.assertEqual(int(projected.mask.sum()), int(cup.mask.sum()))
        self.assertTrue(np.all(projected.mask[27:46, 50:56]))
        self.assertEqual(
            encode_scene([cup, table, book], (120, 100))['cup0']['axis_proxy_valid'],
            encode_scene([projected, table, book], (120, 100))['cup0']['axis_proxy_valid'],
        )

    def test_fractional_motion_uses_nearest_pixel_for_mask(self):
        cup, table, book = self.make_objects(support_top=49.2)
        projected = project_onto_support(cup, table)
        self.assertAlmostEqual(projected.bbox[1], 25.2)
        self.assertTrue(np.all(projected.mask[27:46, 50:56]))

    def test_clipped_mask_does_not_create_misleading_residual(self):
        mask = np.zeros((100, 120), dtype=bool)
        mask[10:20, 8:12] = True
        cup = Observation('cup0', 'cup', 0.95, (0, 8, 12, 21), mask=mask)
        table = Observation('table0', 'table', 0.95, (111, 75, 119, 85))
        book = Observation('book0', 'book', 0.9, (70, 65, 90, 80))
        with self.assertRaises(MaskProjectionError):
            project_onto_support(cup, table)
        scores = rule_associator_features([cup, table, book], (120, 100), cup, table)
        self.assertEqual(scores['status'], 'INSUFFICIENT')
        self.assertEqual(audit_scene([cup, table, book], (120, 100), [('cup', 'table')])['decision'], 'WARN')

    def test_empty_mask_translation(self):
        mask = np.zeros((20, 20), dtype=bool)
        self.assertFalse(translate_mask(mask, 2, 4).any())


if __name__ == '__main__':
    unittest.main()
