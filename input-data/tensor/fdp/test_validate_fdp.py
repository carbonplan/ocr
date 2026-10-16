"""Small scientific correctness checks; no downloads or large fixtures."""

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer
from shapely.geometry import box
from validate_fdp import (
    assign_huc,
    balanced_metrics,
    dollars,
    index_xy,
    score_points,
    top_area_recall,
)


class ValidationChecks(unittest.TestCase):
    def test_top_area_metric_breaks_score_ties(self):
        background = pd.DataFrame({'huc2': ['01'] * 10, 'fdp': [1.0] * 4 + [0.0] * 6})
        positive = pd.DataFrame({'huc2': ['01'] * 2, 'fdp': [1.0, 0.0]})
        self.assertAlmostEqual(top_area_recall(positive, background, 0.1), 0.125)
        self.assertAlmostEqual(top_area_recall(positive, background, 0.2), 0.25)

    def test_overlapping_watersheds_do_not_duplicate_samples(self):
        frame = pd.DataFrame({'x': [5, 15], 'y': [5, 5]})
        hucs = gpd.GeoDataFrame(
            {'huc2': ['01', '02']}, geometry=[box(0, 0, 10, 10), box(4, 0, 20, 10)], crs=5070
        )
        assigned = assign_huc(frame, hucs)
        self.assertEqual(len(assigned), 2)
        self.assertTrue(pd.isna(assigned.huc2.iloc[0]))
        self.assertTrue(assigned.ambiguous_huc.iloc[0])
        self.assertEqual(assigned.huc2.iloc[1], '02')

    def test_dollars_preserves_unknowns(self):
        values = dollars(pd.Series(['1.5K', '2M', '0.00K', None, 'unknown', '3B']))
        np.testing.assert_allclose(values.iloc[[0, 1, 2, 5]], [1500, 2e6, 0, 3e9])
        self.assertTrue(values.iloc[3:5].isna().all())

    def test_grid_edges_do_not_wrap(self):
        row, col, valid = index_xy(
            [50, 150, -1, 200], [150, 50, 150, 50], Affine(100, 0, 0, 0, -100, 200), (2, 2)
        )
        np.testing.assert_array_equal(row, [0, 1, 0, 1])
        np.testing.assert_array_equal(col, [0, 1, -1, 2])
        np.testing.assert_array_equal(valid, [True, True, False, False])

    def test_background_weights_match_huc_distribution(self):
        positive = pd.DataFrame({'huc2': ['01', '01', '02'], 'fdp': [0.9, 0.8, 0.7]})
        background = pd.DataFrame({'huc2': ['01', '02', '02'], 'fdp': [0.2, 0.1, 0.3]})
        metrics, joined = balanced_metrics(positive, background)
        self.assertEqual(metrics['auc'], 1)
        np.testing.assert_allclose(joined.groupby('label').weight.sum(), [3, 3])
        np.testing.assert_allclose(joined[joined.label == 0].groupby('huc2').weight.sum(), [2, 1])

    def test_sampling_drops_nodata_and_outside_coordinates(self):
        # Build a real projected 2x2 raster around a CONUS location and transform
        # its known cell centers back to NOAA's lon/lat coordinate system.
        forward = Transformer.from_crs(4326, 5070, always_xy=True)
        backward = Transformer.from_crs(5070, 4326, always_xy=True)
        x, y = forward.transform(-90, 35)
        transform = Affine(100, 0, x, 0, -100, y)
        lon, lat = backward.transform([x + 50, x + 150, x + 250], [y - 50] * 3)
        frame = pd.DataFrame({'lon': lon, 'lat': lat, 'conus_state': True})
        hucs = gpd.GeoDataFrame(
            {'huc2': ['01']}, geometry=[box(x - 1, y - 201, x + 301, y + 1)], crs=5070
        )
        scored, qa = score_points(frame, np.array([[0.8, np.nan], [0.2, 0.3]]), transform, hucs)
        self.assertEqual(len(scored), 1)
        self.assertAlmostEqual(scored.fdp.iloc[0], 0.8)
        self.assertEqual(qa.qa_status.tolist(), ['ok', 'raster_nodata', 'outside_raster_extent'])


if __name__ == '__main__':
    unittest.main()
