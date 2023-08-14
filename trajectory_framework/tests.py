import unittest
import numpy as np
from unittest.mock import patch

from framework import VehicleData


class TestVehicleData(unittest.TestCase):
    def setUp(self):
        self.obj = VehicleData('data.npy')

    def test_by_id(self):
        segment = self.obj.by_id(0)
        self.assertIsInstance(segment, np.ndarray)

    def test_filter(self):
        def length(trajectory):
            return len(trajectory)

        filtered_segments = self.obj.filter(length)
        self.assertIsInstance(filtered_segments, list)

    @patch('matplotlib.pyplot.show')
    def test_plot(self, mock_show):
        self.obj.plot()
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
