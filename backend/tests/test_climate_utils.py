import unittest
from backend.climate_utils import get_offline_climate

class TestGetOfflineClimate(unittest.TestCase):

    def test_none_inputs(self):
        self.assertEqual(get_offline_climate(None, None), "Unknown")
        self.assertEqual(get_offline_climate(10, None), "Unknown")
        self.assertEqual(get_offline_climate(None, 10), "Unknown")

    def test_invalid_types(self):
        with self.assertRaises(ValueError):
            get_offline_climate("abc", "def")
        with self.assertRaises(ValueError):
            get_offline_climate(10, "def")
        with self.assertRaises(ValueError):
            get_offline_climate("abc", 10)

    def test_tropical_general(self):
        self.assertEqual(get_offline_climate(10, 50), "Tropical")
        self.assertEqual(get_offline_climate(23.5, 50), "Tropical")
        self.assertEqual(get_offline_climate(-23.5, 50), "Tropical")
        self.assertEqual(get_offline_climate(0, 0), "Tropical")

    def test_tropical_india_humid(self):
        self.assertEqual(get_offline_climate(10, 68), "Tropical (Humid)")
        self.assertEqual(get_offline_climate(19.9, 80), "Tropical (Humid)")
        self.assertEqual(get_offline_climate(0, 97), "Tropical (Humid)")
        self.assertEqual(get_offline_climate(-10, 80), "Tropical (Humid)")
        self.assertEqual(get_offline_climate(-19.9, 97), "Tropical (Humid)")

    def test_tropical_india_monsoon(self):
        self.assertEqual(get_offline_climate(20, 80), "Tropical (Monsoon-prone)")
        self.assertEqual(get_offline_climate(23.5, 97), "Tropical (Monsoon-prone)")
        self.assertEqual(get_offline_climate(23.5, 68), "Tropical (Monsoon-prone)")
        self.assertEqual(get_offline_climate(-20, 80), "Tropical (Monsoon-prone)")

    def test_subtropical_general(self):
        self.assertEqual(get_offline_climate(23.6, 50), "Humid Subtropical")
        self.assertEqual(get_offline_climate(35, 50), "Humid Subtropical")
        self.assertEqual(get_offline_climate(-35, 50), "Humid Subtropical")
        self.assertEqual(get_offline_climate(-23.6, 50), "Humid Subtropical")
        self.assertEqual(get_offline_climate(30, 81), "Humid Subtropical")
        self.assertEqual(get_offline_climate(30, 59), "Humid Subtropical")

    def test_subtropical_dry_arid(self):
        self.assertEqual(get_offline_climate(25, 60), "Dry / Arid")
        self.assertEqual(get_offline_climate(35, 80), "Dry / Arid")
        self.assertEqual(get_offline_climate(-25, 70), "Dry / Arid")
        self.assertEqual(get_offline_climate(23.6, 70), "Dry / Arid")

    def test_temperate(self):
        self.assertEqual(get_offline_climate(35.1, 0), "Temperate")
        self.assertEqual(get_offline_climate(66.5, 0), "Temperate")
        self.assertEqual(get_offline_climate(-66.5, 0), "Temperate")
        self.assertEqual(get_offline_climate(-35.1, 0), "Temperate")

    def test_polar(self):
        self.assertEqual(get_offline_climate(66.6, 0), "Polar / Cold")
        self.assertEqual(get_offline_climate(90, 0), "Polar / Cold")
        self.assertEqual(get_offline_climate(-90, 0), "Polar / Cold")
        self.assertEqual(get_offline_climate(-66.6, 0), "Polar / Cold")

    def test_string_inputs(self):
        self.assertEqual(get_offline_climate("20", "80"), "Tropical (Monsoon-prone)")
        self.assertEqual(get_offline_climate("-30", "70"), "Dry / Arid")
        self.assertEqual(get_offline_climate("23.5", "68"), "Tropical (Monsoon-prone)")
        self.assertEqual(get_offline_climate("35", "80"), "Dry / Arid")
        self.assertEqual(get_offline_climate("66.5", "0"), "Temperate")

if __name__ == '__main__':
    unittest.main()
