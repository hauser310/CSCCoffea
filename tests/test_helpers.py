import unittest
import helpers


class TestHelpers(unittest.TestCase):
    """Unit tester for helpers."""

    @classmethod
    def setUpClass(cls):
        """Sets up variables needed for testing."""
        cls.ids = {
            26: {"endcap": 1, "station": 1, "ring": 1, "chamber": 27},
            313: {"endcap": 1, "station": 2, "ring": 1, "chamber": 58},
            420: {"endcap": 1, "station": 2, "ring": 3, "chamber": 37},
            953: {"endcap": 1, "station": 4, "ring": 3, "chamber": 58},
            1337: {"endcap": 2, "station": 2, "ring": 1, "chamber": 58},
            2047: {"endcap": 2, "station": 4, "ring": 4, "chamber": 64},
        }

    def test_serial_to_endcap(self):
        for ch_id in self.ids:
            self.assertEqual(helpers.serial_to_endcap(ch_id), self.ids[ch_id]["endcap"])

    def test_serial_to_station(self):
        for ch_id in self.ids:
            self.assertEqual(
                helpers.serial_to_station(ch_id), self.ids[ch_id]["station"]
            )

    def test_serial_to_ring(self):
        for ch_id in self.ids:
            self.assertEqual(helpers.serial_to_ring(ch_id), self.ids[ch_id]["ring"])

    def test_serial_to_chamber(self):
        for ch_id in self.ids:
            self.assertEqual(
                helpers.serial_to_chamber(ch_id), self.ids[ch_id]["chamber"]
            )
