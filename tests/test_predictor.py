import unittest

from api.predictor import validate_prediction


class ConfidenceValidationTests(unittest.TestCase):
    def confidence(self, entries):
        result = {"arcs": [0, 0, 1], "route": [0, 1], "confidence": entries}
        return validate_prediction(result, obs_count=3, arc_count=2)[2]

    def test_fallback_arc_keeps_null_confidence(self):
        self.assertEqual(self.confidence([0.75, None, 1.0]).tolist(), [0.75, None, 1.0])

    def test_confidence_may_be_unavailable_for_the_whole_trip(self):
        self.assertIsNone(self.confidence(None))
        self.assertEqual(self.confidence([None, None, None]).tolist(), [None, None, None])

    def test_invalid_confidence_is_rejected(self):
        for entries in ([0.75], [0.75, float("nan"), None], [0.75, float("inf"), None],
                        [0.75, 1.1, None]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.confidence(entries)
