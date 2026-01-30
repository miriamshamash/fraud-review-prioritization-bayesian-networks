import unittest
from BayesNet import BayesNet

PLACES = 2

class TestEnumerateAsk(unittest.TestCase):

    # Test Sprinkler
    def test_eask_sprinkler(self):
        bn = BayesNet("./nets/sprinkler.json")

        res = bn.enumerate_ask("WetGrass", {})
        self.assertAlmostEqual(res["T"], 0.353, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.647, places=PLACES)

        res = bn.enumerate_ask("WetGrass", {"Rain": "T"})
        self.assertAlmostEqual(res["T"], 0.622, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.378, places=PLACES)

        res = bn.enumerate_ask("WetGrass", {"Rain": "F"})
        self.assertAlmostEqual(res["T"], 0.084, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.916, places=PLACES)

        res = bn.enumerate_ask("Cloudy", {"WetGrass": "T"})
        self.assertAlmostEqual(res["T"], 0.639, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.361, places=PLACES)

        res = bn.enumerate_ask("Cloudy", {"WetGrass": "F"})
        self.assertAlmostEqual(res["T"], 0.424, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.576, places=PLACES)

        res = bn.enumerate_ask("WetGrass", {"Cloudy": "F", "Rain": "T"})
        self.assertAlmostEqual(res["T"], 0.91, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.09, places=PLACES)

        res = bn.enumerate_ask("WetGrass", {"Cloudy": "F", "Rain": "T", "Sprinkler": "T"})
        self.assertAlmostEqual(res["T"], 1, places=PLACES)
        self.assertAlmostEqual(res["F"], 0, places=PLACES)

    # Test Books
    def test_eask_books(self):
        bn = BayesNet("./nets/books.json")

        res = bn.enumerate_ask("Recommendation", {})
        self.assertAlmostEqual(res["1"], 0.07, places=PLACES)
        self.assertAlmostEqual(res["2"], 0.141, places=PLACES)
        self.assertAlmostEqual(res["3"], 0.293, places=PLACES)
        self.assertAlmostEqual(res["4"], 0.249, places=PLACES)
        self.assertAlmostEqual(res["5"], 0.248, places=PLACES)

        res = bn.enumerate_ask("Recommendation", {"Quality": "1"})
        self.assertAlmostEqual(res["1"], 0.602, places=PLACES)
        self.assertAlmostEqual(res["2"], 0.089, places=PLACES)
        self.assertAlmostEqual(res["3"], 0.101, places=PLACES)
        self.assertAlmostEqual(res["4"], 0.125, places=PLACES)
        self.assertAlmostEqual(res["5"], 0.083, places=PLACES)

        res = bn.enumerate_ask("Recommendation", {"Quality": "5"})
        self.assertAlmostEqual(res["1"], 0.002, places=PLACES)
        self.assertAlmostEqual(res["2"], 0.01, places=PLACES)
        self.assertAlmostEqual(res["3"], 0.057, places=PLACES)
        self.assertAlmostEqual(res["4"], 0.134, places=PLACES)
        self.assertAlmostEqual(res["5"], 0.796, places=PLACES)

        res = bn.enumerate_ask("Recommendation", {"Quality": "1", "Kindness": "5"})
        self.assertAlmostEqual(res["1"], 0.48, places=PLACES)
        self.assertAlmostEqual(res["2"], 0.1, places=PLACES)
        self.assertAlmostEqual(res["3"], 0.1, places=PLACES)
        self.assertAlmostEqual(res["4"], 0.1, places=PLACES)
        self.assertAlmostEqual(res["5"], 0.22, places=PLACES)

        res = bn.enumerate_ask("Recommendation", {"Quality": "3", "Kindness": "5"})
        self.assertAlmostEqual(res["1"], 0, places=PLACES)
        self.assertAlmostEqual(res["2"], 0, places=PLACES)
        self.assertAlmostEqual(res["3"], 0.52, places=PLACES)
        self.assertAlmostEqual(res["4"], 0.16, places=PLACES)
        self.assertAlmostEqual(res["5"], 0.32, places=PLACES)

        res = bn.enumerate_ask("Recommendation", {"Quality": "5", "Kindness": "1", "Honesty": "F"})
        self.assertAlmostEqual(res["1"], 0.25, places=PLACES)
        self.assertAlmostEqual(res["2"], 0.5, places=PLACES)
        self.assertAlmostEqual(res["3"], 0.25, places=PLACES)
        self.assertAlmostEqual(res["4"], 0, places=PLACES)
        self.assertAlmostEqual(res["5"], 0, places=PLACES)

        res = bn.enumerate_ask("Honesty", {"Recommendation": "1"})
        self.assertAlmostEqual(res["T"], 0.86, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.14, places=PLACES)

        res = bn.enumerate_ask("Honesty", {"Recommendation": "1", "Quality": "1"})
        self.assertAlmostEqual(res["T"], 0.983, places=PLACES)
        self.assertAlmostEqual(res["F"], 0.017, places=PLACES)

        res = bn.enumerate_ask("Honesty", {"Recommendation": "1", "Quality": "1", "Kindness": "5"})
        self.assertAlmostEqual(res["T"], 1, places=PLACES)
        self.assertAlmostEqual(res["F"], 0, places=PLACES)