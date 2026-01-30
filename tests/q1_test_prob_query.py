import unittest
from BayesNet import BayesNet

class TestProbQuery(unittest.TestCase):
    def test_probquery_sprinkler(self):
        bn = BayesNet("./nets/sprinkler.json")

        self.assertEqual(bn.query_prob("Cloudy", "T", {}), 0.5)
        self.assertEqual(bn.query_prob("Cloudy", "F", {}), 0.5)

        self.assertEqual(bn.query_prob("Sprinkler", "T", {"Cloudy": "T"}), 0.5)
        self.assertEqual(bn.query_prob("Sprinkler", "T", {"Cloudy": "F"}), 0.9)
        self.assertEqual(bn.query_prob("Sprinkler", "F", {"Cloudy": "T"}), 0.5)
        self.assertEqual(bn.query_prob("Sprinkler", "F", {"Cloudy": "F"}), 0.1)
        self.assertEqual(bn.query_prob("Sprinkler", "F", {}), None)

        self.assertEqual(bn.query_prob("WetGrass", "T", 
                                       {"Sprinkler": "T", "Rain": "T"}), 1)
        self.assertEqual(bn.query_prob("WetGrass", "F", 
                                       {"Sprinkler": "T", "Rain": "T"}), 0)

        self.assertEqual(bn.query_prob("WetGrass", "T", 
                                       {"Sprinkler": "T", "Rain": "F"}), 0.1)
        self.assertEqual(bn.query_prob("WetGrass", "F", 
                                       {"Sprinkler": "T", "Rain": "F"}), 0.9)
        
    def test_probquery_books(self):
        bn = BayesNet("./nets/books.json")

        self.assertEqual(bn.query_prob("Honesty", "T", {}), 0.8)
        self.assertEqual(bn.query_prob("Honesty", "F", {}), 0.2)

        self.assertEqual(bn.query_prob("Quality", "1", {}), 0.1)
        self.assertEqual(bn.query_prob("Quality", "2", {}), 0.2)
        self.assertEqual(bn.query_prob("Quality", "3", {}), 0.4)
        self.assertEqual(bn.query_prob("Quality", "4", {}), 0.2)
        self.assertEqual(bn.query_prob("Quality", "5", {}), 0.1)

        self.assertEqual(bn.query_prob("Recommendation", "1",
                                       {"Honesty": "T", 
                                        "Quality": "1", 
                                        "Kindness": "5"}), 0.6)
        
        self.assertEqual(bn.query_prob("Recommendation", "2",
                                       {"Honesty": "T", 
                                        "Quality": "1", 
                                        "Kindness": "5"}), 0.125)
        
        self.assertEqual(bn.query_prob("Recommendation", "3",
                                       {"Honesty": "T", 
                                        "Quality": "1", 
                                        "Kindness": "5"}), 0.125)

        self.assertEqual(bn.query_prob("Recommendation", "4",
                                       {"Honesty": "T", 
                                        "Quality": "1", 
                                        "Kindness": "5"}), 0.125)
        
        self.assertEqual(bn.query_prob("Recommendation", "5",
                                       {"Honesty": "T", 
                                        "Quality": "1", 
                                        "Kindness": "5"}), 0.025)


