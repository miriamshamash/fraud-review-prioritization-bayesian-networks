import unittest
from BayesNet import BayesNet

PLACES = 2

class TestEnumerateAll(unittest.TestCase):
    def test_enumerate_all_sprinkler(self):
        bn = BayesNet("./nets/sprinkler.json")
        vars = list(bn.nodes)

        # Wet Grass
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"WetGrass": "T"}), 
                               0.353, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(
                vars, {"WetGrass": "F"}), 
                               0.647, 
                               places=PLACES)
        
        # WetGrass | Rain=T
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"WetGrass": "T", "Rain":"T"}), 
                               0.311, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"WetGrass": "F", "Rain":"T"}), 
                               0.189, 
                               places=PLACES)
        
        # Cloudy | WetGrass = T
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Cloudy": "T", "WetGrass":"T"}), 
                               0.226, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Cloudy": "F", "WetGrass":"T"}), 
                               0.127, 
                               places=PLACES)
        
        # WetGrass | Cloudy = F, Rain = T
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"WetGrass": "T", "Cloudy":"F", "Rain":"T"}), 
                               0.091, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"WetGrass": "F", "Cloudy":"F", "Rain":"T"}), 
                               0.009, 
                               places=PLACES)
    
    def test_enumerate_all_books(self):
        bn = BayesNet("./nets/books.json")
        vars = list(bn.nodes)


        # Recommendation | Quality 1
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Recommendation": "1", "Quality": "1"}), 
                               0.060, 
                               places=PLACES)

        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Recommendation": "2" , "Quality": "1"}), 
                               0.009, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Recommendation": "3", "Quality": "1"}), 
                               0.010, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Recommendation": "4", "Quality": "1"}), 
                               0.0125, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Recommendation": "5", "Quality": "1"}), 
                               0.0083, 
                               places=PLACES)
        
        # Honesty | Recommendation = 1, Quality = 1
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Honesty": "T", "Recommendation": "1", "Quality": "1"}), 
                               0.0592, 
                               places=PLACES)
        
        self.assertAlmostEqual(
            bn.enumerate_all(vars, {"Honesty": "F", "Recommendation": "1", "Quality": "1"}), 
                               0.001, 
                               places=PLACES)
        


