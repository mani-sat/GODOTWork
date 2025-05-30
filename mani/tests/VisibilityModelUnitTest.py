import unittest
import numpy as np
from VisibilityModel import VisibilityModel

class TestVisibilityModel(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        # Initialize the EvaluateContact class with a radius
        # of 1 and 2 for the moon and earth respectively.
        self.evaluator = VisibilityModel(moon_radius=1.0, earth_radius=2.0)

    # Test cases for "point_within_sphere" method
    def test_point_within_sphere(self):
        point = np.array([1, 0, 0])
        centre = np.array([0, 0, 0])  
        radius = 5
        self.assertTrue(self.evaluator.point_within_sphere(point,centre,radius))

    def test_point_outside_sphere(self):
        point = np.array([10, 0, 0])
        centre = np.array([0, 0, 0])
        radius = 5
        self.assertFalse(self.evaluator.point_within_sphere(point,centre,radius))
    
    def test_point_on_sphere(self):
        point = np.array([10, 0, 0])
        centre = np.array([0, 0, 0])
        radius = 10
        self.assertFalse(self.evaluator.point_within_sphere(point,centre,radius))
    
    # Test cases for "calculate_within" method
    def test_calculate_within(self):
        point = np.array([1, 0, 0], dtype=np.float64)
        basis = np.array([0, 1, 0], dtype=np.float64)
        sphere_centre = np.array([0, 0, 0], dtype=np.float64)
        radius = 5
        self.assertTrue(self.evaluator.calculate_within(basis, point, sphere_centre, radius))
    
    def test_calculate_within_outside(self):
        point = np.array([6, 0, 0], dtype=np.float64)
        basis = np.array([0, 1, 0], dtype=np.float64)
        sphere_centre = np.array([0, 0, 0], dtype=np.float64)
        radius = 5
        self.assertFalse(self.evaluator.calculate_within(basis, point, sphere_centre, radius))
    
    def test_calculate_within_on_sphere(self):
        point = np.array([10, 0, 0], dtype=np.float64)
        basis = np.array([0, 1, 0], dtype=np.float64)
        sphere_centre = np.array([0, 0, 0], dtype=np.float64)
        radius = 10
        self.assertFalse(self.evaluator.calculate_within(basis, point, sphere_centre, radius))

    def test_calculate_within_shifted_centre(self):
        point = np.array([10, 0, 0], dtype=np.float64)
        basis = np.array([0, 1, 0], dtype=np.float64)
        sphere_centre = np.array([8, 0, 0], dtype=np.float64)
        radius = 3
        self.assertTrue(self.evaluator.calculate_within(basis, point, sphere_centre, radius))
    
    # Test cases for "los_from_gs_to_sc" method
    def test_los_from_gs_to_sc_due_to_dot(self):
        spacecraft = np.array([1, 1, 0], dtype=np.float64) 
        groundstation = np.array([2, 0, 0], dtype=np.float64)
        self.assertTrue(self.evaluator.los_from_gs_to_sc(spacecraft, groundstation))
    
    def test_los_from_gs_to_sc_los_due_to_not_within(self):
        spacecraft = np.array([1, 4, 0], dtype=np.float64)
        groundstation = np.array([-3, 0, 0], dtype=np.float64)
        self.assertTrue(self.evaluator.los_from_gs_to_sc(spacecraft, groundstation))

    def test_los_from_gs_to_sc_no_los_due_to_within(self):
        spacecraft = np.array([2, 0, 0], dtype=np.float64)
        groundstation = np.array([-10, 0, 0], dtype=np.float64)
        self.assertFalse(self.evaluator.los_from_gs_to_sc(spacecraft, groundstation))

    # Test cases for "sun_light_on_spacecraft" method
    def test_sun_light_on_spacecraft(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([-2, -2, 0], dtype=np.float64)
        moon_sc = np.array([0, 2, 0], dtype=np.float64)
        self.assertTrue(self.evaluator.sun_light_on_spacecraft(moon_sun, moon_earth, moon_sc))
    
    def test_sun_light_on_spacecraft_hidden_moon(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([-2, -2, 0], dtype=np.float64)
        moon_sc = np.array([2, 0, 0], dtype=np.float64)
        self.assertFalse(self.evaluator.sun_light_on_spacecraft(moon_sun, moon_earth, moon_sc))

    def test_sun_light_on_spacecraft_hidden_earth(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([-2, -2, 0], dtype=np.float64)
        moon_sc = np.array([0, -3, 0], dtype=np.float64)
        self.assertFalse(self.evaluator.sun_light_on_spacecraft(moon_sun, moon_earth, moon_sc))

    def test_sun_light_on_spacecraft_in_front(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([2, 2, 0], dtype=np.float64)
        moon_sc = np.array([-3, -3, 0], dtype=np.float64)
        self.assertTrue(self.evaluator.sun_light_on_spacecraft(moon_sun, moon_earth, moon_sc))

    # Test cases for "sun_light_on_moon" method
    def test_sun_light_on_moon(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([-2, -2, 0], dtype=np.float64)
        moon_sc = np.array([-0.1, 2, 0], dtype=np.float64)
        self.assertTrue(self.evaluator.sun_light_on_moon(moon_sun, moon_earth, moon_sc))

    def test_sun_light_on_moon_dark_side(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([-2, -2, 0], dtype=np.float64)
        moon_sc = np.array([0.1, 2, 0], dtype=np.float64)
        self.assertFalse(self.evaluator.sun_light_on_moon(moon_sun, moon_earth, moon_sc))
    
    def test_sun_light_on_moon_hidden_moon(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([-2, -2, 0], dtype=np.float64)
        moon_sc = np.array([2, 0, 0], dtype=np.float64)
        self.assertFalse(self.evaluator.sun_light_on_moon(moon_sun, moon_earth, moon_sc))

    def test_sun_light_on_moon_hidden_earth(self):
        moon_sun = np.array([-10, 0, 0], dtype=np.float64)
        moon_earth = np.array([-3, -3, 0], dtype=np.float64)
        moon_sc = np.array([0, -3, 0], dtype=np.float64)
        self.assertFalse(self.evaluator.sun_light_on_moon(moon_sun, moon_earth, moon_sc))

if __name__ == "__main__":
    unittest.main()