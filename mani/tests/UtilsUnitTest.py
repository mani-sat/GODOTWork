import unittest
import numpy as np
from utils import compute_projection_matrix, project_point

class TestUtils(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
    # Test cases for "compute_projection_matrix" method
    def test_compute_projection_matrix(self):
        basis = np.array([1, 1, 1], dtype=np.float64)
        expected_result = np.array([[2/3, -1/3, -1/3], [-1/3, 2/3, -1/3], [-1/3, -1/3, 2/3]])
        result = compute_projection_matrix(basis)
        np.testing.assert_array_almost_equal(result, expected_result)
        # Check if the result is a 3x3 matrix
        self.assertEqual(result.shape, (3, 3))
        
    def test_compute_projection_matrix_invalid_input(self):
        basis = np.array([1, 2], dtype=np.float64)
        with self.assertRaises(ValueError):
            compute_projection_matrix(basis)
    
    def test_compute_projection_matrix_zero_vector(self):
        basis = np.array([0, 0, 0], dtype=np.float64)
        with self.assertRaises(ValueError):
            compute_projection_matrix(basis)

    # Test cases for "project_point" method
    def test_project_point(self):
        P = np.eye(3)  # Identity matrix
        point = np.array([1, 2, 3], dtype=np.float64)
        expected_result = np.array([1, 2, 3])
        result = project_point(P, point)
        np.testing.assert_array_almost_equal(result, expected_result)
    
    def test_project_point_invalid_input(self):
        P = np.eye(3)
        point = np.array([1, 2], dtype=np.float64)
        with self.assertRaises(ValueError):
            project_point(P, point)
    
    def test_project_point_zero_vector(self):
        P = np.eye(3)
        point = np.array([0, 0, 0], dtype=np.float64)
        expected_result = np.array([0, 0, 0])
        result = project_point(P, point)
        np.testing.assert_array_almost_equal(result, expected_result)

if __name__ == "__main__":
    unittest.main()