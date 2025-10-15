from unittest.mock import patch
import unittest
import numpy as np
from comp_dart.targets.surrogate import SurrogateModelTarget
from comp_dart.targets.linear_mixture import LinearMixtureTarget
from comp_dart.targets.density_combined import DensityTarget
from comp_dart.core.interfaces import TargetResult


class TestSurrogateModelTarget(unittest.TestCase):
    """Test SurrogateModelTarget class"""
    
    def test_surrogate_model_target_initialization(self):
        """Test SurrogateModelTarget initialization"""
        models = ["model1", "model2"]
        target = SurrogateModelTarget(models=models, requires_structure=True)
        
        self.assertEqual(target.models, models)
        self.assertTrue(target.requires_structure)
        self.assertIsNone(target.mean)
        self.assertIsNone(target.std)
        
    def test_surrogate_model_target_with_normalization(self):
        """Test SurrogateModelTarget with normalization parameters"""
        models = ["model1", "model2"]
        target = SurrogateModelTarget(models=models, mean=0.5, std=0.1, requires_structure=False)
        
        self.assertEqual(target.mean, 0.5)
        self.assertEqual(target.std, 0.1)
        self.assertFalse(target.requires_structure)
        
    def test_surrogate_model_target_invalid_normalization(self):
        """Test SurrogateModelTarget with invalid normalization parameters"""
        models = ["model1", "model2"]
        with self.assertRaises(ValueError):
            SurrogateModelTarget(models=models, mean=0.5, std=None)
            
        with self.assertRaises(ValueError):
            SurrogateModelTarget(models=models, mean=None, std=0.1)
            
    def test_surrogate_model_target_predict_without_structure(self):
        """Test SurrogateModelTarget predict method without structure"""
        models = ["model1", "model2"]
        target = SurrogateModelTarget(models=models, requires_structure=False)
        composition = np.array([0.5, 0.3, 0.2])
        
        # Should not raise an error since requires_structure is False
        result = target.predict(composition)
        
    def test_surrogate_model_target_predict_with_structure(self):
        """Test SurrogateModelTarget predict method with structure"""
        models = ["model1", "model2"]
        target = SurrogateModelTarget(models=models, requires_structure=True)
        composition = np.array([0.5, 0.3, 0.2])
        structures = ["test_structure1", "test_structure2"]
        # Mock the pred function to avoid actual model loading
        with patch("comp_dart.targets.surrogate.pred") as mock_pred:
            mock_pred.return_value = np.array([1.0])
            result = target.predict(composition, structures)
            self.assertIsInstance(result, TargetResult)
            result = target.predict(composition, structures)
        
    def test_surrogate_model_target_predict_missing_structure(self):
        """Test SurrogateModelTarget predict method missing required structure"""
        models = ["model1", "model2"]
        target = SurrogateModelTarget(models=models, requires_structure=True)
        composition = np.array([0.5, 0.3, 0.2])
        
        # Should raise an error since structure is required but not provided
        with self.assertRaises(ValueError):
            target.predict(composition)
            
    def test_surrogate_model_target_no_models(self):
        """Test SurrogateModelTarget with no models"""
        target = SurrogateModelTarget(models=[], requires_structure=False)
        composition = np.array([0.5, 0.3, 0.2])
        
        # Should raise an error since no models are provided
        with self.assertRaises(ValueError):
            target.predict(composition)


class TestLinearMixtureTarget(unittest.TestCase):
    """Test LinearMixtureTarget class"""
    
    def test_linear_mixture_target_initialization(self):
        """Test LinearMixtureTarget initialization"""
        element_properties = {"Fe": 7.87, "Ni": 8.91, "Co": 8.90}
        target = LinearMixtureTarget(element_properties=element_properties, requires_structure=False)
        
        self.assertEqual(target.element_properties, element_properties)
        self.assertFalse(target.requires_structure)
        
    def test_linear_mixture_target_predict(self):
        """Test LinearMixtureTarget predict method"""
        element_properties = {"Fe": 7.87, "Ni": 8.91, "Co": 8.90}
        target = LinearMixtureTarget(element_properties=element_properties, requires_structure=False)
        composition = np.array([0.5, 0.3, 0.2])  # Fe=50%, Ni=30%, Co=20%
        
        result = target.predict(composition)
        
        # Calculate expected value: 0.5*7.87 + 0.3*8.91 + 0.2*8.90 = 7.998
        expected_value = 0.5 * 7.87 + 0.3 * 8.91 + 0.2 * 8.90
        self.assertAlmostEqual(result.value, expected_value)
        self.assertEqual(result.metadata["method"], "linear_mixture")


class TestDensityTarget(unittest.TestCase):
    """Test DensityTarget class"""
    
    def test_density_target_initialization(self):
        """Test DensityTarget initialization"""
        element_densities = {"Fe": 7.87, "Ni": 8.91, "Co": 8.90}
        target = DensityTarget(element_densities=element_densities, requires_structure=True)
        
        self.assertEqual(target.element_densities, element_densities)
        self.assertTrue(target.requires_structure)
        self.assertIsNotNone(target.linear_target)
        
    def test_density_target_initialization_without_densities(self):
        """Test DensityTarget initialization without element densities"""
        target = DensityTarget(requires_structure=False, preferred_methods=["linear"])
        
        self.assertIsNone(target.element_densities)
        self.assertFalse(target.requires_structure)
        self.assertIsNone(target.linear_target)
        
    def test_density_target_predict_with_structure(self):
        """Test DensityTarget predict method with structure"""
        element_densities = {"Fe": 7.87, "Ni": 8.91, "Co": 8.90}
        target = DensityTarget(element_densities=element_densities, requires_structure=True)
        composition = np.array([0.5, 0.3, 0.2])
        structures = ["test_structure1", "test_structure2"]
        # Mock the pred function to avoid actual model loading
        with patch("comp_dart.targets.surrogate.pred") as mock_pred:
            mock_pred.return_value = np.array([1.0])
            result = target.predict(composition, structures)
            self.assertIsInstance(result, TargetResult)
        # Should try structure-based calculation first
        self.assertIn("structure_based", result.metadata["methods_tried"])
        
    def test_density_target_predict_without_structure(self):
        """Test DensityTarget predict method without structure"""
        element_densities = {"Fe": 7.87, "Ni": 8.91, "Co": 8.90}
        target = DensityTarget(element_densities=element_densities, requires_structure=True)
        composition = np.array([0.5, 0.3, 0.2])
        
        # Should fall back to linear method when structure is not available
        result = target.predict(composition)
        self.assertIn("linear", result.metadata["methods_tried"])
        self.assertEqual(result.metadata["method_used"], "linear")
        
    def test_density_target_predict_no_methods_available(self):
        """Test DensityTarget predict method with no available methods"""
        target = DensityTarget(requires_structure=True, preferred_methods=["structure_based"])  # No element_densities provided
        composition = np.array([0.5, 0.3, 0.2])
        
        # Should raise an error since no methods are available
        with self.assertRaises(ValueError):
            target.predict(composition)


if __name__ == '__main__':
    unittest.main()