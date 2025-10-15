import unittest
import numpy as np
from comp_dart.generators.template_filler import TemplateLatticeFiller
from comp_dart.generators.substitution_generator import SubstitutionGenerator


class TestTemplateLatticeFiller(unittest.TestCase):
    """Test TemplateLatticeFiller class"""
    
    def test_template_lattice_filler_initialization(self):
        """Test TemplateLatticeFiller initialization"""
        filler = TemplateLatticeFiller(template_path="/path/to/template.cif")
        
        self.assertEqual(filler.template_path, "/path/to/template.cif")
        
    def test_template_lattice_filler_default_initialization(self):
        """Test TemplateLatticeFiller default initialization"""
        filler = TemplateLatticeFiller()
        
        self.assertIsNone(filler.template_path)
        
    def test_template_lattice_filler_generate(self):
        """Test TemplateLatticeFiller generate method"""
        filler = TemplateLatticeFiller(template_path="/path/to/template.cif")
        composition = np.array([0.5, 0.3, 0.2])
        elements = ["Fe", "Ni", "Co"]
        
        structures = filler.generate(composition, elements)
        
        # Check that we get a list of structures
        self.assertIsInstance(structures, list)
        self.assertGreater(len(structures), 0)
        # Check that each structure is a dict (placeholder)
        for structure in structures:
            self.assertIsInstance(structure, dict)
            self.assertIn("composition", structure)
            self.assertIn("elements", structure)
            self.assertIn("seed", structure)
            self.assertIn("template_path", structure)


class TestSubstitutionGenerator(unittest.TestCase):
    """Test SubstitutionGenerator class"""
    
    def test_substitution_generator_initialization(self):
        """Test SubstitutionGenerator initialization"""
        generator = SubstitutionGenerator(
            host_structure="host_structure",
            substitution_sites=[1, 2, 3]
        )
        
        self.assertEqual(generator.host_structure, "host_structure")
        self.assertEqual(generator.substitution_sites, [1, 2, 3])
        
    def test_substitution_generator_default_initialization(self):
        """Test SubstitutionGenerator default initialization"""
        generator = SubstitutionGenerator()
        
        self.assertIsNone(generator.host_structure)
        self.assertEqual(generator.substitution_sites, [])
        
    def test_substitution_generator_generate(self):
        """Test SubstitutionGenerator generate method"""
        generator = SubstitutionGenerator(
            host_structure="host_structure",
            substitution_sites=[1, 2, 3]
        )
        composition = np.array([0.5, 0.3, 0.2])
        elements = ["Fe", "Ni", "Co"]
        
        structures = generator.generate(composition, elements)
        
        # Check that we get a list of structures
        self.assertIsInstance(structures, list)
        self.assertEqual(len(structures), 3)  # We generate 3 structures
        # Check that each structure is a dict (placeholder)
        for structure in structures:
            self.assertIsInstance(structure, dict)
            self.assertIn("composition", structure)
            self.assertIn("elements", structure)
            self.assertIn("seed", structure)
            self.assertIn("type", structure)
            self.assertIn("substitution_sites", structure)
            self.assertEqual(structure["type"], "substituted")


if __name__ == '__main__':
    unittest.main()