import sys
sys.path.append('./')

import unittest
from src.curve_gen.cg import CurveGen
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from src.utils_ import read_coords

import matplotlib
matplotlib.use('Agg')

coords = read_coords('./data/aerofoil_data/naca2410.dat')
k_params = get_kulfan_parameters(coordinates=coords, n_weights_per_side=6)

class TestCurveGenKParams(unittest.TestCase):

    def setUp(self):
        self.cg_kulf = CurveGen(k_params=k_params)

    def test_plots(self):
        try:
            self.cg_kulf.plot_curve()
        except Exception as e:
            self.fail(f"cg_kulf.plot_curve() raised an exception: {e}")
    
    def test_data(self):
        data = self.cg_kulf.get_data()
        self.assertIsInstance(data, dict)
        self.assertIn('k_params', data)
        self.assertIn('predictions', data)
        self.assertIn('aero_name', data)


class TestCurveGenCoords(unittest.TestCase):

    def setUp(self):
        self.cg_coords = CurveGen(coords=coords)


    def test_coords(self):
        try:
            self.cg_coords.plot_curve()
        except Exception as e:
            self.fail(f"cg_coords.plot_curve() raised an exception: {e}")

        try:
            self.cg_coords.plot_aero()
        except Exception as e:
            self.fail(f"cg_coords.plot_aero() raised an exception: {e}")

    def test_data(self):
        data = self.cg_coords.get_data()
        self.assertIsInstance(data, dict)
        self.assertIn('k_params', data)
        self.assertIn('predictions', data)
        self.assertIn('aero_name', data)
        

class TestCurveGenAerofoilName(unittest.TestCase):

    def setUp(self):
        self.cg_aero = CurveGen(aero_name='naca2410')
    

    def test_aerofoil_name(self):
        try:
            self.cg_aero.plot_curve()
        except Exception as e:
            self.fail(f"cg_aero.plot_curve() raised an exception: {e}")
        
        try:
            self.cg_aero.plot_aero()
        except Exception as e:
            self.fail(f"cg_aero.plot_aero() raised an exception: {e}")

    def test_data(self):
        data = self.cg_aero.get_data()
        self.assertIsInstance(data, dict)
        self.assertIn('k_params', data)
        self.assertIn('predictions', data)
        self.assertIn('aero_name', data)


class TestCurveGenDatFile(unittest.TestCase):

    def setUp(self):
        self.cg_dat = CurveGen(dat_path='./data/aerofoil_data/naca2410.dat')
    

    def test_dat_file(self):
        try:
            self.cg_dat.plot_curve()
        except Exception as e:
            self.fail(f"cg_dat.plot_curve() raised an exception: {e}")
        
        try:
            self.cg_dat.plot_aero()
        except Exception as e:
            self.fail(f"cg_dat.plot_aero() raised an exception: {e}")

    
    def test_data(self):
        data = self.cg_dat.get_data()
        self.assertIsInstance(data, dict)
        self.assertIn('k_params', data)
        self.assertIn('predictions', data)
        self.assertIn('aero_name', data)


if __name__ == '__main__':
    unittest.main()