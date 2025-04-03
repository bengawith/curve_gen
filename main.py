'''
Example usage of CurveGen class for plotting predicted lift curve.
'''

from curve_gen.curve_gen import CurveGen
from src.utils_ import read_coords
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters

coords = read_coords('./data/aerofoil_data/naca2410.dat')
k_params = get_kulfan_parameters(coordinates=coords, n_weights_per_side=6)

if __name__ == '__main__':

    cg = CurveGen(k_params=k_params, dat_path=None, coords=None, aero_name=None)
    cg.plot_curve()
    
    cg = CurveGen(k_params=None, dat_path='./data/aerofoil_data/clarkx.dat', coords=None, aero_name=None)
    cg.plot_curve()

    cg = CurveGen(k_params=None, dat_path=None, coords=coords, aero_name=None)
    cg.plot_curve()

    cg = CurveGen(k_params=None, dat_path=None, coords=None, aero_name='ag19')
    cg.plot_curve()
