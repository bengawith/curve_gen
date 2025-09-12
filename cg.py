from curve_gen.curve_gen import CurveGen
import argparse
import sys, os, json
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Generate curves using CurveGen.")
    parser.add_argument('--k_params', type=str, help='Kulfan parameters for the curve generation.')
    parser.add_argument('--dat_path', type=str, help='Path to the .dat file containing coordinates.')
    parser.add_argument('--coords', type=str, help='Coordinates for the aerofoil.')
    parser.add_argument('--aero_name', type=str, help='Name of the aerofoil for the curve generation.')
    if len(vars(parser.parse_args())) < 2:
        parser.error('Please specify one of the following: --k_params (Kulfan Parameters), --dat_path (Path to .dat file), --coords (Coordinates as a string), or --aero_name (Name of the aerofoil).')
    args = parser.parse_args()
    
    if args.k_params:
        curve_gen = CurveGen(k_params=json.loads(args.k_params))
    elif args.dat_path:
        curve_gen = CurveGen(dat_path=args.dat_path)
    elif args.coords:
        curve_gen = CurveGen(coords=json.loads(args.coords))
    elif args.aero_name:
        curve_gen = CurveGen(aero_name=args.aero_name)
    else:
        raise ValueError("No valid input provided. Please specify one of the required arguments.")
    
    curve_gen.plot_curve()
    curve_gen.plot_aero()

if __name__ == "__main__":
    # Example usage:
    # python cg.py --k_params '{"param1": value1, "param2": value2}'
    # python cg.py --dat_path '<path_to_dat_file>'
    # python cg.py --coords '[[x_1, y_1], [x_2, y_2]]'
    # python cg.py --aero_name 'drgnfly'
    main()