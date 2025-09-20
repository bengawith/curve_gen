#!/usr/bin/env python3
"""
Command Line Interface for CurveGen - Aerodynamic Curve Generation Tool

This CLI provides access to all CurveGen functionality through command-line arguments.
Supports multiple input types: Kulfan parameters, .dat files, coordinates, and aerofoil names.

Examples:
    python cg.py --aero_name "naca2410"
    python cg.py --dat_path "./data/aerofoil_data/clarkx.dat" --save_plots
    python cg.py --coords "[[0,0],[0.5,0.1],[1,0]]" --output_dir "./results"
    python cg.py --k_params '{"lower_weights":[0.1,0.2],"upper_weights":[0.1,0.2],"TE_thickness":0.01,"leading_edge_weight":0.5}'
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curve_gen.curve_gen import CurveGen, CurveGenError


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate aerodynamic curves using CurveGen neural network.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --aero_name "naca2410"
  %(prog)s --dat_path "./data/aerofoil_data/clarkx.dat" --save_plots
  %(prog)s --coords "[[0,0],[0.5,0.1],[1,0]]" --output_dir "./results"
  %(prog)s --k_params '{"lower_weights":[0.1,0.2,0.3,0.4,0.5,0.6],"upper_weights":[0.1,0.2,0.3,0.4,0.5,0.6],"TE_thickness":0.01,"leading_edge_weight":0.5}'

Input Types (choose exactly one):
  --k_params     Kulfan parameters as JSON string
  --dat_path     Path to .dat file containing airfoil coordinates  
  --coords       Coordinates as JSON array [[x1,y1],[x2,y2],...]
  --aero_name    Name of airfoil from database

Output Options:
  --save_plots   Save plots as PNG files
  --save_data    Save prediction data as JSON file
  --output_dir   Directory for output files (default: current directory)
  --plot_only    Only generate plots, don't save data
  --data_only    Only save data, don't show plots
        """
    )
    
    # Input group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--k_params', 
        type=str, 
        help='Kulfan parameters as JSON string with keys: lower_weights, upper_weights, TE_thickness, leading_edge_weight'
    )
    input_group.add_argument(
        '--dat_path', 
        type=str, 
        help='Path to .dat file containing airfoil coordinates'
    )
    input_group.add_argument(
        '--coords', 
        type=str, 
        help='Coordinates as JSON array: [[x1,y1],[x2,y2],...]'
    )
    input_group.add_argument(
        '--aero_name', 
        type=str, 
        help='Name of airfoil (e.g., "naca2410", "ag19", "clarkx")'
    )
    
    # Output options
    parser.add_argument(
        '--save_plots', 
        action='store_true',
        help='Save plots as PNG files'
    )
    parser.add_argument(
        '--save_data', 
        action='store_true',
        help='Save prediction data as JSON file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='.',
        help='Directory for output files (default: current directory)'
    )
    parser.add_argument(
        '--plot_only', 
        action='store_true',
        help='Only generate plots, don\'t save data'
    )
    parser.add_argument(
        '--data_only', 
        action='store_true',
        help='Only save data, don\'t show plots'
    )
    
    # Advanced options
    parser.add_argument(
        '--cache_size', 
        type=int, 
        default=128,
        help='LRU cache size for predictions (default: 128)'
    )
    parser.add_argument(
        '--model_dir', 
        type=str,
        help='Custom directory containing model files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plot display (useful for batch processing)'
    )
    
    # Plot customization
    parser.add_argument(
        '--plot_style',
        type=str,
        default='default',
        choices=['default', 'bmh', 'seaborn', 'ggplot', 'classic'],
        help='Matplotlib style for plots'
    )
    parser.add_argument(
        '--curve_color',
        type=str,
        default='blue',
        help='Color for lift curve plot'
    )
    parser.add_argument(
        '--aero_color',
        type=str,
        default='red',
        help='Color for airfoil shape plot'
    )
    
    return parser.parse_args()


def validate_json_input(json_str: str, input_type: str) -> Any:
    """Validate and parse JSON input."""
    try:
        data = json.loads(json_str)
        
        if input_type == 'k_params':
            required_keys = ['lower_weights', 'upper_weights', 'TE_thickness', 'leading_edge_weight']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"Missing required keys in k_params: {missing_keys}")
                
        elif input_type == 'coords':
            if not isinstance(data, list) or not all(isinstance(point, list) and len(point) == 2 for point in data):
                raise ValueError("Coordinates must be a list of [x, y] pairs")
                
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for {input_type}: {e}")


def create_curvegen_instance(args: argparse.Namespace) -> CurveGen:
    """Create CurveGen instance based on command line arguments."""
    kwargs = {
        'cache_size': args.cache_size,
        'model_dir': args.model_dir
    }
    
    if args.k_params:
        k_params = validate_json_input(args.k_params, 'k_params')
        kwargs['k_params'] = k_params
        
    elif args.dat_path:
        if not os.path.exists(args.dat_path):
            raise FileNotFoundError(f"DAT file not found: {args.dat_path}")
        kwargs['dat_path'] = args.dat_path
        
    elif args.coords:
        coords = validate_json_input(args.coords, 'coords')
        kwargs['coords'] = coords
        
    elif args.aero_name:
        kwargs['aero_name'] = args.aero_name
    
    return CurveGen(**kwargs)


def setup_output_directory(output_dir: str) -> Path:
    """Create and return output directory path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def generate_output_filename(cg: CurveGen, file_type: str) -> str:
    """Generate appropriate filename based on input type."""
    if cg.aero_name:
        base_name = cg.aero_name
    elif cg.dat_path:
        base_name = Path(cg.dat_path).stem
    else:
        base_name = "curvegen_output"
    
    extensions = {
        'lift_curve': '_lift_curve.png',
        'aero_shape': '_aero_shape.png', 
        'data': '_data.json'
    }
    
    return f"{base_name}{extensions.get(file_type, '.out')}"


def display_analysis(cg: CurveGen, verbose: bool = False) -> None:
    """Display analysis results."""
    data = cg.get_data()
    predictions = data['predictions']
    
    print(f"\n{'='*60}")
    print(f"CURVEGEN ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Input type: ", end='')
    if cg.aero_name:
        print(f"Airfoil name '{cg.aero_name}'")
    elif cg.dat_path:
        print(f"DAT file '{cg.dat_path}'")
    elif cg.coords:
        print(f"Coordinates ({len(cg.coords)} points)")
    elif cg.k_params:
        print(f"Kulfan parameters")
    
    # Prediction analysis
    n = len(predictions) // 2
    cls = predictions[:n]
    alphas = predictions[n:]
    
    print(f"Predictions: {len(predictions)} values ({n} CL, {n} alpha)")
    print(f"CL range: {np.min(cls):.4f} to {np.max(cls):.4f}")
    print(f"Alpha range: {np.min(alphas):.2f}° to {np.max(alphas):.2f}°")
    
    # Key aerodynamic points
    max_cl_idx = np.argmax(cls)
    min_cl_idx = np.argmin(cls)
    zero_cl_idx = np.argmin(np.abs(cls))
    
    print(f"\nKey aerodynamic points:")
    print(f"  Maximum CL: {cls[max_cl_idx]:.4f} at α = {alphas[max_cl_idx]:.2f}°")
    print(f"  Minimum CL: {cls[min_cl_idx]:.4f} at α = {alphas[min_cl_idx]:.2f}°")
    print(f"  Zero CL: {cls[zero_cl_idx]:.4f} at α = {alphas[zero_cl_idx]:.2f}°")
    
    if verbose and cg.k_params:
        print(f"\nKulfan parameters:")
        for key, value in cg.k_params.items():
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"  {key}: [{', '.join(f'{v:.4f}' for v in value[:3])}...] (length: {len(value)})")
            else:
                print(f"  {key}: {value}")


def main():
    """Main CLI function."""
    args = None  # Initialize args variable
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up matplotlib style
        if args.plot_style != 'default':
            import matplotlib.pyplot as plt
            plt.style.use(args.plot_style)
        
        # Validate conflicting options
        if args.plot_only and args.data_only:
            raise ValueError("Cannot specify both --plot_only and --data_only")
        
        if args.verbose:
            print("CurveGen CLI - Starting analysis...")
            print(f"Cache size: {args.cache_size}")
            if args.model_dir:
                print(f"Model directory: {args.model_dir}")
        
        # Create CurveGen instance
        if args.verbose:
            print("Creating CurveGen instance...")
        cg = create_curvegen_instance(args)
        
        # Set up output directory
        output_path = setup_output_directory(args.output_dir)
        
        # Display analysis
        display_analysis(cg, args.verbose)
        
        # Generate plots
        if not args.data_only:
            print(f"\nGenerating plots...")
            
            # Plot lift curve
            if args.save_plots or args.plot_only:
                curve_filename = output_path / generate_output_filename(cg, 'lift_curve')
                if args.verbose:
                    print(f"  Saving lift curve to: {curve_filename}")
                cg.plot_curve(save_path=str(curve_filename) if args.save_plots else None,
                             color=args.curve_color, linewidth=2, alpha=0.8)
            else:
                if not args.no_plots:
                    cg.plot_curve(color=args.curve_color, linewidth=2, alpha=0.8)
            
            # Plot airfoil shape (if coordinates available)
            if cg.coords is not None:
                if args.save_plots or args.plot_only:
                    aero_filename = output_path / generate_output_filename(cg, 'aero_shape')
                    if args.verbose:
                        print(f"  Saving airfoil shape to: {aero_filename}")
                    cg.plot_aero(save_path=str(aero_filename) if args.save_plots else None,
                                color=args.aero_color, linewidth=2, alpha=0.8)
                else:
                    if not args.no_plots:
                        cg.plot_aero(color=args.aero_color, linewidth=2, alpha=0.8)
            elif args.verbose:
                print("  Skipping airfoil shape plot (no coordinates available)")
        
        # Save data
        if args.save_data or args.data_only or (not args.plot_only and not args.save_plots):
            data_filename = output_path / generate_output_filename(cg, 'data')
            if args.verbose:
                print(f"\nSaving data to: {data_filename}")
            cg.save_data(str(data_filename))
            print(f"Data saved to: {data_filename}")
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
    except CurveGenError as e:
        print(f"CurveGen Error: {e.message}", file=sys.stderr)
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"File Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except ValueError as e:
        print(f"Input Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        if args and hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()