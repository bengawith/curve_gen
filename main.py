"""
Comprehensive example usage of CurveGen class for aerodynamic curve generation.

This script demonstrates all the different ways to use the CurveGen class:
1. Using Kulfan parameters
2. Using .dat file paths
3. Using coordinate arrays
4. Using aerofoil names

It also shows the enhanced features like saving plots and data.
"""

import numpy as np
from curve_gen.curve_gen import CurveGen, CurveGenError
from src.utils_ import read_coords
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters


def example_kulfan_parameters():
    """Example 1: Using Kulfan parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Using Kulfan Parameters")
    print("="*60)
    
    # Read coordinates and extract Kulfan parameters
    coords = read_coords('./data/aerofoil_data/naca2410.dat')
    k_params = get_kulfan_parameters(coordinates=coords, n_weights_per_side=6)
    
    try:
        # Create CurveGen instance with Kulfan parameters
        cg = CurveGen(k_params=k_params)
        
        print(f"Successfully created CurveGen with Kulfan parameters")
        print(f"  - Model device: {cg.device}")
        print(f"  - Cache size: {cg._cache_size}")
        
        # Get prediction data
        data = cg.get_data()
        print(f"  - Prediction shape: {data['predictions'].shape}")
        print(f"  - Aerofoil name: {data['aero_name']}")
        
        # Plot the curve and aerofoil
        print("  - Generating lift curve plot...")
        cg.plot_curve(color='blue', linestyle='-', marker='o', markersize=3)
        
        print("  - Generating aerofoil shape plot...")
        cg.plot_aero(color='red', linewidth=3)
        
        # Save data to file
        cg.save_data('output_kulfan_params.json')
        print("  - Data saved to 'output_kulfan_params.json'")
        
    except CurveGenError as e:
        print(f"Error: {e}")


def example_dat_file():
    """Example 2: Using .dat file path"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Using .dat File Path")
    print("="*60)
    
    try:
        # Create CurveGen instance with .dat file path
        cg = CurveGen(dat_path='./data/aerofoil_data/clarkx.dat')
        
        print(f"Successfully created CurveGen with .dat file")
        print(f"  - Aerofoil name: {cg.aero_name}")
        
        # Get and display prediction data
        data = cg.get_data()
        print(f"  - Prediction shape: {data['predictions'].shape}")
        
        # Plot with custom styling and save
        print("  - Generating plots with custom styling...")
        cg.plot_curve(save_path='clarkx_lift_curve.png', 
                     color='green', linewidth=2, alpha=0.8)
        
        if cg.coords is not None:
            cg.plot_aero(save_path='clarkx_shape.png',
                        color='purple', linewidth=2)
        
        print("  - Plots saved as PNG files")
        
    except CurveGenError as e:
        print(f"Error: {e}")


def example_coordinates():
    """Example 3: Using coordinate arrays"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Using Coordinate Arrays")
    print("="*60)
    
    try:
        # Read coordinates from file
        coords = read_coords('./data/aerofoil_data/naca2410.dat')
        
        # Create CurveGen instance with coordinates
        cg = CurveGen(coords=coords.tolist())  # Convert to list for JSON serialization
        
        print(f"Successfully created CurveGen with coordinates")
        print(f"  - Number of coordinate points: {len(coords)}")
        
        # Display prediction statistics
        data = cg.get_data()
        predictions = data['predictions']
        n = len(predictions) // 2
        cls = predictions[:n]
        alphas = predictions[n:]
        
        print(f"  - CL range: {np.min(cls):.3f} to {np.max(cls):.3f}")
        print(f"  - Alpha range: {np.min(alphas):.3f}° to {np.max(alphas):.3f}°")
        
        # Plot with enhanced styling
        print("  - Generating enhanced plots...")
        cg.plot_curve(color='orange', linewidth=3, alpha=0.9, 
                     marker='s', markersize=2, markerfacecolor='red')
        
        cg.plot_aero(color='darkblue', linewidth=2, alpha=0.8)
        
    except CurveGenError as e:
        print(f"Error: {e}")


def example_aerofoil_name():
    """Example 4: Using aerofoil name"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Using Aerofoil Name")
    print("="*60)
    
    try:
        # Create CurveGen instance with aerofoil name
        cg = CurveGen(aero_name='ag19')
        
        print(f"Successfully created CurveGen with aerofoil name")
        print(f"  - Aerofoil: {cg.aero_name}")
        
        # Get comprehensive data
        data = cg.get_data()
        
        # Display detailed analysis
        predictions = data['predictions']
        n = len(predictions) // 2
        cls = predictions[:n]
        alphas = predictions[n:]
        
        # Find key aerodynamic points
        max_cl_idx = np.argmax(cls)
        zero_cl_idx = np.argmin(np.abs(cls))
        
        print(f"  - Maximum CL: {cls[max_cl_idx]:.3f} at α = {alphas[max_cl_idx]:.1f}°")
        print(f"  - Zero CL at α ≈ {alphas[zero_cl_idx]:.1f}°")
        
        # Plot with comprehensive styling
        print("  - Generating comprehensive analysis plots...")
        
        # Enhanced lift curve plot
        cg.plot_curve(color='crimson', linewidth=2.5, alpha=0.9,
                     marker='o', markersize=4, markerfacecolor='yellow',
                     markeredgecolor='black', markeredgewidth=0.5)
        
        # Enhanced aerofoil plot if coordinates available
        if cg.coords is not None:
            cg.plot_aero(color='navy', linewidth=3, alpha=0.8)
        
        # Save comprehensive data
        cg.save_data(f'{cg.aero_name}_analysis.json')
        print(f"  - Analysis saved to '{cg.aero_name}_analysis.json'")
        
    except CurveGenError as e:
        print(f"Error: {e}")


def example_caching_performance():
    """Example 5: Demonstrating caching performance"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Caching Performance Demonstration")
    print("="*60)
    
    try:
        import time
        
        # Create instance with custom cache size
        cg = CurveGen(aero_name='naca2410', cache_size=64)
        
        # First generation (cold cache)
        start_time = time.time()
        data1 = cg.get_data()
        first_time = time.time() - start_time
        
        # Second generation (warm cache) - simulate by creating new instance with same input
        start_time = time.time()
        data2 = cg.get_data()
        second_time = time.time() - start_time
        
        print(f"Caching performance test completed")
        print(f"  - First generation time: {first_time:.4f} seconds")
        print(f"  - Second generation time: {second_time:.4f} seconds")
        print(f"  - Speedup factor: {first_time/second_time if second_time > 0 else 0:.2f}x")
        print(f"  - Results identical: {np.array_equal(data1['predictions'], data2['predictions'])}")
        
    except CurveGenError as e:
        print(f"Error: {e}")


def example_error_handling():
    """Example 6: Error handling and validation"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Error Handling and Validation")
    print("="*60)
    
    # Test various error conditions
    test_cases = [
        ("No inputs", {}),
        ("Multiple inputs", {"k_params": {}, "aero_name": "test"}),
        ("Invalid file path", {"dat_path": "nonexistent_file.dat"}),
        ("Invalid aerofoil name", {"aero_name": "invalid_aero_name"}),
    ]
    
    for test_name, kwargs in test_cases:
        try:
            print(f"\n  Testing: {test_name}")
            cg = CurveGen(**kwargs)
            print(f"  Expected error but creation succeeded")
        except CurveGenError as e:
            print(f"  Correctly caught error: {e.message}")
        except Exception as e:
            print(f"  ⚠ Unexpected error type: {type(e).__name__}: {e}")


if __name__ == '__main__':
    print("CurveGen Class Comprehensive Example")
    print("This script demonstrates all features of the updated CurveGen class")
    
    # Run all examples
    example_kulfan_parameters()
    example_dat_file()
    example_coordinates()
    example_aerofoil_name()
    example_caching_performance()
    example_error_handling()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)
    print("Check the generated files:")
    print("  - output_kulfan_params.json")
    print("  - clarkx_lift_curve.png")
    print("  - clarkx_shape.png")
    print("  - ag19_analysis.json")
    print("\nFor more information, see the CurveGen class documentation.")
