# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018

@author: Timofey  Golubev

This just contains the function for reading photogeneration rate from a generation rate data file.
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19, 2018
Enhanced for 2D simulation

@author: Timofey Golubev, Extended for 2D

2D Photogeneration Module
Handles the spatial distribution of optical generation rates in 2D semiconductor devices.
Supports both uniform and spatially-dependent generation profiles based on optical modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline

def get_photogeneration_2d(params):
    '''
    Read or generate photogeneration rate distribution for 2D device simulation.
    
    The function supports multiple input formats:
    - Single value: uniform generation across the device
    - 1D profile: applied uniformly in the perpendicular direction  
    - 2D profile: full spatial distribution
    
    Parameters:
        params: Params object containing device geometry and generation parameters
        
    Returns:
        photogen_rate: 2D array of generation rates (m^-3 s^-1)
    '''
    
    try:
        gen_file = open(params.gen_rate_file_name, "r")
    except FileNotFoundError:
        print(f"Unable to open generation file {params.gen_rate_file_name}")
        print("Generating uniform photogeneration profile...")
        return _create_uniform_generation(params)
            
    try:
        # Read generation data and determine format
        gen_data = np.loadtxt(params.gen_rate_file_name)
        gen_file.close()
        
        # Determine data format and process accordingly
        if gen_data.ndim == 1:
            # 1D profile - extend to 2D
            photogen_rate = _extend_1d_to_2d(gen_data, params)
        elif gen_data.ndim == 2:
            # Check if it's a 2D spatial profile or coordinate-value pairs
            if gen_data.shape[1] == 3:
                # Format: x, y, generation_rate
                photogen_rate = _interpolate_scattered_data(gen_data, params)
            else:
                # Direct 2D array format
                photogen_rate = _process_2d_array(gen_data, params)
        else:
            raise ValueError("Unsupported generation data format")
            
    except Exception as e:
        print(f"Error processing generation file: {e}")
        print("Falling back to uniform generation profile...")
        photogen_rate = _create_uniform_generation(params)
    
    # Apply scaling factor
    photogen_rate = params.Photogen_scaling * photogen_rate / np.max(photogen_rate)
    
    return photogen_rate

def _create_uniform_generation(params):
    '''
    Create uniform photogeneration profile across the 2D device.
    
    Parameters:
        params: Params object with device geometry
        
    Returns:
        photogen_rate: Uniform 2D generation array
    '''
    n_total = params.num_cell_x * params.num_cell_y
    return np.ones(n_total)

def _extend_1d_to_2d(gen_1d, params):
    '''
    Extend 1D generation profile to 2D by replicating along one dimension.
    Assumes the 1D profile represents variation along the x-direction.
    
    Parameters:
        gen_1d: 1D generation profile
        params: Params object with device geometry
        
    Returns:
        photogen_rate: Extended 2D generation array
    '''
    # Interpolate 1D data to match grid resolution
    if len(gen_1d) != params.num_cell_x:
        x_orig = np.linspace(0, params.L_x, len(gen_1d))
        x_new = np.linspace(0, params.L_x, params.num_cell_x)
        gen_1d_interp = np.interp(x_new, x_orig, gen_1d)
    else:
        gen_1d_interp = gen_1d
    
    # Extend to 2D by replicating in y-direction
    photogen_2d = np.tile(gen_1d_interp, (params.num_cell_y, 1))
    
    return photogen_2d.flatten()

def _process_2d_array(gen_2d, params):
    '''
    Process direct 2D array input and interpolate to simulation grid.
    
    Parameters:
        gen_2d: 2D generation array
        params: Params object with device geometry
        
    Returns:
        photogen_rate: Interpolated 2D generation array
    '''
    # Check if dimensions match simulation grid
    if gen_2d.shape == (params.num_cell_y, params.num_cell_x):
        return gen_2d.flatten()
    
    # Interpolate to match simulation grid
    ny_orig, nx_orig = gen_2d.shape
    
    # Create coordinate arrays for interpolation
    x_orig = np.linspace(0, params.L_x, nx_orig)
    y_orig = np.linspace(0, params.L_y, ny_orig)
    x_new = np.linspace(0, params.L_x, params.num_cell_x)
    y_new = np.linspace(0, params.L_y, params.num_cell_y)
    
    # Perform 2D interpolation
    interpolator = RectBivariateSpline(y_orig, x_orig, gen_2d, kx=1, ky=1)
    gen_2d_interp = interpolator(y_new, x_new)
    
    return gen_2d_interp.flatten()

def _interpolate_scattered_data(gen_data, params):
    '''
    Interpolate scattered (x, y, value) data points to regular grid.
    
    Parameters:
        gen_data: Array with columns [x, y, generation_rate]
        params: Params object with device geometry
        
    Returns:
        photogen_rate: Interpolated 2D generation array on regular grid
    '''
    from scipy.spatial import griddata
    
    # Extract coordinates and values
    points = gen_data[:, 0:2]  # x, y coordinates
    values = gen_data[:, 2]    # generation rates
    
    # Create regular grid for interpolation
    x_grid = np.linspace(0, params.L_x, params.num_cell_x)
    y_grid = np.linspace(0, params.L_y, params.num_cell_y)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Interpolate using linear method with nearest neighbor extrapolation
    grid_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
    gen_interp = griddata(points, values, grid_points, 
                         method='linear', fill_value=np.mean(values))
    
    return gen_interp

def create_analytical_generation_profile(params, profile_type='gaussian'):
    '''
    Create analytical photogeneration profiles for testing and demonstration.
    
    Parameters:
        params: Params object with device geometry
        profile_type: Type of analytical profile ('gaussian', 'exponential', 'uniform')
        
    Returns:
        photogen_rate: Analytical 2D generation array
    '''
    # Create coordinate grids
    x = np.linspace(0, params.L_x, params.num_cell_x)
    y = np.linspace(0, params.L_y, params.num_cell_y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if profile_type == 'gaussian':
        # Gaussian beam profile centered in the device
        x_center = params.L_x / 2
        y_center = params.L_y / 2
        sigma_x = params.L_x / 6
        sigma_y = params.L_y / 6
        
        photogen_2d = np.exp(-((X - x_center)**2 / (2 * sigma_x**2) + 
                              (Y - y_center)**2 / (2 * sigma_y**2)))
                              
    elif profile_type == 'exponential':
        # Exponential decay from illuminated surface
        absorption_length = params.L_x / 5
        photogen_2d = np.exp(-X / absorption_length)
        
    elif profile_type == 'linear_gradient':
        # Linear gradient across the device
        photogen_2d = 1.0 - X / params.L_x
        
    elif profile_type == 'interference':
        # Interference pattern from optical modeling
        period_x = params.L_x / 8
        period_y = params.L_y / 6
        photogen_2d = (1 + 0.5 * np.cos(2 * np.pi * X / period_x) * 
                      np.cos(2 * np.pi * Y / period_y))
                      
    else:  # uniform
        photogen_2d = np.ones_like(X)
    
    return photogen_2d.flatten()

def visualize_generation_profile(photogen_rate, params, save_path=None):
    '''
    Create visualization of the 2D photogeneration profile.
    
    Parameters:
        photogen_rate: 1D array of generation rates (flattened 2D)
        params: Params object with device geometry
        save_path: Optional path to save the figure
    '''
    # Reshape to 2D for visualization
    photogen_2d = photogen_rate.reshape(params.num_cell_y, params.num_cell_x)
    
    # Create coordinate grids in micrometers for better readability
    x_um = np.linspace(0, params.L_x * 1e6, params.num_cell_x)
    y_um = np.linspace(0, params.L_y * 1e6, params.num_cell_y)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    # Main contour plot
    plt.subplot(2, 2, (1, 2))
    contour = plt.contourf(x_um, y_um, photogen_2d, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Generation Rate (normalized)')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')
    plt.title('2D Photogeneration Profile')
    plt.axis('equal')
    
    # X-direction profile at center
    plt.subplot(2, 2, 3)
    center_y = params.num_cell_y // 2
    plt.plot(x_um, photogen_2d[center_y, :], 'b-', linewidth=2)
    plt.xlabel('X Position (μm)')
    plt.ylabel('Generation Rate')
    plt.title('X-Direction Profile (Center)')
    plt.grid(True, alpha=0.3)
    
    # Y-direction profile at center
    plt.subplot(2, 2, 4)
    center_x = params.num_cell_x // 2
    plt.plot(y_um, photogen_2d[:, center_x], 'r-', linewidth=2)
    plt.xlabel('Y Position (μm)')
    plt.ylabel('Generation Rate')
    plt.title('Y-Direction Profile (Center)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Demonstration functionality
if __name__ == "__main__":
    # Create example parameters for demonstration
    class DemoParams:
        def __init__(self):
            self.L_x = 300e-9
            self.L_y = 300e-9
            self.num_cell_x = 61
            self.num_cell_y = 61
            self.Photogen_scaling = 1e24
            self.gen_rate_file_name = "demo_gen_rate.inp"
    
    params = DemoParams()
    
    print("2D Photogeneration Module - Demonstration")
    print("=" * 50)
    
    # Generate different analytical profiles
    profiles = ['uniform', 'gaussian', 'exponential', 'interference']
    
    for profile_type in profiles:
        print(f"\nGenerating {profile_type} profile...")
        photogen = create_analytical_generation_profile(params, profile_type)
        
        print(f"Profile statistics:")
        print(f"  Min: {np.min(photogen):.3e}")
        print(f"  Max: {np.max(photogen):.3e}")
        print(f"  Mean: {np.mean(photogen):.3e}")
        
        # Optionally visualize (uncomment the following line)
        # visualize_generation_profile(photogen, params, f"{profile_type}_profile.png")
    
    print("\n2D photogeneration module demonstration completed successfully.")
    
# import numpy as np

# def get_photogeneration(params):
#     '''
#     Reads photogeneration rate from an input file.
#     Inputs: 
#         params: Params object which contains several necessary parameters such as the name of generation
#                 rate file as well as the photogeneration scaling to use. The photogeneration scaling
#                 is determined by finding a value which will result in the correct short-circuit current.
#     '''
    
#     try:
#         gen_file = open(params.gen_rate_file_name, "r")
#     except:
#         print(f"Unable to open file{params.gen_rate_file_name}")
            
#     photogen_rate = np.loadtxt(params.gen_rate_file_name)  
#     #photogen_rate = np.ones(params.num_cell+1)
    
#     photogen_rate = params.Photogen_scaling * photogen_rate/np.max(photogen_rate)
    
#     gen_file.close()
    
#     return photogen_rate
            
  
            
        