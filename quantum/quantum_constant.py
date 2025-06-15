# -*- coding: utf-8 -*-
"""
Created for quantum correction implementation

@author:Xiaoyan Xu
        (XIDIAN UNIVERSITY)
Quantum correction constants for Density Gradient model implementation.
Contains all physical constants and parameters needed for quantum potential calculation.
"""
# -*- coding: utf-8 -*-
"""
Quantum Constants for 2D Semiconductor Device Simulation

This module defines the quantum mechanical parameters used in the 2D quantum-corrected
drift-diffusion simulation. These constants determine the strength of quantum effects
in nanoscale semiconductor devices.

@author: Extended for 2D simulation
"""

import numpy as np

# Quantum correction parameters (in meters)
# These parameters control the magnitude of quantum mechanical effects
# in the drift-diffusion simulation through the Bohm potential formulation

# Quantum length scale for electrons
# Typical values range from 0.5 nm to 5 nm depending on device dimensions
# and material properties. Smaller values reduce quantum effects.
delta_n = 2.0e-9  # meters (2 nm)

# Quantum length scale for holes  
# Generally similar to delta_n but can be different due to effective mass differences
# between electrons and holes in the semiconductor material
delta_p = 2.0e-9  # meters (2 nm)

# Physical interpretation:
# - delta_n and delta_p represent the characteristic length scales over which
#   quantum mechanical effects become important
# - They are related to the de Broglie wavelength of carriers in the material
# - For silicon: typically 1-3 nm
# - For III-V semiconductors: can be larger due to smaller effective masses

# Temperature-dependent scaling (optional)
# Quantum effects generally become more pronounced at lower temperatures
# and in devices with smaller dimensions
T_ref = 300.0  # Reference temperature (K)

def get_temperature_scaled_deltas(T, delta_n_ref=delta_n, delta_p_ref=delta_p):
    """
    Calculate temperature-scaled quantum parameters
    
    Quantum effects typically scale as sqrt(T_ref/T) due to thermal de Broglie wavelength
    
    Parameters:
        T: Operating temperature (K)
        delta_n_ref: Reference quantum parameter for electrons at T_ref
        delta_p_ref: Reference quantum parameter for holes at T_ref
        
    Returns:
        tuple: (delta_n_scaled, delta_p_scaled)
    """
    scale_factor = np.sqrt(T_ref / T)
    return delta_n_ref * scale_factor, delta_p_ref * scale_factor

# Material-specific quantum parameters
# Different semiconductor materials have different effective masses
# and band structures, leading to different quantum length scales

MATERIAL_PARAMETERS = {
    'silicon': {
        'delta_n': 1.5e-9,  # nm
        'delta_p': 1.5e-9,  # nm
        'effective_mass_ratio_n': 0.26,  # relative to free electron mass
        'effective_mass_ratio_p': 0.39,  # relative to free electron mass
    },
    'gaas': {
        'delta_n': 3.0e-9,  # nm - larger due to smaller effective mass
        'delta_p': 2.0e-9,  # nm
        'effective_mass_ratio_n': 0.067,
        'effective_mass_ratio_p': 0.48,
    },
    'organic': {
        'delta_n': 2.0e-9,  # nm - typical for organic semiconductors
        'delta_p': 2.0e-9,  # nm
        'effective_mass_ratio_n': 1.0,  # Often approximated as free electron mass
        'effective_mass_ratio_p': 1.0,
    }
}

def get_material_quantum_parameters(material='organic'):
    """
    Get quantum parameters for specific semiconductor materials
    
    Parameters:
        material: Material type ('silicon', 'gaas', 'organic')
        
    Returns:
        dict: Material-specific quantum parameters
    """
    if material in MATERIAL_PARAMETERS:
        return MATERIAL_PARAMETERS[material]
    else:
        print(f"Warning: Material '{material}' not found. Using organic parameters.")
        return MATERIAL_PARAMETERS['organic']

# Device size scaling
# Quantum effects become more important in smaller devices
# This function provides guidance for parameter selection

def suggest_quantum_parameters(device_width, device_height, material='organic'):
    """
    Suggest quantum parameters based on device dimensions and material
    
    Parameters:
        device_width: Device width in meters
        device_height: Device height in meters  
        material: Semiconductor material type
        
    Returns:
        dict: Suggested quantum parameters with rationale
    """
    # Get base material parameters
    mat_params = get_material_quantum_parameters(material)
    
    # Characteristic device dimension
    char_length = min(device_width, device_height)
    
    # Scale quantum parameters based on device size
    # Quantum effects are more important when delta ~ device_dimension/10 to device_dimension/5
    suggested_delta = char_length / 10
    
    # Apply material-specific scaling
    delta_n_suggested = min(suggested_delta, mat_params['delta_n'])
    delta_p_suggested = min(suggested_delta, mat_params['delta_p'])
    
    # Ensure minimum reasonable values
    delta_n_suggested = max(delta_n_suggested, 0.5e-9)  # Minimum 0.5 nm
    delta_p_suggested = max(delta_p_suggested, 0.5e-9)  # Minimum 0.5 nm
    
    return {
        'delta_n': delta_n_suggested,
        'delta_p': delta_p_suggested,
        'device_char_length': char_length,
        'material': material,
        'quantum_importance': 'high' if char_length < 50e-9 else 'medium' if char_length < 200e-9 else 'low'
    }

# Numerical stability parameters
# These help prevent numerical issues in quantum correction calculations

MIN_CARRIER_DENSITY = 1e10  # m^-3 - minimum density to prevent division by zero
MAX_QUANTUM_POTENTIAL = 1.0  # V - maximum allowed quantum potential magnitude
QUANTUM_SMOOTHING_FACTOR = 1e-20  # Small value for numerical stability

# Convergence parameters for quantum-corrected simulations
QUANTUM_CONVERGENCE_FACTOR = 0.1  # Additional convergence criterion when quantum effects are strong
QUANTUM_MIXING_FACTOR = 0.5  # Mixing parameter for quantum potential updates

# Export the current values for easy import
__all__ = [
    'delta_n', 'delta_p', 'T_ref',
    'get_temperature_scaled_deltas',
    'get_material_quantum_parameters', 
    'suggest_quantum_parameters',
    'MATERIAL_PARAMETERS',
    'MIN_CARRIER_DENSITY',
    'MAX_QUANTUM_POTENTIAL',
    'QUANTUM_SMOOTHING_FACTOR'
]

# Module information
if __name__ == "__main__":
    print("2D Quantum Constants Module")
    print("=" * 40)
    print(f"Default electron quantum parameter: {delta_n*1e9:.1f} nm")
    print(f"Default hole quantum parameter: {delta_p*1e9:.1f} nm")
    print(f"Reference temperature: {T_ref:.1f} K")
    
    # Example usage
    device_w, device_h = 300e-9, 300e-9  # 300 nm device
    suggestions = suggest_quantum_parameters(device_w, device_h, 'organic')
    
    print(f"\nSuggested parameters for {device_w*1e9:.0f}×{device_h*1e9:.0f} nm organic device:")
    print(f"  delta_n: {suggestions['delta_n']*1e9:.1f} nm")
    print(f"  delta_p: {suggestions['delta_p']*1e9:.1f} nm")
    print(f"  Quantum importance: {suggestions['quantum_importance']}")
    
    # Material comparison
    print(f"\nMaterial comparison:")
    for material in MATERIAL_PARAMETERS:
        params = get_material_quantum_parameters(material)
        print(f"  {material:8}: δₙ={params['delta_n']*1e9:.1f}nm, δₚ={params['delta_p']*1e9:.1f}nm")
        
# import numpy as np
# import constants as const

# delta_n = 1e-15
# delta_p = 1e-15

# # Fundamental quantum constants
# hbar = 1.054571817e-34  # Reduced Planck's constant, J·s
# hbar_eV = 6.582119569e-16  # Reduced Planck's constant, eV·s

# # Default effective masses (in units of free electron mass m0)
# # These should be adjusted based on the specific semiconductor material
# m0 = 9.1093837015e-31  # Free electron mass, kg
# m_n_star = 0.26 * m0   # Electron effective mass (typical for GaAs), kg
# m_p_star = 0.38 * m0   # Hole effective mass (typical for GaAs), kg

# # Quantum correction scaling factors
# # These parameters r_n and r_p are material-dependent scaling factors
# # For most semiconductors, they are close to unity
# r_n = 1.0  # Electron quantum correction scaling factor
# r_p = 1.0  # Hole quantum correction scaling factor

# # Calculate quantum correction coefficients b_n and b_p
# # b_n = (ℏ²r_n)/(12*q*m_n*)
# # b_p = (ℏ²r_p)/(12*q*m_p*)

# def calculate_b_n():
#     """
#     Calculate electron quantum correction coefficient b_n
    
#     Formula: b_n = (ℏ²r_n)/(12*q*m_n*)
    
#     Returns:
#     --------
#     float : b_n in units of J·m²
#     """
#     b_n = (hbar**2 * r_n) / (12.0 * const.q * m_n_star)
#     return b_n

# def calculate_b_p():
#     """
#     Calculate hole quantum correction coefficient b_p
    
#     Formula: b_p = (ℏ²r_p)/(12*q*m_p*)
    
#     Returns:
#     --------
#     float : b_p in units of J·m²
#     """
#     b_p = (hbar**2 * r_p) / (12.0 * const.q * m_p_star)
#     return b_p

# # Pre-calculate the coefficients for efficiency
# b_n = calculate_b_n()
# b_p = calculate_b_p()

# # Convert to more convenient units for numerical computation
# # Since we work with thermal voltage Vt = kT/q, it's convenient to express
# # b_n and b_p in terms of Vt and length units
# b_n_normalized = b_n / (const.Vt * const.q)  # in units of m²
# b_p_normalized = b_p / (const.Vt * const.q)  # in units of m²

# print(f"Quantum correction coefficients calculated:")
# print(f"b_n = {b_n:.6e} J·m²")
# print(f"b_p = {b_p:.6e} J·m²")
# print(f"b_n_normalized = {b_n_normalized:.6e} m²")
# print(f"b_p_normalized = {b_p_normalized:.6e} m²")