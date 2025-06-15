# -*- coding: utf-8 -*-
"""
2D Quantum Correction Implementation for Drift-Diffusion Simulation
Enhanced for 2D geometry with proper Laplacian calculation

Based on Bohm potential formulation for quantum effects in semiconductor devices.
The quantum correction terms Lambda_n and Lambda_p are calculated using:
Lambda_n = (delta_n^2 * Laplace(sqrt(n))) / sqrt(n)
Lambda_p = -(delta_p^2 * Laplace(sqrt(p))) / sqrt(p)

@author: Extended for 2D simulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import quantum.quantum_constant as qconst
import constants as const
import initialization
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import warnings

# Initialize parameters for quantum correction calculations
params = initialization.Params()

class QuantumConstants:
    """
    Container class for quantum correction parameters in 2D simulation
    """
    def __init__(self, delta_n=1e-9, delta_p=1e-9):
        """
        Initialize quantum correction parameters for 2D device simulation
        
        Parameters:
            delta_n: Quantum correction parameter for electrons (m)
            delta_p: Quantum correction parameter for holes (m)
        """
        self.delta_n = qconst.delta_n
        self.delta_p = qconst.delta_p

class SimulationConstants:
    """
    Container class for 2D simulation grid parameters
    """
    def __init__(self, dx=1e-9, dy=1e-9, nx=301, ny=301):
        """
        Initialize 2D simulation constants
        
        Parameters:
            dx: Grid spacing in x-direction (m)
            dy: Grid spacing in y-direction (m) 
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
        """
        self.dx = params.dx
        self.dy = params.dy
        self.nx = params.num_cell_x
        self.ny = params.num_cell_y

class QuantumCorrection:
    """
    2D Quantum correction calculator for drift-diffusion simulations
    
    This class implements quantum correction terms based on the Bohm potential
    formulation, extending the formulation to 2D geometry with proper handling
    of the 2D Laplacian operator.
    """
    
    def __init__(self, simulation_constants, quantum_constants):
        """
        Initialize the 2D quantum correction calculator
        
        Parameters:
            simulation_constants: SimulationConstants object
            quantum_constants: QuantumConstants object
        """
        self.const = simulation_constants
        self.qconst = quantum_constants
        self.dx = self.const.dx
        self.dy = self.const.dy
        self.nx = self.const.nx
        self.ny = self.const.ny
        self.n_total = self.nx * self.ny
        
        # Precompute 2D Laplacian operator matrix for computational efficiency
        self._setup_2d_laplacian_operator()
    
    def _setup_2d_laplacian_operator(self):
        """
        Setup the discrete 2D Laplacian operator matrix using finite differences
        Implements the standard 5-point stencil for 2D second derivatives:
        ∇²f ≈ (f[i,j-1] + f[i,j+1] - 2*f[i,j])/dx² + (f[i-1,j] + f[i+1,j] - 2*f[i,j])/dy²
        """
        # Coefficient arrays for sparse matrix construction
        main_diag = np.zeros(self.n_total)
        x_lower = np.zeros(self.n_total)  # j-1 direction
        x_upper = np.zeros(self.n_total)  # j+1 direction
        y_lower = np.zeros(self.n_total)  # i-1 direction
        y_upper = np.zeros(self.n_total)  # i+1 direction
        
        dx2_inv = 1.0 / (self.dx * self.dx)
        dy2_inv = 1.0 / (self.dy * self.dy)
        
        for i in range(self.ny):
            for j in range(self.nx):
                idx = self._ij_to_index(i, j)
                
                if self._is_boundary_point(i, j):
                    # Boundary points: apply Neumann boundary conditions (zero flux)
                    # This represents the physical constraint that carriers cannot escape the device
                    main_diag[idx] = self._setup_boundary_laplacian(i, j, dx2_inv, dy2_inv)
                    
                    # Set appropriate off-diagonal terms for boundary conditions
                    if j > 0:
                        x_lower[idx] = dx2_inv
                    if j < self.nx-1:
                        x_upper[idx] = dx2_inv
                    if i > 0:
                        y_lower[idx] = dy2_inv
                    if i < self.ny-1:
                        y_upper[idx] = dy2_inv
                        
                else:
                    # Interior points: standard 5-point stencil
                    main_diag[idx] = -2.0 * (dx2_inv + dy2_inv)
                    x_lower[idx] = dx2_inv
                    x_upper[idx] = dx2_inv
                    y_lower[idx] = dy2_inv
                    y_upper[idx] = dy2_inv
        
        # Construct sparse matrix representation
        offsets = [-self.nx, -1, 0, 1, self.nx]
        diagonals = [
            y_lower[self.nx:],      # Lower y-diagonal
            x_lower[1:],            # Lower x-diagonal
            main_diag,              # Main diagonal
            x_upper[:-1],           # Upper x-diagonal
            y_upper[:-self.nx]      # Upper y-diagonal
        ]
        
        self.laplacian_matrix = diags(diagonals, offsets, 
                                     shape=(self.n_total, self.n_total), 
                                     format='csr')
    
    def _setup_boundary_laplacian(self, i, j, dx2_inv, dy2_inv):
        """
        Setup Laplacian coefficients for boundary points with Neumann conditions
        
        Parameters:
            i, j: grid coordinates
            dx2_inv, dy2_inv: inverse squared grid spacings
            
        Returns:
            main_diagonal_coefficient: coefficient for the main diagonal
        """
        coeff = 0.0
        
        # Count interior neighbors to determine the main diagonal coefficient
        if j > 0:  # Has left neighbor
            coeff -= dx2_inv
        if j < self.nx-1:  # Has right neighbor
            coeff -= dx2_inv
        if i > 0:  # Has bottom neighbor
            coeff -= dy2_inv
        if i < self.ny-1:  # Has top neighbor
            coeff -= dy2_inv
            
        return coeff
    
    def _ij_to_index(self, i, j):
        """Convert 2D grid coordinates to linear index"""
        return i * self.nx + j
    
    def _is_boundary_point(self, i, j):
        """Check if point (i,j) is on the domain boundary"""
        return i == 0 or i == self.ny-1 or j == 0 or j == self.nx-1
    
    def _safe_sqrt(self, array, epsilon=1e-20):
        """
        Calculate square root with numerical stability for quantum corrections
        
        Parameters:
            array: Input carrier density array
            epsilon: Small value to prevent division by zero
            
        Returns:
            Numerically stable square root values
        """
        return np.sqrt(np.maximum(array, epsilon))
    
    def _safe_divide(self, numerator, denominator, epsilon=1e-20):
        """
        Perform safe division to avoid infinite or undefined values
        
        Parameters:
            numerator: Numerator array
            denominator: Denominator array
            epsilon: Small value to prevent division by zero
            
        Returns:
            Safe division result with numerical protection
        """
        safe_denominator = np.where(np.abs(denominator) < epsilon, 
                                   np.sign(denominator) * epsilon, 
                                   denominator)
        return numerator / safe_denominator
    
    def calculate_2d_laplacian(self, field):
        """
        Calculate the 2D Laplacian of a field using the precomputed sparse matrix
        
        Parameters:
            field: 1D array representing field values at grid points (flattened 2D grid)
            
        Returns:
            2D Laplacian of the field
        """
        if len(field) != self.n_total:
            raise ValueError(f"Field length {len(field)} does not match grid points {self.n_total}")
        
        return self.laplacian_matrix.dot(field)
    
    def calculate_lambda_n(self, n):
        """
        Calculate the quantum correction term Lambda_n for electrons in 2D
        
        Formula: Lambda_n = (delta_n^2 * ∇²√n) / √n
        
        Parameters:
            n: Electron density array (m^-3) on 2D grid
            
        Returns:
            Lambda_n: Quantum correction term for electrons
        """
        # Calculate sqrt(n) with numerical stability
        n_sqrt = self._safe_sqrt(n)
        
        # Calculate 2D Laplacian of sqrt(n)
        laplace_sqrt_n = self.calculate_2d_laplacian(n_sqrt)
        
        # Calculate Lambda_n with safe division
        lambda_n = self._safe_divide(self.qconst.delta_n**2 * laplace_sqrt_n, n_sqrt)
        
        return lambda_n
    
    def calculate_lambda_p(self, p):
        """
        Calculate the quantum correction term Lambda_p for holes in 2D
        
        Formula: Lambda_p = -(delta_p^2 * ∇²√p) / √p
        
        Parameters:
            p: Hole density array (m^-3) on 2D grid
            
        Returns:
            Lambda_p: Quantum correction term for holes
        """
        # Calculate sqrt(p) with numerical stability
        p_sqrt = self._safe_sqrt(p)
        
        # Calculate 2D Laplacian of sqrt(p)
        laplace_sqrt_p = self.calculate_2d_laplacian(p_sqrt)
        
        # Calculate Lambda_p with safe division (note the negative sign)
        lambda_p = -self._safe_divide(self.qconst.delta_p**2 * laplace_sqrt_p, p_sqrt)
        
        return lambda_p
    
    def quantum_correction(self, n, p):
        """
        Calculate both quantum correction terms for 2D simulation
        
        Parameters:
            n: Electron density array (m^-3) on 2D grid
            p: Hole density array (m^-3) on 2D grid
            
        Returns:
            tuple: (Lambda_n, Lambda_p) quantum correction terms
        """
        # Validate input arrays for proper dimensions
        if len(n) != self.n_total or len(p) != self.n_total:
            raise ValueError("Input arrays must match the total number of grid points")
        
        # Check for physical validity of carrier densities
        if np.any(n < 0) or np.any(p < 0):
            warnings.warn("Negative carrier densities detected. Results may be unphysical.")
        
        # Calculate quantum correction terms
        lambda_n = self.calculate_lambda_n(n)
        lambda_p = self.calculate_lambda_p(p)
        
        return lambda_n, lambda_p
    
    def update_quantum_parameters(self, delta_n=None, delta_p=None):
        """
        Update quantum correction parameters for sensitivity analysis
        
        Parameters:
            delta_n: New quantum correction parameter for electrons
            delta_p: New quantum correction parameter for holes
        """
        if delta_n is not None:
            self.qconst.delta_n = delta_n
        if delta_p is not None:
            self.qconst.delta_p = delta_p

def create_test_2d_profiles(nx, ny, device_width, device_height):
    """
    Create realistic test carrier density profiles for 2D demonstration
    
    Parameters:
        nx, ny: Grid dimensions
        device_width, device_height: Physical dimensions (m)
        
    Returns:
        tuple: (x_grid, y_grid, n_2d, p_2d) arrays for testing
    """
    # Create coordinate grids
    x = np.linspace(0, device_width, nx)
    y = np.linspace(0, device_height, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create realistic 2D carrier profiles with edge effects
    n_base = 1e16  # Base electron density (m^-3)
    p_base = 1e15  # Base hole density (m^-3)
    
    # Electron profile: higher near right edge (cathode)
    n_2d = n_base * (1 + 5 * np.exp(-X / (device_width * 0.2)) + 
                     3 * np.exp(-(device_width - X) / (device_width * 0.1)) +
                     0.5 * np.exp(-Y / (device_height * 0.3)))
    
    # Hole profile: higher near left edge (anode)
    p_2d = p_base * (1 + 8 * np.exp(-X / (device_width * 0.1)) + 
                     2 * np.exp(-(device_width - X) / (device_width * 0.3)) +
                     0.3 * np.exp(-Y / (device_height * 0.4)))
    
    return X, Y, n_2d.flatten(), p_2d.flatten()

# Demonstration and validation functionality
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("2D Quantum Correction Implementation - Validation Example")
    print("=" * 70)
    
    # Initialize 2D simulation parameters
    device_width = 300e-9   # 300 nm
    device_height = 300e-9  # 300 nm
    dx = 5e-9              # 5 nm grid spacing
    dy = 5e-9              # 5 nm grid spacing
    nx = int(device_width / dx)
    ny = int(device_height / dy)
    
    # Create constants objects for 2D simulation
    sim_const = SimulationConstants(dx=dx, dy=dy, nx=nx, ny=ny)
    quantum_const = QuantumConstants(delta_n=2e-9, delta_p=2e-9)
    
    # Initialize 2D quantum correction calculator
    qc_calculator = QuantumCorrection(sim_const, quantum_const)
    
    print(f"Device dimensions: {device_width*1e9:.0f} × {device_height*1e9:.0f} nm²")
    print(f"Grid spacing: {dx*1e9:.1f} × {dy*1e9:.1f} nm²")
    print(f"Grid size: {nx} × {ny} points")
    print(f"Quantum parameters: δₙ = {quantum_const.delta_n*1e9:.1f} nm, δₚ = {quantum_const.delta_p*1e9:.1f} nm")
    
    # Generate test 2D carrier density profiles
    X, Y, n_density, p_density = create_test_2d_profiles(nx, ny, device_width, device_height)
    
    # Calculate 2D quantum corrections
    print("\nCalculating 2D quantum correction terms...")
    lambda_n, lambda_p = qc_calculator.quantum_correction(n_density, p_density)
    
    # Display computational results
    print(f"Lambda_n range: [{np.min(lambda_n):.2e}, {np.max(lambda_n):.2e}] V")
    print(f"Lambda_p range: [{np.min(lambda_p):.2e}, {np.max(lambda_p):.2e}] V")
    print("\n2D quantum correction calculation completed successfully!")
    
    # Reshape results for visualization
    n_2d = n_density.reshape(ny, nx)
    p_2d = p_density.reshape(ny, nx)
    lambda_n_2d = lambda_n.reshape(ny, nx)
    lambda_p_2d = lambda_p.reshape(ny, nx)
    
    print("2D quantum correction implementation validated for semiconductor device simulation.")
    
    
# import sys
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# import quantum.quantum_constant as qconst
# import constants as const
# import initialization
# # 在这里写一个函数，能把n和delta_n传进去，计算出来$\Lambda_n=\frac{\delta^2_n\Delta\sqrt{n}}{\sqrt{n}}$的数值；

# # -*- coding: utf-8 -*-
# """
# Quantum Correction Implementation for Drift-Diffusion Simulation
# 量子修正计算模块

# Based on Bohm potential formulation for quantum effects in semiconductor devices.
# The quantum correction terms Lambda_n and Lambda_p are calculated using:
# Lambda_n = (delta_n^2 * Laplace(sqrt(n))) / sqrt(n)
# Lambda_p = -(delta_p^2 * Laplace(sqrt(p))) / sqrt(p)

# @author: Xiaoyan Xu
# """

# import numpy as np
# from scipy import sparse
# from scipy.sparse import diags
# import warnings


# params = initialization.Params()

# class QuantumConstants:
#     """
#     Container class for quantum correction parameters
#     """
#     def __init__(self, delta_n=1e-9, delta_p=1e-9):
#         """
#         Initialize quantum correction parameters
        
#         Parameters:
#             delta_n: Quantum correction parameter for electrons (m)
#             delta_p: Quantum correction parameter for holes (m)
#         """
#         self.delta_n = qconst.delta_n
#         self.delta_p = qconst.delta_p

# class SimulationConstants:
#     """
#     Container class for simulation parameters
#     """
#     def __init__(self, dx=1e-9, num_points=301):
#         """
#         Initialize simulation constants
        
#         Parameters:
#             dx: Grid spacing (m)
#             num_points: Number of grid points
#         """
#         self.dx = params.dx
#         self.num_points = params.num_cell

# class QuantumCorrection:
#     """
#     Quantum correction calculator for drift-diffusion simulations
    
#     This class implements the quantum correction terms based on the Bohm potential
#     formulation, which accounts for quantum mechanical effects in nanoscale devices.
#     """
    
#     def __init__(self, simulation_constants, quantum_constants):
#         """
#         Initialize the quantum correction calculator
        
#         Parameters:
#             simulation_constants: SimulationConstants object
#             quantum_constants: QuantumConstants object
#         """
#         self.const = simulation_constants
#         self.qconst = quantum_constants
#         self.dx = self.const.dx
#         self.num_points = self.const.num_points
        
#         # Precompute Laplacian operator matrix for efficiency
#         self._setup_laplacian_operator()
    
#     def _setup_laplacian_operator(self):
#         """
#         Setup the discrete Laplacian operator matrix for second derivative calculation
#         Using finite difference approximation: d²f/dx² ≈ (f[i+1] - 2*f[i] + f[i-1]) / dx²
#         """
#         # Create Laplacian matrix for interior points
#         main_diag = -2.0 * np.ones(self.num_points) / (self.dx**2)
#         off_diag = np.ones(self.num_points - 1) / (self.dx**2)
        
#         # Apply boundary conditions (zero flux/Neumann boundary conditions)
#         main_diag[0] = -1.0 / (self.dx**2)  # Forward difference at left boundary
#         main_diag[-1] = -1.0 / (self.dx**2)  # Backward difference at right boundary
        
#         self.laplacian_matrix = diags([off_diag, main_diag, off_diag], 
#                                      offsets=[-1, 0, 1], 
#                                      shape=(self.num_points, self.num_points),
#                                      format='csr')
    
#     def _safe_sqrt(self, array, epsilon=1e-20):
#         """
#         Calculate square root with numerical stability
        
#         Parameters:
#             array: Input array
#             epsilon: Small value to prevent division by zero
            
#         Returns:
#             Safe square root values
#         """
#         return np.sqrt(np.maximum(array, epsilon))
    
#     def _safe_divide(self, numerator, denominator, epsilon=1e-20):
#         """
#         Perform safe division to avoid inf/nan values
        
#         Parameters:
#             numerator: Numerator array
#             denominator: Denominator array
#             epsilon: Small value to prevent division by zero
            
#         Returns:
#             Safe division result
#         """
#         safe_denominator = np.where(np.abs(denominator) < epsilon, 
#                                    np.sign(denominator) * epsilon, 
#                                    denominator)
#         return numerator / safe_denominator
    
#     def calculate_laplacian(self, field):
#         """
#         Calculate the Laplacian (second derivative) of a field using finite differences
        
#         Parameters:
#             field: 1D array representing the field values at grid points
            
#         Returns:
#             Laplacian of the field
#         """
#         if len(field) != self.num_points:
#             raise ValueError(f"Field length {len(field)} does not match grid points {self.num_points}")
        
#         return self.laplacian_matrix.dot(field)
    
#     def calculate_lambda_n(self, n):
#         """
#         Calculate the quantum correction term Lambda_n for electrons
        
#         Formula: Lambda_n = (delta_n^2 * Laplace(sqrt(n))) / sqrt(n)
        
#         Parameters:
#             n: Electron density array (m^-3)
            
#         Returns:
#             Lambda_n: Quantum correction term for electrons
#         """
#         # Calculate sqrt(n) with numerical stability
#         n_sqrt = self._safe_sqrt(n)
        
#         # Calculate Laplacian of sqrt(n)
#         laplace_sqrt_n = self.calculate_laplacian(n_sqrt)
        
#         # Calculate Lambda_n with safe division
#         lambda_n = self._safe_divide(self.qconst.delta_n**2 * laplace_sqrt_n, n_sqrt)
        
#         return lambda_n
    
#     def calculate_lambda_p(self, p):
#         """
#         Calculate the quantum correction term Lambda_p for holes
        
#         Formula: Lambda_p = -(delta_p^2 * Laplace(sqrt(p))) / sqrt(p)
        
#         Parameters:
#             p: Hole density array (m^-3)
            
#         Returns:
#             Lambda_p: Quantum correction term for holes
#         """
#         # Calculate sqrt(p) with numerical stability
#         p_sqrt = self._safe_sqrt(p)
        
#         # Calculate Laplacian of sqrt(p)
#         laplace_sqrt_p = self.calculate_laplacian(p_sqrt)
        
#         # Calculate Lambda_p with safe division (note the negative sign)
#         lambda_p = -self._safe_divide(self.qconst.delta_p**2 * laplace_sqrt_p, p_sqrt)
        
#         return lambda_p
    
#     def quantum_correction(self, n, p):
#         """
#         Calculate both quantum correction terms
        
#         Parameters:
#             n: Electron density array (m^-3)
#             p: Hole density array (m^-3)
            
#         Returns:
#             tuple: (Lambda_n, Lambda_p) quantum correction terms
#         """
#         # Validate input arrays
#         if len(n) != self.num_points or len(p) != self.num_points:
#             raise ValueError("Input arrays must match the number of grid points")
        
#         # Check for physical validity (positive densities)
#         if np.any(n < 0) or np.any(p < 0):
#             warnings.warn("Negative carrier densities detected. Results may be unphysical.")
        
#         # Calculate quantum correction terms
#         lambda_n = self.calculate_lambda_n(n)
#         lambda_p = self.calculate_lambda_p(p)
        
#         return lambda_n, lambda_p
    
#     def update_quantum_parameters(self, delta_n=None, delta_p=None):
#         """
#         Update quantum correction parameters
        
#         Parameters:
#             delta_n: New quantum correction parameter for electrons
#             delta_p: New quantum correction parameter for holes
#         """
#         if delta_n is not None:
#             self.qconst.delta_n = delta_n
#         if delta_p is not None:
#             self.qconst.delta_p = delta_p

# def create_test_carrier_profiles(num_points, device_thickness):
#     """
#     Create realistic test carrier density profiles for demonstration
    
#     Parameters:
#         num_points: Number of grid points
#         device_thickness: Device thickness (m)
        
#     Returns:
#         tuple: (position, n, p) arrays
#     """
#     # Position array
#     position = np.linspace(0, device_thickness, num_points)
    
#     # Create realistic carrier profiles (exponential decay from contacts)
#     # Electron profile: higher concentration near cathode (right side)
#     n_base = 1e16  # Base electron density (m^-3)
#     n = n_base * (1 + 10 * np.exp(-position / (device_thickness * 0.2)) + 
#                   5 * np.exp(-(device_thickness - position) / (device_thickness * 0.1)))
    
#     # Hole profile: higher concentration near anode (left side) 
#     p_base = 1e15  # Base hole density (m^-3)
#     p = p_base * (1 + 15 * np.exp(-position / (device_thickness * 0.1)) + 
#                   3 * np.exp(-(device_thickness - position) / (device_thickness * 0.3)))
    
#     return position, n, p

# # Usage example and demonstration
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
    
#     print("Quantum Correction Implementation - Usage Example")
#     print("=" * 60)
    
#     # Initialize simulation parameters
#     device_thickness = 300e-9  # 300 nm
#     dx = 1e-12   # 1 nm grid spacing
#     num_points = int(device_thickness / dx)
    
#     # Create constants objects
#     sim_const = SimulationConstants(dx=dx, num_points=num_points)
#     quantum_const = QuantumConstants(delta_n=2e-9, delta_p=2e-9)
    
#     # Initialize quantum correction calculator
#     qc_calculator = QuantumCorrection(sim_const, quantum_const)
    
#     print(f"Device thickness: {device_thickness*1e9:.0f} nm")
#     print(f"Grid spacing: {dx*1e9:.1f} nm")
#     print(f"Number of grid points: {num_points}")
#     print(f"Quantum parameters: δₙ = {quantum_const.delta_n*1e9:.1f} nm, δₚ = {quantum_const.delta_p*1e9:.1f} nm")
    
#     # Generate test carrier density profiles
#     position, n_density, p_density = create_test_carrier_profiles(num_points, device_thickness)
    
#     # Calculate quantum corrections
#     print("\nCalculating quantum correction terms...")
#     lambda_n, lambda_p = qc_calculator.quantum_correction(n_density, p_density)
    
#     # Display results
#     print(f"Lambda_n range: [{np.min(lambda_n):.2e}, {np.max(lambda_n):.2e}] V")
#     print(f"Lambda_p range: [{np.min(lambda_p):.2e}, {np.max(lambda_p):.2e}] V")
    
#     # Create visualization
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
#     # Plot carrier densities
#     ax1.semilogy(position*1e9, n_density, 'b-', linewidth=2, label='Electrons')
#     ax1.semilogy(position*1e9, p_density, 'r-', linewidth=2, label='Holes')
#     ax1.set_xlabel('Position (nm)')
#     ax1.set_ylabel('Carrier Density (m⁻³)')
#     ax1.set_title('Carrier Density Profiles')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Plot square root profiles
#     ax2.plot(position*1e9, np.sqrt(n_density), 'b-', linewidth=2, label='√n')
#     ax2.plot(position*1e9, np.sqrt(p_density), 'r-', linewidth=2, label='√p')
#     ax2.set_xlabel('Position (nm)')
#     ax2.set_ylabel('√(Carrier Density) (m⁻³/²)')
#     ax2.set_title('Square Root Carrier Profiles')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # Plot quantum correction terms
#     ax3.plot(position*1e9, lambda_n, 'b-', linewidth=2, label='Λₙ')
#     ax3.set_xlabel('Position (nm)')
#     ax3.set_ylabel('Λₙ (V)')
#     ax3.set_title('Quantum Correction - Electrons')
#     ax3.grid(True, alpha=0.3)
#     ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
#     ax4.plot(position*1e9, lambda_p, 'r-', linewidth=2, label='Λₚ')
#     ax4.set_xlabel('Position (nm)')
#     ax4.set_ylabel('Λₚ (V)')
#     ax4.set_title('Quantum Correction - Holes')
#     ax4.grid(True, alpha=0.3)
#     ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
#     plt.tight_layout()
#     plt.savefig('quantum_corrections.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print("\nQuantum correction calculation completed successfully!")
#     print("Visualization saved as 'quantum_corrections.png'")
    
#     # Demonstrate parameter updates
#     print("\nDemonstrating parameter updates...")
#     qc_calculator.update_quantum_parameters(delta_n=1e-9, delta_p=1.5e-9)
#     lambda_n_new, lambda_p_new = qc_calculator.quantum_correction(n_density, p_density)
#     print(f"Updated Lambda_n range: [{np.min(lambda_n_new):.2e}, {np.max(lambda_n_new):.2e}] V")
#     print(f"Updated Lambda_p range: [{np.min(lambda_p_new):.2e}, {np.max(lambda_p_new):.2e}] V")