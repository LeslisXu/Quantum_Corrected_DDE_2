# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 03:46:33 2018

@author: Tim
"""

#Physical Constants (these should not be changed)  (constexpr b/c known at compile-time)
q =  1.60217646e-19         #elementary charge, C
kb = 1.3806503e-23          #Boltzmann const., J/k
T = 296.                    #temperature K
Vt = (kb*T)/q               #thermal voltage V
epsilon_0 =  8.85418782e-12 #F/m


# 2D Grid Utility Functions
def ij_to_index(i, j, nx):
    """Convert 2D grid coordinates (i,j) to linear index"""
    return i * nx + j

def index_to_ij(idx, nx):
    """Convert linear index to 2D grid coordinates (i,j)"""
    i = idx // nx
    j = idx % nx
    return i, j

def get_neighbors_2d(i, j, nx, ny):
    """
    Get the linear indices of neighboring points in 2D grid
    Returns: (left, right, bottom, top) neighbors or None if boundary
    """
    left = ij_to_index(i, j-1, nx) if j > 0 else None
    right = ij_to_index(i, j+1, nx) if j < nx-1 else None
    bottom = ij_to_index(i-1, j, nx) if i > 0 else None
    top = ij_to_index(i+1, j, nx) if i < ny-1 else None
    
    return left, right, bottom, top

def is_boundary_point(i, j, nx, ny):
    """Check if point (i,j) is on the boundary of the grid"""
    return i == 0 or i == ny-1 or j == 0 or j == nx-1

def get_boundary_type(i, j, nx, ny):
    """
    Determine boundary type for point (i,j)
    Returns: 'left', 'right', 'bottom', 'top', 'corner', or 'interior'
    """
    if not is_boundary_point(i, j, nx, ny):
        return 'interior'
    
    # Corner points
    if (i == 0 and j == 0) or (i == 0 and j == nx-1) or \
       (i == ny-1 and j == 0) or (i == ny-1 and j == nx-1):
        return 'corner'
    
    # Edge points
    if i == 0:
        return 'bottom'
    elif i == ny-1:
        return 'top'
    elif j == 0:
        return 'left'
    elif j == nx-1:
        return 'right'

def create_2d_laplacian_matrix(nx, ny, dx, dy):
    """
    Create 2D Laplacian matrix for finite difference discretization
    Using 5-point stencil: (V[i,j-1] + V[i,j+1])/dx² + (V[i-1,j] + V[i+1,j])/dy² - 2*V[i,j]*(1/dx² + 1/dy²)
    """
    from scipy.sparse import diags, csr_matrix
    
    n_total = nx * ny
    
    # Coefficient arrays
    main_diag = np.zeros(n_total)
    x_lower = np.zeros(n_total)  # j-1 direction
    x_upper = np.zeros(n_total)  # j+1 direction  
    y_lower = np.zeros(n_total)  # i-1 direction
    y_upper = np.zeros(n_total)  # i+1 direction
    
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    for i in range(ny):
        for j in range(nx):
            idx = ij_to_index(i, j, nx)
            
            if is_boundary_point(i, j, nx, ny):
                # Boundary points - will be handled by boundary conditions
                main_diag[idx] = 1.0
            else:
                # Interior points - 5-point stencil
                main_diag[idx] = -2.0 * (dx2_inv + dy2_inv)
                
                # x-direction neighbors
                if j > 0:
                    x_lower[idx] = dx2_inv
                if j < nx-1:
                    x_upper[idx] = dx2_inv
                    
                # y-direction neighbors  
                if i > 0:
                    y_lower[idx] = dy2_inv
                if i < ny-1:
                    y_upper[idx] = dy2_inv
    
    # Create sparse matrix
    # Offsets: [-nx, -1, 0, 1, nx] for [y_lower, x_lower, main, x_upper, y_upper]
    offsets = [-nx, -1, 0, 1, nx]
    diagonals = [y_lower[nx:], x_lower[1:], main_diag, x_upper[:-1], y_upper[:-nx]]
    
    matrix = diags(diagonals, offsets, shape=(n_total, n_total), format='csr')
    
    return matrix