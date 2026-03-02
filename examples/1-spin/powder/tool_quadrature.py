import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import lebedev_rule
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

def generate_combined_quadrature(lebedev_degree=31, n_gamma_points=12, save_file=True):
    r"""
    Generate combined quadrature points for powder average calculation:
    - Lebedev quadrature for orientation on the sphere (α, β)
    - Gauss-Legendre quadrature for rotation around molecular axis (γ)
    
    Derivation of the combined quadrature:
          \int_0^2pi d alpha \int_0^pi sin(beta) d beta \int_0^2pi d gamma f(alpha, beta, gamma) 
        = \int_0^2pi d alpha \int_0^pi sin(beta) d beta (\sum_k w^{gauss_legendre}_k f(alpha, beta, gamma_k)) # gauss_legendre
        = \int_0^2pi d alpha \int_0^pi sin(beta) d beta g(alpha, beta)
        = sum_i,j w^lebedev_ij g(alpha_i, beta_j) # Lebedev quadrature
        = sum_i,j,k w^lebedev_ij w^{gauss_legendre}_k * f(alpha_i, beta_j, gamma_k)
    Parameters:
    -----------
    lebedev_degree : int
        Degree of Lebedev quadrature
        Available degrees in quadpy: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
        31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131
    n_gamma_points : int
        Number of Gauss-Legendre points for γ integration
        
    Returns:
    --------
    euler_angles : ndarray, shape (n, 3)
        Array of combined (α, β, γ) points for integration
        Intrisic ZYZ, alpha-first Euler angles
    weights : ndarray, shape (n,)
        Array of combined weights for each point
    """
    # 1. Generate Lebedev quadrature points for the sphere (α, β)
    try:
        sphere_points, sphere_weights = lebedev_rule(lebedev_degree)
    except:
        # Fall back to available degree if specified one is not available
        available_degrees = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 
                             35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 
                             113, 119, 125, 131]
        # Find closest available degree
        closest_degree = min(available_degrees, key=lambda x: abs(x - lebedev_degree))
        print(f"Lebedev degree {lebedev_degree} not available, using {closest_degree} instead")
        sphere_points, sphere_weights = lebedev_rule(closest_degree)

    # Rescale weights for averaging over the sphere
    sphere_weights = sphere_weights / (4 * np.pi) # 4 * np.pi is the surface area of the unit sphere
    
    # 2. Generate Gauss-Legendre quadrature points for γ rotation
    gamma_points, gamma_weights = roots_legendre(n_gamma_points)
    
    # Rescale γ points from [-1, 1] to [0, 2π]
    gamma_points = gamma_points * np.pi + np.pi
    gamma_weights = gamma_weights * np.pi # For integration over [0, 2π]

    # Rescale weights for averaging over the full circle
    gamma_weights = gamma_weights / (2*np.pi) # 2 * pi is the circumference of the unit circle
    
    # 3. Combine the quadrature schemes
    euler_angles = []
    weights = []
    
    # Convert sphere points to Euler angles
    for i in range(sphere_points[0].shape[0]):
        x = sphere_points[0][i]
        y = sphere_points[1][i]
        z = sphere_points[2][i]
        # Convert Cartesian to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)  # Should be 1.0
        beta = np.arccos(z / r)  # Polar angle (0 to π)
        alpha = np.arctan2(y, x)  # Azimuthal angle (0 to 2π)
        if alpha < 0:
            alpha += 2*np.pi
            
        sphere_weight = sphere_weights[i]
        
        # Combine with γ rotation
        for j, gamma in enumerate(gamma_points):
            gamma_weight = gamma_weights[j]
            
            # Combined Euler angles
            euler_angles.append([alpha, beta, gamma])
            
            # Combined weight
            combined_weight = sphere_weight * gamma_weight
            weights.append(combined_weight)
    
    # Convert to numpy arrays
    euler_angles = np.array(euler_angles)
    weights = np.array(weights)
    
    # Save the quadrature points and weights to a file
    if save_file:
        with open(f'points_and_weights.txt', 'w') as f:
            f.write(f"# Lebedev-Gauss-Legendre quadrature points and weights\n")
            f.write(f"# {len(weights)} total quadrature points\n")
            f.write(f"# {sphere_points[0].shape[0]} Lebedev points (degree {lebedev_degree}) for sampling alpha and beta\n")
            f.write(f"# {len(gamma_points)} Gauss-Legendre points for sampling gamma\n")
            f.write("# alpha (deg) beta (deg) gamma (deg) weight\n")
            for angles, weight in zip(euler_angles, weights):
                angles_deg = np.degrees(angles)
                f.write(f"{angles_deg[0]:10.5f} {angles_deg[1]:10.5f} {angles_deg[2]:10.5f} {weight:15.6e}\n")
        print(f"Quadrature points and weights saved to points_and_weights.txt")
    
    return euler_angles, weights

def unit_function(alpha, beta, gamma):
    return 1

def integrate_unit_function(euler_angles, weights):
    """
    Integrate a unit function over the Lebedev-Gauss-Legendre quadrature points.
    
    Parameters:
    -----------
    euler_angles : ndarray, shape (n, 3)
        Array of combined (α, β, γ) points for integration
    weights : ndarray, shape (n,)
        Array of combined weights for each point
        
    Returns:
    --------
    integral : float
        Result of the integration
    """
    integral = 0.0
    for i in range(len(weights)):
        alpha, beta, gamma = euler_angles[i]
        integral += unit_function(alpha, beta, gamma) * weights[i]
    
    return integral

# Visualize the Lebedev quadrature points

def visualize_lebedev_points(euler_angles):

    # Extract sphere points (α, β) by taking unique combinations
    unique_ab = set()
    sphere_points = []
    for alpha, beta, _ in euler_angles:
        if (alpha, beta) not in unique_ab:
            unique_ab.add((alpha, beta))
            # Convert from spherical to Cartesian for visualization
            x = np.sin(beta) * np.cos(alpha)
            y = np.sin(beta) * np.sin(alpha)
            z = np.cos(beta)
            sphere_points.append([x, y, z])
    
    sphere_points = np.array(sphere_points)
    
    # Visualize the Lebedev points on the sphere
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot unit sphere wireframe
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        x = 0.98 * np.outer(np.cos(u), np.sin(v))
        y = 0.98 * np.outer(np.sin(u), np.sin(v))
        z = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='c', alpha=0.1)
        
        # Plot Lebedev points
        ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], 
                   color='r', s=50, label=f'Lebedev Points ({len(sphere_points)})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lebedev Quadrature Points on Unit Sphere')
        plt.legend()
        plt.tight_layout()
        plt.savefig('lebedev_points.png')
        plt.close()
        print("Visualization saved as 'lebedev_points.png'")
    except:
        print("3D visualization could not be created")

# Visualize the Gauss-Legendre points
def visualize_gauss_legendre_points(euler_angles):
    # Extract γ points from euler_angles
    gamma_points = euler_angles[:, 2]

    # Convert radians to degrees
    gamma_points = np.degrees(gamma_points)

    # Remove duplicates and sort
    gamma_points = np.unique(gamma_points)
    gamma_points = np.sort(gamma_points)

    plt.figure(figsize=(8, 4))
    plt.plot(gamma_points, np.zeros_like(gamma_points), 'ro', markersize=10)
    plt.title('Gauss-Legendre Points for γ Rotation')
    plt.xlabel('γ (degrees)')
    # Set x ticks at intervals of 30 degrees
    plt.xticks(np.arange(0, 361, 30))
    plt.yticks([])
    plt.grid()
    plt.tight_layout()
    plt.savefig('gauss_legendre_points.png')
    plt.close()
    print("Visualization saved as 'gauss_legendre_points.png'")

# Example usage
if __name__ == "__main__":

    # The saved angles are in degrees
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=3, n_gamma_points=13, save_file=True)
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=9, n_gamma_points=13, save_file=True)
    euler_angles, weights = generate_combined_quadrature(lebedev_degree=15, n_gamma_points=13, save_file=True)
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=31, n_gamma_points=13, save_file=True)
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=51, n_gamma_points=13, save_file=True)

    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=9, n_gamma_points=5, save_file=True)
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=9, n_gamma_points=9, save_file=True)
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=9, n_gamma_points=15, save_file=True)
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=9, n_gamma_points=21, save_file=True)
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=9, n_gamma_points=31, save_file=True)

    # The returned angles are in radians
    # euler_angles, weights = generate_combined_quadrature(lebedev_degree=9, n_gamma_points=13, save_file=False)

    # visualize_lebedev_points(euler_angles)
    # visualize_gauss_legendre_points(euler_angles)

    # integral = integrate_unit_function(euler_angles, weights)
    # print(f"Integral of unit function: {integral}")



