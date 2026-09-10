import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import lebedev_rule
import matplotlib.pyplot as plt

# Degrees for which a Lebedev rule is available
LEBEDEV_DEGREES = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                   35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107,
                   113, 119, 125, 131]

def closest_lebedev_degree(lebedev_degree):
    """
    The available Lebedev degree closest to the requested one. See LEBEDEV_DEGREES.
    """
    if lebedev_degree in LEBEDEV_DEGREES:
        return lebedev_degree
    closest_degree = min(LEBEDEV_DEGREES, key=lambda x: abs(x - lebedev_degree))
    print(f"Lebedev degree {lebedev_degree} not available, using {closest_degree} instead")
    return closest_degree

def field_direction(euler_angles):
    r"""
    Direction of the static magnetic field in the ZFS reference frame.

    euler_angles: ndarray, shape (n, 3), intrinsic ZYZ angles in degrees.
    Returns an ndarray of shape (n, 3) of unit vectors.

    The field lies along the global z axis, and tool_powder.get_emat writes the
    ZFS reference frame as emat = transpose(R_ZYZ(alpha, beta, gamma)), whose
    rows are the ZFS axes in global coordinates. The components of the field in
    the ZFS frame are therefore the third row of R_ZYZ, i.e. emat @ [0, 0, 1].
    """
    rotmats = R.from_euler('ZYZ', np.atleast_2d(euler_angles), degrees=True).as_matrix()
    return rotmats[:, 2, :]

def generate_powder_quadrature(lebedev_degree=31, save_file=True):
    r"""
    Generate quadrature points and weights for a powder average.

    The powder average of this model needs two angles, not three. The static
    field is along the global z axis and the g tensor is isotropic, so the only
    geometric variable is the direction of the field in the ZFS reference frame,

        n(beta, gamma) = emat @ [0, 0, 1]
                       = (-sin(beta) cos(gamma), sin(beta) sin(gamma), cos(beta))

    which does not contain the first Euler angle alpha. Rotating the ZFS frame
    about the field axis maps H -> U H U^dagger with U = exp(-i alpha S_z). That
    leaves the spectrum, <S_z>, the magnetization operator and the |Delta M_z| = 1
    phonon-coupling operator X unchanged, hence M(B) and M(t) as well.

    So a single Lebedev rule over n replaces the earlier product of a Lebedev
    rule over (alpha, beta) with a Gauss-Legendre rule over gamma: each Lebedev
    point is read as a field direction and mapped back onto Euler angles by

        alpha = 0,  beta = arccos(n_z),  gamma = atan2(n_y, -n_x)

    This costs n_lebedev orientations instead of n_lebedev * n_gamma, and is
    exact through spherical-harmonic degree lebedev_degree.

    Caveat: alpha is redundant only while the field stays along one global axis
    and the g tensor is isotropic (or its reference frame is co-rotated with the
    ZFS frame). A transverse drive, or any operator in the dissipator that is
    fixed in the global frame and is not S_z, brings the third angle back.

    Parameters:
    -----------
    lebedev_degree : int
        Degree of the Lebedev rule. See LEBEDEV_DEGREES for the available ones.
    save_file : bool
        Write the points and weights to points_and_weights.txt.

    Returns:
    --------
    euler_angles : ndarray, shape (n, 3)
        Array of (alpha, beta, gamma) points in degrees, as saved to the file.
        Intrinsic ZYZ, alpha-first Euler angles, with alpha = 0 throughout.
    weights : ndarray, shape (n,)
        Array of weights for each point. They sum up to 1.
    """
    # 1. Generate the Lebedev points, read as field directions in the ZFS frame
    lebedev_degree = closest_lebedev_degree(lebedev_degree)
    sphere_points, sphere_weights = lebedev_rule(lebedev_degree)

    # Rescale weights for averaging over the sphere
    weights = sphere_weights / (4 * np.pi) # 4 * np.pi is the surface area of the unit sphere

    # 2. Map each direction back onto Euler angles by inverting n(beta, gamma)
    nx, ny, nz = sphere_points
    alphas = np.zeros_like(nz)
    betas = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))
    gammas = np.degrees(np.arctan2(ny, -nx) % (2*np.pi))
    euler_angles = np.column_stack((alphas, betas, gammas))

    # 3. Check that the angles do reproduce the Lebedev directions
    error = np.abs(field_direction(euler_angles) - np.transpose(sphere_points)).max()
    if error > 1e-12:
        print(f"Error: the Euler angles miss the Lebedev directions by {error:.3e}. Stopping ...")
        exit(1)

    # Save the quadrature points and weights to a file
    if save_file:
        with open(f'points_and_weights.txt', 'w') as f:
            f.write(f"# Lebedev quadrature points and weights\n")
            f.write(f"# {len(weights)} Lebedev points (degree {lebedev_degree}) for sampling\n")
            f.write(f"# the field direction in the ZFS reference frame\n")
            f.write(f"# alpha is redundant and is therefore held at 0\n")
            f.write("# alpha (deg) beta (deg) gamma (deg) weight\n")
            for angles, weight in zip(euler_angles, weights):
                f.write(f"{angles[0]:10.5f} {angles[1]:10.5f} {angles[2]:10.5f} {weight:15.6e}\n")
        print(f"Quadrature points and weights saved to points_and_weights.txt")

    return euler_angles, weights

def unit_function(alpha, beta, gamma):
    return 1

def integrate_unit_function(euler_angles, weights):
    """
    Integrate a unit function over the Lebedev quadrature points.

    Parameters:
    -----------
    euler_angles : ndarray, shape (n, 3)
        Array of (alpha, beta, gamma) points for integration
    weights : ndarray, shape (n,)
        Array of weights for each point

    Returns:
    --------
    integral : float
        Result of the integration. It should be 1.
    """
    integral = 0.0
    for i in range(len(weights)):
        alpha, beta, gamma = euler_angles[i]
        integral += unit_function(alpha, beta, gamma) * weights[i]

    return integral

# Visualize the sampled field directions

def visualize_lebedev_points(euler_angles):

    # The sampled field directions in the ZFS reference frame
    sphere_points = field_direction(euler_angles)

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

# Example usage
if __name__ == "__main__":

    # The angles are in degrees, both saved and returned
    # euler_angles, weights = generate_powder_quadrature(lebedev_degree=3, save_file=True)
    # euler_angles, weights = generate_powder_quadrature(lebedev_degree=9, save_file=True)
    euler_angles, weights = generate_powder_quadrature(lebedev_degree=15, save_file=True)
    # euler_angles, weights = generate_powder_quadrature(lebedev_degree=31, save_file=True)
    # euler_angles, weights = generate_powder_quadrature(lebedev_degree=51, save_file=True)

    # visualize_lebedev_points(euler_angles)

    # integral = integrate_unit_function(euler_angles, weights)
    # print(f"Integral of unit function: {integral}")
