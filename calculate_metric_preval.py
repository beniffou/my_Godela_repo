import numpy as np
import h5py
import ot
from scipy.spatial.distance import cdist
import os

# Configuration - Set the two objects to compare
OBJECT1_PATH = "results/T_beam_131.h5"  # Change to your first object path
OBJECT2_PATH = "results/T_beam_132.h5"  # Change to your second object path

def load_pointcloud(h5_filename):
    """Load nodal data from HDF5 file."""
    with h5py.File(h5_filename, 'r') as f:
        nodal_data = f['nodal_data'][:]
    return nodal_data

def normalize_data(data1, data2):
    """
    Normalize coordinates and forces using statistics from both point clouds.
    This ensures both point clouds are on a comparable scale.
    """
    # Extract coordinates and forces
    coords1 = data1[:, :3]
    coords2 = data2[:, :3]
    forces1 = data1[:, 3:6]
    forces2 = data2[:, 3:6]
    
    # Combine data for normalization
    all_coords = np.vstack([coords1, coords2])
    all_forces = np.vstack([forces1, forces2])
    
    # Calculate mean and std
    mean_coords = np.mean(all_coords, axis=0)
    std_coords = np.std(all_coords, axis=0)
    mean_forces = np.mean(all_forces, axis=0)
    std_forces = np.std(all_forces, axis=0)
    
    # Avoid division by zero
    std_coords[std_coords == 0] = 1.0
    std_forces[std_forces == 0] = 1.0
    
    # Normalize each dataset
    norm_coords1 = (coords1 - mean_coords) / std_coords
    norm_forces1 = (forces1 - mean_forces) / std_forces
    norm_data1 = np.hstack([norm_coords1, norm_forces1, data1[:, 6:7]])
    
    norm_coords2 = (coords2 - mean_coords) / std_coords
    norm_forces2 = (forces2 - mean_forces) / std_forces
    norm_data2 = np.hstack([norm_coords2, norm_forces2, data2[:, 6:7]])
    
    return norm_data1, norm_data2

def wasserstein_distance(P, Q, num_iter_max=1000000):
    """
    Compute Wasserstein distance between two point clouds P and Q.
    P and Q are arrays with columns: [norm_x, norm_y, norm_z, norm_fx, norm_fy, norm_fz, s].
    """
    n_p = len(P)
    n_q = len(Q)
    
    # Compute cost matrix using Euclidean distance between all features
    cost_matrix = cdist(P, Q, 'euclidean')
    
    # Uniform distributions
    a = np.ones(n_p) / n_p
    b = np.ones(n_q) / n_q
    
    # Compute Wasserstein distance with increased iteration limit
    try:
        # Try with the increased iteration limit
        distance = ot.emd2(a, b, cost_matrix, numItermax=num_iter_max)
    except Exception as e:
        print(f"Error in EMD calculation: {e}")
        print("Falling back to Sinkhorn approximation")
        # Fall back to Sinkhorn approximation if EMD fails
        reg = 0.1  # Regularization parameter
        distance = ot.sinkhorn2(a, b, cost_matrix, reg=reg)
    
    return distance

def main():
    # Load the two point clouds
    data1 = load_pointcloud(OBJECT1_PATH)
    data2 = load_pointcloud(OBJECT2_PATH)
    
    print(f"Point cloud 1 size: {data1.shape}")
    print(f"Point cloud 2 size: {data2.shape}")
    
    # Normalize the data using combined statistics
    data1_norm, data2_norm = normalize_data(data1, data2)
    
    # Compute Wasserstein distance with increased iteration limit
    distance = wasserstein_distance(data1_norm, data2_norm, num_iter_max=1000000)
    
    print(f"Wasserstein distance between {os.path.basename(OBJECT1_PATH)} and {os.path.basename(OBJECT2_PATH)}: {distance}")
    
    # Return the distance for potential further use
    return distance

if __name__ == "__main__":
    main()