import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time

def project_points(points_3d, camera_params, K):
    """Projects 3D points to 2D image plane."""
    points_proj = []
    
    # Process in batches or loop (loop is fine for readability here)
    for i in range(len(points_3d)):
        # Unpack
        r_vec = camera_params[i, :3]
        t_vec = camera_params[i, 3:]
        
        # Convert to Rotation Matrix
        R, _ = cv2.Rodrigues(r_vec)
        
        # Transform: P_cam = R * P_world + t
        # (Using dot for explicit clarity, though slower than broadcasting)
        p_cam = np.dot(R, points_3d[i]) + t_vec
        
        # Avoid division by zero if point is behind camera
        z = p_cam[2]
        if z < 1e-5: z = 1e-5
            
        # Project
        p_img = p_cam[:2] / z
        
        # Apply Intrinsic K
        u = K[0,0] * p_img[0] + K[0,2]
        v = K[1,1] * p_img[1] + K[1,2]
        
        points_proj.append([u, v])
        
    return np.array(points_proj)

def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Computes residuals."""
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    points_3d_obs = points_3d[point_indices]
    camera_params_obs = camera_params[camera_indices]
    
    points_proj = project_points(points_3d_obs, camera_params_obs, K)
    
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Computes the Jacobian sparsity structure.
    This tells the solver which parameters affect which residuals.
    """
    m = camera_indices.size * 2 # number of residuals (2 per observation)
    n = n_cameras * 6 + n_points * 3 # total parameters
    
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    
    # Mark camera parameters (6 params per camera)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    # Mark point parameters (3 params per point)
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def bundle_adjustment(poses, points_3d, observations, K):
    """
    Fast Sparse Bundle Adjustment.
    """
    start_time = time.time()
    n_cameras = len(poses)
    n_points = len(points_3d)
    
    print(f"   > Setup: {n_cameras} cams, {n_points} points, {len(observations)} obs.")

    # 1. Prepare Parameters
    camera_params = []
    for R, t in poses:
        r_vec, _ = cv2.Rodrigues(R)
        camera_params.append(np.hstack((r_vec.flatten(), t.flatten())))
    camera_params = np.array(camera_params)
    
    # 2. Prepare Indices
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for cam_idx, pt_idx, u, v in observations:
        camera_indices.append(cam_idx)
        point_indices.append(pt_idx)
        points_2d.append([u, v])
        
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    # 3. Compute Sparsity Matrix (The Speed Fix!)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    # 4. Run Optimization
    x0 = np.hstack((camera_params.ravel(), np.array(points_3d).ravel()))
    
    # We restrict max_nfev to 20 to prevent it from running forever even with sparsity
    res = least_squares(
        reprojection_error, 
        x0, 
        jac_sparsity=A,       # <--- CRITICAL: Uses sparse solver
        verbose=2, 
        x_scale='jac', 
        ftol=1e-3,            # Relaxed tolerance for speed
        method='trf', 
        max_nfev=30,          # <--- CRITICAL: Hard limit on iterations
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K)
    )
    
    # 5. Unpack
    optimized_params = res.x
    new_camera_params = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
    new_points_3d = optimized_params[n_cameras * 6:].reshape((n_points, 3))
    
    new_poses = []
    for params in new_camera_params:
        r_vec = params[:3]
        t_vec = params[3:].reshape(3, 1)
        R, _ = cv2.Rodrigues(r_vec)
        new_poses.append((R, t_vec))
        
    print(f"   > BA Finished in {time.time() - start_time:.2f}s")
    return new_poses, new_points_3d.tolist()