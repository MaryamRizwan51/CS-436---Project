import cv2
import numpy as np
import open3d as o3d

def get_intrinsic_matrix(img_shape):
    """
    Approximate the intrinsic camera matrix K.
    Assumption from project manual:
    - Principal point (cx, cy) is the image center.
    - Focal lengths (fx, fy) are equal to the image width.
    """
    h, w = img_shape[:2]
    fx = w
    fy = w
    cx = w / 2
    cy = h / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
    ])
    return K

def get_points_from_matches(matches, kp1, kp2):
    """
    Convert keypoint objects to numpy arrays of coordinates.
    Returns: pts1, pts2 (Nx2 arrays)
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def reconstruct_two_views(pts1, pts2, K):
    
    # 1. Find Essential Matrix using RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=3.0)
    
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    
    print(f"Essential Matrix found. Inliers: {len(pts1_inliers)} / {len(pts1)}")

    # 2. Recover Pose 
    points, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    
    # 3. Triangulate Points
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ P1
    
    P2 = np.hstack((R, t))
    P2 = K @ P2
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
    
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    # Return the 3D points, Pose, AND the inliers
    return points_3d, R, t, pts1_inliers

def create_point_cloud(points_3d, colors=None):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
    return pcd

def save_ply(filename, points_3d, colors=None):
    
    pcd = create_point_cloud(points_3d, colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")