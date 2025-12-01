import cv2
import numpy as np
from src.reconstruction import reconstruct_two_views, get_points_from_matches
from src.feature_matching import find_and_filter_matches
from src.optimization import bundle_adjustment  # Requires src/optimization.py

class SfMMap:
    def __init__(self, K):
        self.K = K
        self.points_3d = []  #  [x, y, z]
        self.colors = []     # [r, g, b]
        self.poses = []      #  (R, t)
        
        # Feature Management
        self.point_cloud_des = [] # List of descriptors for 3D points
        
        # Frame-to-Frame Tracking map
        self.map_2d_3d = {}
        
        # Bundle Adjustment Data
        self.observations = [] 
        
        self.last_img = None
        self.last_kp = None
        self.last_des = None

    def initialize(self, img1, img2, lowe_ratio=0.75):
        """Bootstrap the map with the first two images."""
        print("Initializing Map with first two images...")
        
        # 1. Match and Reconstruct
        matches, kp1, kp2, _ = find_and_filter_matches(img1, img2, lowe_ratio=lowe_ratio)
        pts1, pts2 = get_points_from_matches(matches, kp1, kp2)
        
        if len(pts1) < 8:
            print("Initialization failed: Not enough matches.")
            return

        # Reconstruct
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        mask = mask.ravel()
        pts1_inliers = pts1[mask == 1]
        pts2_inliers = pts2[mask == 1]
        
        _, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, self.K)
        
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))
        points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        # Store 3D Points & Colors
        self.points_3d = list(points_3d)
        self.poses.append((np.eye(3), np.zeros((3, 1)))) # Camera 0
        self.poses.append((R, t))                        # Camera 1
        
        for i in range(len(pts1_inliers)):
            # Frame 0 sees point i
            self.observations.append((0, i, pts1_inliers[i][0], pts1_inliers[i][1]))
            # Frame 1 sees point i
            self.observations.append((1, i, pts2_inliers[i][0], pts2_inliers[i][1]))

        for pt in pts1_inliers:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < img1.shape[0] and 0 <= x < img1.shape[1]:
                self.colors.append(img1[y, x] / 255.0)
            else:
                self.colors.append([0, 0, 0])

        # 2. Store Descriptors and Build Map
        valid_match_indices = [i for i, m in enumerate(mask) if m == 1]
        
        sift = cv2.SIFT_create()
        kp2_full, des2_full = sift.detectAndCompute(img2, None)
        
        self.point_cloud_des = []
        self.map_2d_3d = {} # Reset map
        
        for i, match_idx in enumerate(valid_match_indices):
            kp2_idx = matches[match_idx].trainIdx
            
            self.point_cloud_des.append(des2_full[kp2_idx])
            self.map_2d_3d[kp2_idx] = i
            
        self.point_cloud_des = np.array(self.point_cloud_des)

        self.last_img = img2
        self.last_kp = kp2_full
        self.last_des = des2_full
        
        print(f"Map initialized with {len(self.points_3d)} points.")

    def add_view(self, new_img, lowe_ratio=0.75):
        """Add a new view using Hybrid Tracking (Map + Frame-to-Frame)."""
        
        sift = cv2.SIFT_create()
        kp_new, des_new = sift.detectAndCompute(new_img, None)
        
        if des_new is None: 
            print("No features found.")
            return

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Gather PnP Correspondences
        pnp_matches = {} 

        #  Match against 3D Point Descriptors (Global)
        if len(self.point_cloud_des) > 0:
            knn_matches_3d = bf.knnMatch(des_new, self.point_cloud_des, k=2)
            for m, n in knn_matches_3d:
                if m.distance < 0.8 * n.distance:
                    pnp_matches[m.queryIdx] = m.trainIdx

        #  Match against Last Frame (Local Fallback)
        knn_matches_2d = bf.knnMatch(self.last_des, des_new, k=2)
        good_matches_2d = [] 
        
        for m, n in knn_matches_2d:
            if m.distance < 0.8 * n.distance:
                good_matches_2d.append(m)
                if m.queryIdx in self.map_2d_3d:
                    pt3d_idx = self.map_2d_3d[m.queryIdx]
                    if m.trainIdx not in pnp_matches:
                        pnp_matches[m.trainIdx] = pt3d_idx

        # Prepare Data for solvePnPRansac
        object_points = []
        image_points = []
        
        pnp_keys = list(pnp_matches.keys())
        
        for kp_idx in pnp_keys:
            pt3d_idx = pnp_matches[kp_idx]
            object_points.append(self.points_3d[pt3d_idx])
            image_points.append(kp_new[kp_idx].pt)

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        if len(object_points) < 6:
            print(f"PnP Failed: Not enough points ({len(object_points)} < 6). Skipping.")
            return

        # PHASE 2: Pose Estimation
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.K, None, 
            flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=8.0, confidence=0.99
        )
        
        if not success or inliers is None or len(inliers) < 5:
            print("PnP Failed: Geometric check failed.")
            return

        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        self.poses.append((R, t))
        current_cam_idx = len(self.poses) - 1
        
        final_map_2d_3d = {}
        inliers_flat = inliers.ravel()
        
        for i in inliers_flat:
            kp_idx = pnp_keys[i]       
            pt3d_idx = pnp_matches[kp_idx] 
            
            #  Update Map for tracking in next frame
            final_map_2d_3d[kp_idx] = pt3d_idx
            
            #  RECORD OBSERVATION 
            pt_2d = kp_new[kp_idx].pt
            self.observations.append((current_cam_idx, pt3d_idx, pt_2d[0], pt_2d[1]))

        # Triangulate NEW Points
        
        R_last, t_last = self.poses[-2]
        P_last = self.K @ np.hstack((R_last, t_last))
        P_new = self.K @ np.hstack((R, t))
        
        pts_last_tri = []
        pts_new_tri = []
        des_for_new_points = []
        new_kp_indices = []
        
        for m in good_matches_2d:
            # Only triangulating points that are not already in 3D
            if m.trainIdx not in final_map_2d_3d:
                pts_last_tri.append(self.last_kp[m.queryIdx].pt)
                pts_new_tri.append(kp_new[m.trainIdx].pt)
                des_for_new_points.append(des_new[m.trainIdx])
                new_kp_indices.append(m.trainIdx)

        points_new_3d = []
        added_count = 0
        
        if len(pts_last_tri) > 0:
            pts_last_tri_np = np.array(pts_last_tri).T
            pts_new_tri_np = np.array(pts_new_tri).T
            
            points_4d = cv2.triangulatePoints(P_last, P_new, pts_last_tri_np, pts_new_tri_np)
            candidates = (points_4d[:3] / points_4d[3]).T
            
            new_descriptors_list = []
            
            # Start index for new points
            start_new_idx = len(self.points_3d)
            
            for i, pt in enumerate(candidates):
                pt_cam = R @ pt + t.flatten()
                if pt_cam[2] > 0.01: 
                    self.points_3d.append(pt)
                    new_descriptors_list.append(des_for_new_points[i])
                    
                    # Update Map: 
                    new_pt_idx = start_new_idx + added_count
                    final_map_2d_3d[new_kp_indices[i]] = new_pt_idx
                    
                    # --- RECORD OBSERVATIONS (New Point) ---
                    # Seen in Last Frame
                    u_last, v_last = pts_last_tri[i]
                    self.observations.append((current_cam_idx - 1, new_pt_idx, u_last, v_last))
                    
                    # Seen in Current Frame
                    u_new, v_new = pts_new_tri[i]
                    self.observations.append((current_cam_idx, new_pt_idx, u_new, v_new))

                    # Color
                    x, y = int(u_new), int(v_new)
                    if 0 <= y < new_img.shape[0] and 0 <= x < new_img.shape[1]:
                        self.colors.append(new_img[y, x] / 255.0)
                    else:
                        self.colors.append([0, 0, 0])
                    added_count += 1
            
            if len(new_descriptors_list) > 0:
                self.point_cloud_des = np.vstack((self.point_cloud_des, np.array(new_descriptors_list)))

        print(f"View Added. Inliers: {len(inliers)}, New Points: {added_count}, Total: {len(self.points_3d)}")

        # Update State 
        self.map_2d_3d = final_map_2d_3d
        self.last_img = new_img
        self.last_kp = kp_new
        self.last_des = des_new

    def refine(self):
        """
        Runs Bundle Adjustment to refine poses and 3D points.
        Requires enough observations.
        """
        if len(self.observations) < 100:
            print("Not enough observations for Bundle Adjustment.")
            return

        print(f"--- Running Bundle Adjustment on {len(self.poses)} cameras and {len(self.points_3d)} points ---")
        
        try:
            new_poses, new_points = bundle_adjustment(
                self.poses, 
                self.points_3d, 
                self.observations, 
                self.K
            )
            
            # Update State
            self.poses = new_poses
            self.points_3d = new_points
            print("--- Refinement Finished ---")
            
        except Exception as e:
            print(f"Refinement Failed: {e}")