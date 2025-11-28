import cv2
import numpy as np

def find_and_filter_matches(img1, img2, lowe_ratio=0.75):
    
    # 1. Initialize SIFT Detector
    sift = cv2.SIFT_create()

    # 2. Find Keypoints and Descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. Initialize Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 4. Find k-Nearest-Neighbor Matches (k=2)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # 5. Filter Matches using Lowe's Ratio Test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)
            
    print(f"Found {len(kp1)} keypoints in img1 and {len(kp2)} in img2.")
    print(f"Found {len(knn_matches)} initial matches.")
    print(f"Filtered down to {len(good_matches)} good matches using Lowe's ratio test.")

    # 6. Create Visualization
    match_visualization = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good_matches, 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return good_matches, kp1, kp2, match_visualization