import cv2
import numpy as np

def find_and_filter_matches(img1, img2, lowe_ratio=0.75):
    """
    Finds, filters, and visualizes SIFT feature matches between two images.
    
    Args:
        img1 (np.ndarray): The first image (grayscale or color).
        img2 (np.ndarray): The second image (grayscale or color).
        lowe_ratio (float): The ratio for Lowe's ratio test.
        
    Returns:
        tuple: A tuple containing:
            - good_matches (list): A list of good DMatch objects.
            - kp1 (tuple): Keypoints from image 1.
            - kp2 (tuple): Keypoints from image 2.
            - match_visualization (np.ndarray): An image visualizing the matches.
    """
    
    # 1. Initialize SIFT Detector
    # SIFT (Scale-Invariant Feature Transform) is robust to scale and rotation changes.
    sift = cv2.SIFT_create()

    # 2. Find Keypoints and Descriptors
    # Keypoints are the "interesting" points, and descriptors are the unique fingerprints.
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. Initialize Brute-Force Matcher
    # BFMatcher checks all descriptors from one set against all from the other.
    # NORM_L2 is the standard distance metric for SIFT.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 4. Find k-Nearest-Neighbor Matches (k=2)
    # We find the 2 best matches for each descriptor.
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # 5. Filter Matches using Lowe's Ratio Test
    # This is a critical step to get high-quality matches.
    # A match is "good" if its distance is significantly smaller (by lowe_ratio)
    # than the distance of the second-best match.
    good_matches = []
    for m, n in knn_matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)
            
    print(f"Found {len(kp1)} keypoints in img1 and {len(kp2)} in img2.")
    print(f"Found {len(knn_matches)} initial matches.")
    print(f"Filtered down to {len(good_matches)} good matches using Lowe's ratio test.")

    # 6. Create Visualization
    # Draw the good matches on a new image.
    match_visualization = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good_matches, 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return good_matches, kp1, kp2, match_visualization