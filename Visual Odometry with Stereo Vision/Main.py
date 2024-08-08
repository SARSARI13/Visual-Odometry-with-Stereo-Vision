import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import pandas as pd


# Définir les constantes
rectified_value = True


def disparity_mapping(left_image, right_image):
    num_disparities = 6 * 16
    block_size = 7

    matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                    minDisparity=0,
                                    blockSize=block_size,
                                    P1=8 * 1 * block_size ** 2,
                                    P2=32 * 1 * block_size ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                    )

    left_image_disparity_map = matcher.compute(left_image, right_image).astype(np.float32) / 16.0

    return left_image_disparity_map

def decomposition(p):
    intrinsic_matrix, rotation_matrix, translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    translation_vector = (translation_vector / translation_vector[3])[:3]
    return intrinsic_matrix, rotation_matrix, translation_vector

def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified=rectified_value):
    focal_length = left_intrinsic[0][0]

    if rectified:
        baseline = right_translation[0] - left_translation[0]
    else:
        baseline = left_translation[0] - right_translation[0]

    left_disparity_map[left_disparity_map == 0.0] = 0.1
    left_disparity_map[left_disparity_map == -1.0] = 0.1

    depth_map = (focal_length * baseline) / left_disparity_map

    return depth_map

def stereo_depth(left_image, right_image, P0, P1):
    disp_map = disparity_mapping(left_image, right_image)
    l_intrinsic, l_rotation, l_translation = decomposition(P0)
    r_intrinsic, r_rotation, r_translation = decomposition(P1)
    depth = depth_mapping(disp_map, l_intrinsic, l_translation, r_translation)
    return depth

def read_calibration(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    calib = {}
    for line in lines:
        key, value = line.split(':', 1)
        calib[key.strip()] = np.array([float(x) for x in value.strip().split()])
    return calib

def read_image(img_path, img_idx):
    img_file = sorted(os.listdir(img_path))[img_idx]
    img = cv2.imread(os.path.join(img_path, img_file), cv2.IMREAD_GRAYSCALE)
    return img

def feature_matching(kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    query_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    train_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    FM, mask = cv2.findFundamentalMat(query_pts, train_pts, cv2.RANSAC + cv2.FM_8POINT)

    query_pts_IN = query_pts[mask.ravel() == 1]
    train_pts_IN = train_pts[mask.ravel() == 1]

    kp_query_IN = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in query_pts_IN]
    kp_train_IN = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in train_pts_IN]

    matches_IN = [cv2.DMatch(i, i, 0) for i in range(query_pts_IN.shape[0])]
    matches = sorted(matches_IN, key=lambda x: x.distance)

    return kp_query_IN, kp_train_IN, matches

def feature_extractor(image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def visualize_matches(first_image, second_image, kp_query_IN, kp_train_IN, matches):
    show_matches = cv2.drawMatches(first_image, kp_query_IN, second_image, kp_train_IN, matches, None, flags=2)
    plt.figure(figsize=(15, 5), dpi=100)
    plt.imshow(show_matches)
    plt.show()

def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, max_depth=3000):
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((3, 1))

    image1_points = np.float32([firstImage_keypoints[m.queryIdx].pt for m in matches])
    image2_points = np.float32([secondImage_keypoints[m.trainIdx].pt for m in matches])

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    points_3D = np.zeros((0, 3))
    outliers = []

    for indices, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]
        if z > max_depth:
            outliers.append(indices)
            continue
        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        points_3D = np.vstack([points_3D, np.array([x, y, z])])

    image1_points = np.delete(image1_points, outliers, 0)
    image2_points = np.delete(image2_points, outliers, 0)

    _, rvec, translation_vector, _ = cv2.solvePnPRansac(points_3D, image2_points, intrinsic_matrix, None)
    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

def visual_odometry(left_img_path, right_img_path, calibration, num_frames, ground_truth):
    fig = plt.figure(figsize=(14, 7))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.view_init(elev=-20, azim=270)
    ax_2d = fig.add_subplot(122)

    homo_matrix = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = homo_matrix[:3, :]

    P0 = calibration['P0'].reshape((3, 4))
    P1 = calibration['P1'].reshape((3, 4))

    left_intrinsic_matrix, _, _ = decomposition(P0)

    for i in range(num_frames - 1):
        left_image = read_image(left_img_path, i)
        right_image = read_image(right_img_path, i)
        next_left_image = read_image(left_img_path, i + 1)
        next_right_image = read_image(right_img_path, i + 1)

        depth = stereo_depth(left_image, right_image, P0, P1)

        keypoint_left_first, descriptor_left_first = feature_extractor(left_image)
        keypoint_left_next, descriptor_left_next = feature_extractor(next_left_image)

        kpi1, kpi2, matches = feature_matching(keypoint_left_first, keypoint_left_next, descriptor_left_first, descriptor_left_next)

        rotation_matrix, translation_vector, _, _ = motion_estimation(matches, kpi1, kpi2, left_intrinsic_matrix, depth)

        Transformation_matrix = np.eye(4)
        Transformation_matrix[:3, :3] = rotation_matrix
        Transformation_matrix[:3, 3] = translation_vector.T

        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
        trajectory[i + 1, :, :] = homo_matrix[:3, :]

        if i % 10 == 0:
            print(f'{i} frames have been computed')

    # Trajectoire estimée
    xs_estimated = trajectory[:, 0, 3]
    ys_estimated = trajectory[:, 1, 3]
    zs_estimated = trajectory[:, 2, 3]

    # Trajectoire de la vérité terrain
    xs_ground_truth = ground_truth[:, 0, 3]
    ys_ground_truth = ground_truth[:, 1, 3]
    zs_ground_truth = ground_truth[:, 2, 3]

    # Tracer les trajectoires
    ax_3d.plot(xs_estimated, ys_estimated, zs_estimated, c='darkorange', label='Trajectoire Estimée')
    ax_3d.plot(xs_ground_truth, ys_ground_truth, zs_ground_truth, c='dimgray', label='Vérité Terrain')
    ax_3d.set_title("Trajectoire Estimée vs Vérité Terrain")
    ax_3d.legend()

    ax_2d.plot(xs_estimated, zs_estimated, c='darkorange', label='Estimée')
    ax_2d.plot(xs_ground_truth, zs_ground_truth, c='dimgray', label='Vérité Terrain')
    ax_2d.set_xlabel('X')
    ax_2d.set_ylabel('Z')
    ax_2d.set_title('Trajectoire 2D (X vs Z)')
    ax_2d.legend()

    plt.show()

    return trajectory

def main():
    base_path = r'C:\Users\Yassine\Desktop\git\VOS'
    sequence_id = '00'
    left_img_path = os.path.join(base_path, sequence_id, 'image_0')
    right_img_path = os.path.join(base_path, sequence_id, 'image_1')
    calib_file = os.path.join(base_path, sequence_id, 'calib.txt')
    poses_file = os.path.join(base_path, sequence_id, 'pose.txt')

    calibration = read_calibration(calib_file)
    poses = pd.read_csv(poses_file, delimiter=' ', header=None)

    # Initialiser le tableau ground_truth
    ground_truth = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))

    num_frames = len(os.listdir(left_img_path))
    trajectory = visual_odometry(left_img_path, right_img_path, calibration, num_frames, ground_truth)

if __name__ == '__main__':
    main()
