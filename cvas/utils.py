from PIL import Image, ImagePalette
import cv2
import numpy as np
import pathlib
from itertools import combinations


def get_parent_dir_path() -> str:
    return pathlib.Path(__file__).parent.parent.absolute()


def sort_points(points):
    center = np.mean(points, axis=0)

    tl_cond = np.logical_and(points[:, 0] <= center[0], points[:, 1] <= center[1])
    tl = points[tl_cond]
    tr_cond = np.logical_and(points[:, 0] > center[0], points[:, 1] <= center[1])
    tr = points[tr_cond]
    br_cond = np.logical_and(points[:, 0] > center[0], points[:, 1] > center[1])
    br = points[br_cond]
    bl_cond = np.logical_and(points[:, 0] <= center[0], points[:, 1] > center[1])
    bl = points[bl_cond]

    return np.vstack([tl, tr, br, bl])


def get_4points_with_highest_area(points):
    if len(points) < 4:
      return []

    if len(points) == 4:
      return points

    max_combination = []
    max_area = 0

    for i in combinations(points, 4):
      area = cv2.contourArea(np.array(i))

      if max_area < area:
        max_combination = i
        max_area = area

    return max_combination

def is_shape_inside_shape(shape1, shape2):
    for point in shape1:
        if cv2.pointPolygonTest(np.array(shape2), tuple(point), False) < 0:
            return False

    return True

class BBObject:
    points = []
    center = ()

    def __init__(self, points):
        self.center = get_center(points)
        self.points = points

# it returns list of numpy arrays with object coordinates
# One object has 4 points, every point has two coordinates
def get_object_coordinates(segment_frame):
    contours, _ = cv2.findContours(cv2.cvtColor(segment_frame, cv2.COLOR_RGB2GRAY)[:, :].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = []

    def getApprox(contour, alpha):
        epsilon = alpha * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    for contour in contours:
        corner_points = getApprox(contour, 0.01)
        points = np.float32(list(map(lambda x: x[0], corner_points)))
        points = sort_points(points)

        try:
            for point in points:
                cv2.circle(segment_frame, tuple(point), 20, (255, 150, 20), 10)
        except:
            pass

        points = get_4points_with_highest_area(points)

        if len(points) == 4:
            result.append(points)

    for comb in combinations(list(range(len(result))), 2) :
        shape1 = result[comb[0]]
        shape2 = result[comb[1]]

        if len(shape1) == 0 or len(shape2) == 0:
            continue

        if is_shape_inside_shape(shape1, shape2):
            result[comb[0]] = []
        elif is_shape_inside_shape(shape2, shape1):
            result[comb[1]] = []

    final_result = []

    for i in result:
        if len(i) > 0:
            final_result.append(BBObject(i))

    return final_result


def optimize_keypoint(src, src_prev, kp=None, match_count=None):
    match_count = match_count or 20
    detector = cv2.ORB_create()
    print("detector")
    # Detect the keypoints using ORB Detector, compute the descriptors
    src_kp, des1 = detector.detectAndCompute(src, None)
    prev_kp, des2 = detector.detectAndCompute(src_prev, None)
    # print('k, d', kp1, des1)
    # if kp:
    #     kp2, des2 = kp.get("kp"), kp.get("des")
    #     src_kp, prev_kp = kp.get("src"), kp.get("previous")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # matches = matcher.knnMatch(des1, des2, 2)
    matches = matcher.match(des1, des2)
    print(len(matches))
        # matcher = cv2.BFMatcher()
        # matches = matcher.match(src_kp, trans_kp)

    # # -- Filter matches using the Lowe's ratio test
    # ratio_thresh = 0.7
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches.append(m)
    # -- Draw matches
    img_matches = np.empty((max(src.shape[0], src_prev.shape[0]), src.shape[1] + src_prev.shape[1], 3), dtype=np.uint8)
    # final_img = cv2.drawMatches(src, src_kp, src_prev, prev_kp, good_matches, img_matches,
    #                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    final_img = cv2.drawMatches(src, src_kp, src_prev, prev_kp, matches[:match_count], img_matches)

    return final_img


def stabilise_transformation(src, src_prev):
    brows, bcols = src_prev.shape[:2]
    dst = cv2.addWeighted(src, 0.3, src_prev, 0.7, 0)
    rows, cols, channels = src.shape
    src[int(brows / 2) - int(rows / 2):int(brows / 2) + int(rows / 2),
        int(bcols / 2) - int(cols / 2):int(bcols / 2) + int(cols / 2)] = dst
    cv2.normalize(src, dst, 0, 1, cv2.NORM_L1, -1)
    return src


def generate_result_frame(frame, frame_index, replacement, objects):
    target_height, target_width, _ = replacement.shape

    counter = 0
    for global_obj_index in objects:
        global_obj = objects[global_obj_index]
        coords = global_obj.get_frame_coords(frame_index)
        if coords is None:
            continue

        original_coordinates = sort_points(np.array([[0, 0],  [target_width, 0],  [target_width, target_height], [0, target_height]], np.float32))
        object = sort_points(np.array(coords.points, np.float32))

        transformation = cv2.getPerspectiveTransform(original_coordinates, object)

        transformed = cv2.warpPerspective(replacement, transformation, (frame.shape[1], frame.shape[0]))
        transformed = enhance_colors(frame, transformed)


        cv2.fillConvexPoly(frame, sort_points(object.astype(int)), 0, 16)

        frame = frame + transformed

        counter += 1

    return frame


# Adjust coordinates of objects:
# position correcation, smoothing
def adjust_coordinates(object_coordinates) -> np.ndarray:
    return object_coordinates


def enhance_colors(parent, injection):
    # aug = Compose([FDA([parent], p=1, read_fn=lambda x: x)])
    # result = aug(image=injection)['image']
    #
    # cv2.imshow('img', result)
    return injection


def get_center(points):
    M = cv2.moments(np.array(points))
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

def blur_transformed(src, kernel=(4, 4), transformation=None, with_edge=False):
    # Create ROI coordinates
    # x, y = 0, 0
    # w, h = int(src.shape[1]), int(src.shape[0])
    # blur_gaus = cv2.GaussianBlur(cv2.rectangle(src, (1, 1), (src.shape[1]-1, src.shape[0]-1), (150, 150, 150), 2), (9, 9), 0)
    #
    # # Insert ROI back into image
    # src[y:y + h, x:x + w] = blur
    #
    if with_edge:
        cv2.rectangle(src, (1, 1), (src.shape[1]-1, src.shape[0]-1), (150, 150, 150), 2)
    blur = cv2.filter2D(src, -1, np.ones(kernel, np.float32) / 18)
    return blur
