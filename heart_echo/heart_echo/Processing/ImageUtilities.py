import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from scipy import stats


class ArcEstimationError(Exception):
    def __init__(self, *args, **kwargs):
        pass


class ImageUtilities:
    @staticmethod
    def convert_to_gray(img_array):
        return cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def resize_image(img_array, width, height, interpolation=cv2.INTER_LINEAR):
        return cv2.resize(img_array, (width, height), interpolation=interpolation)

    @staticmethod
    def get_frame_mask(img_array, frame_mask, threshold=20):
        two_level = True if threshold > 1 else False

        for r in range(img_array.shape[0]):
            if two_level:
                indices = (img_array[r] > 1).nonzero()[0]

                if len(indices) > 0:
                    frame_mask[r][indices] = np.maximum(frame_mask[r][indices],
                                                        np.repeat(threshold - 1, len(frame_mask[r][indices])))

                    indices = (img_array[r] > threshold).nonzero()[0]
                    frame_mask[r][indices] = 255

            else:
                indices = (img_array[r] > threshold).nonzero()[0]
                frame_mask[r][indices] = 255

        return frame_mask

    @staticmethod
    def smooth_image(img_array):
        # return cv2.bilateralFilter(img_array, 9, 75, 75)

        # kernel = np.ones((5, 5), np.float32) / 25
        # return cv2.filter2D(img_array, -1, kernel)

        return cv2.GaussianBlur(img_array, (5, 5), 0)

    @staticmethod
    def find_echo_segmentation_points(frame, threshold, debug=False):
        top_left, bottom, top_right = ImageUtilities.get_static_points(frame, threshold)
        arc_output = ImageUtilities.get_arc(frame, top_left, bottom, top_right)
        center = arc_output[0]
        radius = arc_output[1]
        bottom_left = ImageUtilities.get_bottom_left_endpoint(frame, top_left, center, radius)
        bottom_right = ImageUtilities.get_bottom_right_endpoint(frame, top_right, bottom_left, center, radius)

        # Calculate the 5th point 20% along the right line -- this is used to crop out any text
        point_5 = (
            int(np.round(top_right[0] + 0.2 * (bottom_right[0] - top_right[0]))),
            int(np.round(top_right[1] + 0.2 * (bottom_right[1] - top_right[1]))))

        if debug:
            ImageUtilities.show_image(
                ImageUtilities.draw_segmentation_points(frame, top_left, bottom_left, bottom, bottom_right,
                                                        point_5, top_right, center, radius))

        return top_left, bottom_left, bottom, bottom_right, point_5, top_right

    @staticmethod
    def get_bottom_left_endpoint(img_array, top_left, center, radius):
        # Use outer points of left line to infer its parameters
        left_slope, left_intercept, left_point_count = ImageUtilities.estimate_left_line_parameters(img_array,
                                                                                                    top_left, 1)

        # NaN can occur if not enough points were used in this frame. Check for this
        if np.isnan(left_slope) or np.isnan(left_intercept):
            raise ValueError("Too few points used in regression")

        left_line_endpt = (int(np.round(left_intercept)), 0)
        bottom_left = ImageUtilities.get_circle_line_intersection(center, radius, top_left, left_line_endpt)

        return bottom_left

    @staticmethod
    def get_bottom_right_endpoint(img_array, top_right, bottom_left, center, radius):
        # Right line
        right_slope, right_intercept, right_point_count = ImageUtilities.estimate_right_line_parameters(img_array,
                                                                                                        bottom_left,
                                                                                                        top_right, 1)

        # NaN can occur if not enough points were used in this frame. Check for this
        if np.isnan(right_slope) or np.isnan(right_intercept):
            raise ValueError("Too few points used in regression")

        right_line_endpt = (int(np.round(right_intercept + img_array.shape[1] * right_slope)), img_array.shape[1] - 1)
        bottom_right = ImageUtilities.get_circle_line_intersection(center, radius, top_right, right_line_endpt)

        return bottom_right

    @staticmethod
    def get_circle_line_intersection(center, radius, p1, p2):
        sl_center = Point(center[1], center[0])
        sl_circle = sl_center.buffer(radius).boundary
        sl_line = LineString([(p1[1], p1[0]), (p2[1], p2[0])])
        sl_intersect = sl_circle.intersection(sl_line)

        if sl_intersect.type == "LineString" and len(sl_intersect.coords) == 0:
            raise ArcEstimationError("No intersection points found")
        elif sl_intersect.type == "MultiPoint":
            raise ArcEstimationError("Too many intersections found")

        return (np.round(sl_intersect.coords[0][1]).astype(int), np.round(sl_intersect.coords[0][0]).astype(int))

    @staticmethod
    def get_static_points(img_array, threshold):
        top_left = (0, 0)
        bottom = (0, 0)
        top_right = (0, 0)

        # Scan from top -> bottom
        for r in range(img_array.shape[0]):
            # Scan from left -> right
            over_thresh = (img_array[r] > threshold).nonzero()
            if len(over_thresh[0]) > 0:
                top_left = (r, over_thresh[0][0])
                break

        # Scan from left -> right
        for c in range(img_array.shape[1]):
            # Scan from bottom to top
            over_thresh = (img_array.T[c] > 1).nonzero()
            if (len(over_thresh[0])) > 0:
                r = over_thresh[0][-1]
                if r > bottom[0]:
                    bottom = (r, c)

        # Scan left -> right from point 1 onwards, identify last point over threshold
        over_thresh = (img_array[top_left[0], top_left[1]:] > threshold).nonzero()

        # Look at the now determined top row, and determine if there is any text present in it. This is usually
        # evident if there are two separate areas with intensity > 1 in this row.
        text_present = np.max(np.diff(over_thresh[0])) > 1

        # If text is present, then scan the row again, this time only in the range of the 1st area
        # Use the threshold value and take the rightmost point
        if text_present:
            over_thresh = \
                (img_array[top_left[0],
                 top_left[1]:top_left[1] + over_thresh[0][np.argmax(np.diff(over_thresh[0]))]] > threshold).nonzero()

            top_right = (top_left[0], top_left[1] + over_thresh[0][-1])

        # If no text is present, take the rightmost point using the threshold value
        else:
            over_thresh = (img_array[top_left[0]] > threshold).nonzero()
            top_right = (top_left[0], over_thresh[0][-1])

        return top_left, bottom, top_right

    @staticmethod
    def get_arc(img_array, top_left, bottom, top_right):
        # Start at y coordinate of the bottom point and scan up, adding the encountered left and rightmost points as
        # on the arc of the cone
        arc_points = []
        last_left = last_right = np.round((top_right[1] + top_left[1]) / 2).astype(int)
        stop_left = stop_right = False
        for r in reversed(range(top_left[0], bottom[0])):
            over_thresh = (img_array[r] > 1).nonzero()[0]
            if len(over_thresh) == 0:
                continue

            left_pt = (r, over_thresh[0])
            right_pt = (r, over_thresh[-1])

            # Skip if all points found are on one side of center (this is an edge case at the very bottom of the video
            # due to artifacts, mostly)
            if (left_pt[1] < last_left and right_pt[1] < last_left) or (
                    left_pt[1] > last_right and right_pt[1] > last_right):
                continue
            # Break if both sides are stopped
            elif stop_left and stop_right:
                break

            if left_pt[1] <= last_left:
                arc_points.append(left_pt)
                last_left = left_pt[1]

            if right_pt[1] >= last_right:
                arc_points.append(right_pt)
                last_right = right_pt[1]

        # TODO: Try to improve this
        if len(arc_points) < 5:
            raise ValueError("Too few points")

        # From https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
        def estimate_circle(points_on_circle):
            from scipy import optimize
            x = np.array([i[1] for i in points_on_circle])
            y = np.array([i[0] for i in points_on_circle])

            def calc_R(xc, yc):
                """ calculate the distance of each 2D points from the center (xc, yc) """
                from math import sqrt
                return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

            def f_2(c):
                """ calculate the algebraic distance between the data points and \
                the mean circle centered at c=(xc, yc) """
                Ri = calc_R(*c)
                return Ri - np.mean(Ri)

            center_estimate = np.mean(x), np.mean(y)
            center_lsq, _ = optimize.leastsq(f_2, center_estimate)
            distances = calc_R(*center_lsq)
            radius = np.mean(distances)
            residuals = np.sum((distances - radius) ** 2) / len(points_on_circle)

            # Convert to image coordinates
            center_lsq = (center_lsq[1], center_lsq[0])
            return center_lsq, radius, residuals

        return (*estimate_circle(arc_points), len(arc_points))

    @staticmethod
    def fill_side_of_line(img_array, m, b):
        # TODO: Support specification of side, amount and colour
        new_img = np.copy(img_array)

        # Fill value either scalar of vector (RGB)
        fill_value = 0 if len(img_array.shape) == 2 else np.array([0, 0, 0])

        fill_until = int(0.2 * img_array.shape[1])

        for r in range(fill_until):
            c = int((b + 3) + r * (1 / m))
            for cx in range(c, img_array.shape[0]):
                new_img[r, cx] = fill_value

        return new_img

    @staticmethod
    def draw_echo_segmentation(img_array, point_1, point_2, point_3, point_4, point_5, point_6):
        if len(img_array.shape) != 3:
            colour_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            colour_image = img_array

        # Draw lines
        cv2.line(colour_image, (point_1[1], point_1[0]), (point_2[1], point_2[0]), (0, 255, 0), 2)
        cv2.line(colour_image, (0, point_3[0]), (img_array.shape[1], point_3[0]), (0, 255, 0), 2)
        cv2.line(colour_image, (point_4[1], point_4[0]), (point_5[1], point_5[0]), (0, 255, 0), 2)
        cv2.line(colour_image, (point_5[1], point_5[0]), (point_6[1], point_6[0]), (0, 255, 0), 2)

        # Draw points
        cv2.circle(colour_image, (point_1[1], point_1[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_2[1], point_2[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_3[1], point_3[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_4[1], point_4[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_5[1], point_5[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_6[1], point_6[0]), radius=3, color=(0, 0, 255), thickness=-1)

        return colour_image

    @staticmethod
    def draw_segmentation_points(img_array, point_1, point_2, point_3, point_4, point_5, point_6, cc, rad):
        if len(img_array.shape) != 3:
            colour_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            colour_image = img_array

        # Draw circle
        cv2.circle(colour_image, (int(np.round(cc[1])), int(np.round(cc[0]))), radius=int(np.round(rad)),
                   color=(255, 0, 0), thickness=3)
        cv2.circle(colour_image, (int(np.round(cc[1])), int(np.round(cc[0]))), radius=3, color=(255, 0, 0),
                   thickness=-1)

        # Draw points
        cv2.circle(colour_image, (point_1[1], point_1[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_2[1], point_2[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_3[1], point_3[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_4[1], point_4[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_5[1], point_5[0]), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(colour_image, (point_6[1], point_6[0]), radius=3, color=(0, 0, 255), thickness=-1)

        return colour_image

    @staticmethod
    def draw_points(img_array, points):
        if len(img_array.shape) != 3:
            colour_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            colour_image = img_array

        for i in range(len(points)):
            cv2.circle(colour_image, (points[i][1], points[i][0]), radius=3, color=(0, 0, 255), thickness=-1)

        return colour_image

    @staticmethod
    def detect_edges(img_array, canny_min, canny_max, blur):
        # Convert image to grayscale
        gs_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        if blur:
            blurred_image = cv2.GaussianBlur(gs_image, (7, 7), 0)

            # Do Canny edge detection
            canny = cv2.Canny(blurred_image, canny_min, canny_max)

        else:
            canny = cv2.Canny(img_array, canny_min, canny_max)

        # Contours
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            cv2.drawContours(img_array, c, -1, ImageUtilities.get_random_rgb_colour(), 2)

        return img_array

    @staticmethod
    def check_frame(frame, threshold=20):
        # Check left and rightmost columns, both should return true
        return ImageUtilities._check_column(frame.T[0], threshold) and ImageUtilities._check_column(frame.T[-1],
                                                                                                    threshold)

    @staticmethod
    def _check_column(column, threshold):
        return len((column > threshold).nonzero()[0]) <= 2

    @staticmethod
    def estimate_left_line_parameters(frame, point_1, threshold=10, debug=False):
        x = [point_1[1]]
        y = [point_1[0]]

        for r in range(point_1[0] + 1, frame.shape[0]):
            over_thresh = (frame[r] > threshold).nonzero()[0]

            if len(over_thresh) == 0:
                continue

            elif over_thresh[0] <= x[-1]:
                x.append(over_thresh[0])
                y.append(r)

        slope, intercept, _, _, _ = stats.linregress(x, y)

        if debug:
            ImageUtilities.show_image(ImageUtilities.draw_points(frame, list(zip(y, x))))

        return slope, intercept, len(x)

    @staticmethod
    def estimate_right_line_parameters(frame, point_2, point_6, threshold=10, debug=False):
        over_thresh = (frame[point_2[0]] > threshold).nonzero()[0]
        x = [over_thresh[-1]]
        y = [point_2[0]]

        for r in reversed(range(0, point_2[0])):
            over_thresh = (frame[r] > threshold).nonzero()[0]

            if len(over_thresh) == 0:
                continue

            elif over_thresh[-1] <= x[-1]:
                x.append(over_thresh[-1])
                y.append(r)

        x.append(point_6[1])
        y.append(point_6[0])

        slope, intercept, _, _, _ = stats.linregress(x, y)

        if debug:
            ImageUtilities.show_image(ImageUtilities.draw_points(frame, list(zip(y, x))))

        return slope, intercept, len(x)

    @staticmethod
    def _slope_intercept_with_y(x1, x2, y1, y2, intercept_y):
        # Note: x and y are reversed here, also we are using pixel space coordinates, so the slope numerically
        # goes in the other direction
        m = (x2 - x1) / (y2 - y1) * -1
        b = x1 - (intercept_y - y1) * m
        b = np.round(b).astype(int)

        return m, b

    @staticmethod
    def show_image(img_array):
        if len(img_array.shape) != 3:
            colour_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            colour_image = img_array

        cv2.imshow("img", colour_image)
        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image

    @staticmethod
    def crop_image(img_array, top, right, bottom, left, mode="proportion"):
        assert len(img_array.shape) in [2, 3, 4], "crop_image only supports 2d, 3d or 4d arrays"

        # Determine which axis is x and y
        if (len(img_array.shape) == 3 and (img_array.shape[2] != 3)) or (len(img_array.shape) == 4):
            y_axis = 1
            x_axis = 2
        else:
            y_axis = 0
            x_axis = 1

        # Determine indices of pixels to keep
        if mode == "proportion":
            top_index = int(top * img_array.shape[y_axis])
            bottom_index = int(img_array.shape[y_axis] - bottom * img_array.shape[y_axis])
            left_index = int(left * img_array.shape[x_axis])
            right_index = int(img_array.shape[x_axis] - right * img_array.shape[x_axis])
        elif mode == "exact":
            top_index = top
            bottom_index = bottom + 1
            left_index = left
            right_index = right + 1
        else:
            raise ValueError("Invalid mode {}".format(mode))

        # Return differently for 2d vs 3d array
        if len(img_array.shape) == 2:
            return img_array[top_index:bottom_index, left_index:right_index]
        elif len(img_array.shape) == 3:
            # RGB image
            if img_array.shape[2] == 3:
                return img_array[top_index:bottom_index, left_index:right_index, :]
            else:
                return img_array[:, top_index:bottom_index, left_index:right_index]
        else:
            return img_array[:, top_index:bottom_index, left_index:right_index, :]

    @staticmethod
    def get_random_rgb_colour():
        return np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)

    @staticmethod
    def compare_images_from_array(left, right):
        assert left.shape[0] == right.shape[0], "Arrays should have the same length"

        img_order = np.arange(left.shape[0])
        np.random.shuffle(img_order)

        for idx in img_order:
            combined_img = np.concatenate((left[idx], right[idx]))
            ImageUtilities.show_image(combined_img)

    @staticmethod
    def is_frame_top_empty(img_array):
        return len((img_array[0] > 40).nonzero()[0]) == 0
