import cv2
import numpy as np

class LaneDetector:
    def __init__(self,
                 height:int = 590,
                 width:int = 1640,
                 ROI: np.ndarray = None,
                 n_initialize: int = 10,
                 ref_height: int = 400,
                 min_line_lane: int = 40,
                 lane_width: int = 400) -> None:
        self.height = height
        self.width = width
        self.n_initialize = n_initialize
        self.ref_height = ref_height
        self.min_line_len = min_line_lane # shorter than this will incur reduction in confidence
        self.lane_width = lane_width

        self.n = 0 # just a counter
        self.angular_resolution = np.pi / 180 # 1 degree
        self.angle_range = (20, 75) # filter
        self.l_lane_para = None # slope, intercept
        self.r_lane_para = None # slope, intercept
        self.left_confidence = 0.5
        self.right_confidence = 0.5
        self.VP = (self.width // 2, 300) # vanish point

        # create ROI mask
        self.base_ROI = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self.base_ROI, ROI, 255)
        self.ROI_mask = self.base_ROI.copy()

    def _updateROI(self, upper_y) -> None:
        if self.r_lane_para is None or self.l_lane_para is None:
            return
        left_lane, right_lane = self.extendLine(self.l_lane_para, upper_y), self.extendLine(self.r_lane_para, upper_y)
        # erase old ROI mask
        self.ROI_mask[:,:] = 0
        # draw new ROI mask
        l_width = np.min([int(self.width * 0.03 / self.left_confidence),  self.width])
        r_width = np.min([int(self.width * 0.03 / self.right_confidence), self.width])
        cv2.line(self.ROI_mask, (left_lane[0],  left_lane[1]),  (left_lane[2],  left_lane[3]),  255, l_width)
        cv2.line(self.ROI_mask, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), 255, r_width)
        self.ROI_mask = cv2.bitwise_and(self.base_ROI, self.ROI_mask)

        # left_x = (600 - self.l_lane_para[1]) / self.l_lane_para[0]
        # right_x = (600 - self.r_lane_para[1]) / self.r_lane_para[0]
        # left_lower_range = int(self.width * 0.03 / self.left_confidence)
        # left_upper_range = left_lower_range // 3
        # right_lower_range = int(self.width * 0.03 / self.right_confidence)
        # right_upper_range = right_lower_range // 3
        # polygons = np.array([[(self.VP[0] + left_upper_range, self.VP[1]),
        #                       (self.VP[0] - left_upper_range, self.VP[1]),
        #                       (left_x - left_lower_range, 600),
        #                       (left_x + left_lower_range, 600)],
        #                      [(self.VP[0] + right_upper_range, self.VP[1]),
        #                       (self.VP[0] - right_upper_range, self.VP[1]),
        #                       (right_x - right_lower_range, 600),
        #                       (right_x + right_lower_range, 600)],]).astype(np.int32)
        # self.ROI_mask[:,:] = 0
        # cv2.fillPoly(self.ROI_mask, polygons, 255)
        # self.ROI_mask = cv2.bitwise_and(self.base_ROI, self.ROI_mask)

    def _updateVP(self) -> None:
        if self.r_lane_para is None or self.l_lane_para is None:
            return
        x = (self.r_lane_para[1] - self.l_lane_para[1]) / (self.l_lane_para[0] - self.r_lane_para[0])
        y = self.l_lane_para[0] * x + self.l_lane_para[1]
        self.VP = (int((x + self.VP[0]) / 2), int((y + self.VP[1]) / 2))

    def _VP_distance(self, slope, intercept):
        '''
        Return the shorted distance between the VP and the given line.
        '''
        return (slope * self.VP[0] - self.VP[1] + intercept) / np.sqrt(slope * slope + 1)

    def _filterLines(self, lines: np.ndarray) -> tuple[np.ndarray]:
        '''
        Differenciate possible left lane line and right lane line from a list of lines.
        '''
        left_candidates  = []
        right_candidates = []
        for line in lines:
            if -self.angle_range[1] <= self._getAngle(line) <= -self.angle_range[0]:
                s, i = self._getSlopeIntercept(line)
                if self.VP[0] is not None and self._VP_distance(s, i) > (20 / self.left_confidence):
                    continue
                left_candidates.append(line)
            elif self.angle_range[0] <= self._getAngle(line) <= self.angle_range[1]:
                s, i = self._getSlopeIntercept(line)
                if self.VP[0] is not None and self._VP_distance(s, i) > (20 / self.right_confidence):
                    continue
                right_candidates.append(line)
        left_candidates  = np.array(left_candidates)
        right_candidates = np.array(right_candidates)
        
        left_lane  = self._getLongestLine(left_candidates)
        right_lane = self._getLongestLine(right_candidates)
        return left_lane, right_lane
    
    def _tracking(self, left_lane: np.ndarray | None, right_lane: np.ndarray | None) -> None:
        '''
        # Lane mark tracking algorithm
        deciding the position of next lane lines base on confidence.

        Input: potential left and right lane.
        '''
        if left_lane is not None:
            left_lane_para = self._getSlopeIntercept(left_lane)
        else:
            left_lane_para = self.l_lane_para
        if right_lane is not None:
            right_lane_para = self._getSlopeIntercept(right_lane)
        else:
            right_lane_para = self.r_lane_para

        # angle filter 2.0 (legacy)
        # if self.n > self.n_initialize:
        #     angle_between = self._angle_between(left_lane_para[0], right_lane_para[0])
        #     prev_angle_between = self._angle_between(self.l_lane_para[0], self.r_lane_para[0])
        #     confidence = (self.left_confidence + self.right_confidence) / 2
        #     ratio = 1.3 / confidence
        #     if not ((1 / ratio) < (angle_between / prev_angle_between) < ratio):
        #         left_lane_para = self.l_lane_para
        #         right_lane_para = self.r_lane_para
        #         self.left_confidence *= 0.7
        #         self.right_confidence *= 0.7
        #         return
        
        # distance filter
        l_change, r_change = True, True
        if self.n > self.n_initialize:
            left_x = (self.ref_height - left_lane_para[1]) / left_lane_para[0]
            right_x = (self.ref_height - right_lane_para[1]) / right_lane_para[0]
            prev_left_x = (self.ref_height - self.l_lane_para[1]) / self.l_lane_para[0]
            prev_right_x = (self.ref_height - self.r_lane_para[1]) / self.r_lane_para[0]
            prev_width = prev_right_x - prev_left_x

            ratio = 1 + 0.2 / self.left_confidence
            #print("left change:", np.abs((left_x - prev_right_x) / prev_width))
            if not ((1 / ratio) < (np.abs((left_x - prev_right_x) / prev_width)) < ratio):
                left_lane_para = self.l_lane_para
                l_change = False
                self.left_confidence *= 0.9
            ratio = 1 + 0.2 / self.right_confidence
            if not ((1 / ratio) < (np.abs((right_x - prev_left_x) / prev_width)) < ratio):
                right_lane_para = self.r_lane_para
                r_change = False
                self.right_confidence *= 0.9

            # confidence = np.sqrt(self.left_confidence * self.right_confidence)
            # ratio = 1.1 / confidence
            # if not ((1 / ratio) < (np.abs((left_x - right_x) / prev_width)) < ratio):
            #     left_lane_para = self.l_lane_para
            #     right_lane_para = self.r_lane_para
            #     self.left_confidence *= 0.9
            #     self.right_confidence *= 0.9
            #     return

        # update confidence
        if left_lane is None:
            self.left_confidence *= 0.7
        elif l_change:
            line_length = self._getLength(left_lane)
            scaler = np.tanh((line_length - self.min_line_len) / self.min_line_len)
            if line_length > self.min_line_len:
                self.left_confidence += (1 - self.left_confidence) * scaler
            else:
                self.left_confidence += self.left_confidence * scaler / 5
            pos_variation = self._getWidth(self.l_lane_para, left_lane_para)
            # print("left pos var", pos_variation)
            if pos_variation > 40:
                self.left_confidence *= np.exp((40 - pos_variation) / 40)
        if right_lane is None:
            self.right_confidence *= 0.7
        elif r_change:
            line_length = self._getLength(right_lane)
            #print("right length: ", line_length)
            scaler = np.tanh((line_length - self.min_line_len) / self.min_line_len)
            if line_length > self.min_line_len:
                self.right_confidence += (1 - self.right_confidence) * scaler
            else:
                self.right_confidence += self.right_confidence * scaler / 5
            pos_variation = self._getWidth(self.r_lane_para, right_lane_para)
            # print("right pos var", pos_variation)
            if pos_variation > 40:
                self.right_confidence *= np.exp((40 - pos_variation) / 40)
        
        if left_lane_para is not None and right_lane_para is not None:
            lane_width = self._getWidth(left_lane_para,right_lane_para)
            if (lane_width > 400):
                self.left_confidence *= np.exp((400 - lane_width) / 400)
                self.right_confidence *= np.exp((400 - lane_width) / 400)
            if (lane_width < 250):
                self.left_confidence *= np.exp((lane_width - 250) / 400)
                self.right_confidence *= np.exp((lane_width - 250) / 400)

        self.l_lane_para = left_lane_para
        self.r_lane_para = right_lane_para

    def _getWidth(self, lineA_para: tuple[float, float] | None, lineB_para: tuple[float, float] | None) -> float | None:
        '''
        Compute the distance between two lines at the reference height.
        '''
        if lineA_para is None or lineB_para is None:
            return 0
        A_x = (self.ref_height - lineA_para[1]) / lineA_para[0]
        B_x = (self.ref_height - lineB_para[1]) / lineB_para[0]
        return np.abs(A_x - B_x)

    def detect(self, img: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Input: an image
        Return: detected lanes, detected lines (raw)
        '''
        #===========================#
        # Image Processing Pipeline #
        #===========================#
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 1.8)
        #img = cv2.bilateralFilter(img, 7, 50, 100)
        median = np.median(img)
        self.canny_high_thrd = int(median)
        edge_img = cv2.Canny(img, self.canny_high_thrd // 3, self.canny_high_thrd)
        edge_img = cv2.bitwise_and(self.ROI_mask, edge_img)
        debug_img = edge_img.copy()
        cv2.putText(debug_img, f"{self.left_confidence:.2f}, {self.right_confidence:.2f}", (600, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow("Edge", debug_img)

        #=================#
        # Line extraction #
        #=================#
        lines = cv2.HoughLinesP(edge_img, 1, self.angular_resolution, 30, None, 10, 5)
        lines = lines.squeeze(1) if lines is not None else [] # removes a redundant dimension
        raw_lines = self._filterLines(lines)

        #===============#
        # Lane Tracking #
        #===============#
        self._tracking(*raw_lines)
        self._updateVP()
        upper_y = 300 if self.VP is None else self.VP[1]
        self._updateROI(upper_y)

        #=======================#
        # Prepare Return Values #
        #=======================#
        lane_lines = (self.extendLine(para, upper_y) for para in [self.l_lane_para, self.r_lane_para])
        lane_lines = [line for line in lane_lines if line is not None]
        self.n += 1
        return lane_lines, raw_lines

    def getLaneFeatures(self) -> tuple | None:
        '''
        [Machine learning]
        return prev_l_slope, prev_l_x, prev_r_slope, prev_r_x, curr_slope, curr_x, curr_length.
        '''
        if self.l_lane_para is None or self.r_lane_para is None:
            return None
        prev_l_slope = self.l_lane_para[0]
        prev_l_x = (self.ref_height - self.l_lane_para[1]) / prev_l_slope
        prev_r_slope = self.r_lane_para[0]
        prev_r_x = (self.ref_height - self.r_lane_para[1]) / prev_r_slope
        return prev_l_slope, prev_l_x, prev_r_slope, prev_r_x
    
    def genData(self, img: np.ndarray):
        '''
        stay tuned...
        '''
        #===========================#
        # image processing pipeline #
        #===========================#
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bilateralFilter(img, 7, 50, 100)
        median = np.median(img)
        self.canny_high_thrd = int(median)
        edge_img = cv2.Canny(img, self.canny_high_thrd // 3, self.canny_high_thrd)
        edge_img = cv2.bitwise_and(self.ROI_mask, edge_img)
        debug_img = edge_img.copy()
        cv2.putText(debug_img, f"{self.left_confidence:.2f}, {self.right_confidence:.2f}", (600, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow("Edge", debug_img)

        lines = cv2.HoughLinesP(edge_img, 1, self.angular_resolution, 30, None, 10, 10)
        lines = lines.squeeze(1) if lines is not None else [] # removes a redundant dimension
        left_lane, right_lane = self._filterLines(lines)
        org_lines = [left_lane, right_lane]
        prev_lanes = [self.l_lane_para, self.r_lane_para]
        prev_lanes_features = self.getLaneFeatures()
        
        self._tracking(left_lane, right_lane)
        self._updateVP()

        upper_y = 300 if self.VP is None else self.VP[1]
        left_lane, right_lane = self.extendLine(self.l_lane_para, upper_y), self.extendLine(self.r_lane_para, upper_y)
        lane_lines = []
        if left_lane is not None:
            lane_lines.append(left_lane)
        if right_lane is not None:
            lane_lines.append(right_lane)

        self.n += 1
        self._updateROI(upper_y)

        left_lane, right_lane = self.extendLine(prev_lanes[0], upper_y), self.extendLine(prev_lanes[1], upper_y)
        lane_lines = []
        if left_lane is not None:
            lane_lines.append(left_lane)
        if right_lane is not None:
            lane_lines.append(right_lane)

        return lane_lines, org_lines, prev_lanes_features
    
    def extendLine(self, line_para: tuple, upper_y: int, lower_y: int | None = None) -> np.ndarray | None:
        '''
        Given a upper bound, return a line.
        The Lower bound, if not provided, will be set to the bottom.

        Input: `line_para`: `(slope, intercept)`
        Return: an array `[x1, y1, x2, y2]`.
        '''
        if line_para is None:
            return None
        if lower_y is None:
            lower_y = self.height
        slope, intercept, = line_para[0], line_para[1]
        y1 = upper_y
        y2 = self.height
        x1 = (y1 - intercept) / slope
        x2 = (y2 - intercept) / slope
        return np.array([x1, y1, x2, y2]).astype(np.int32)
    
###########################################################
# Below are some functions computing properties of a line #
###########################################################

    @staticmethod
    def _getAngle(line: np.ndarray) -> float:
        angle_rad = np.arctan2(line[3] - line[1], line[2] - line[0])
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    @staticmethod
    def _getLength(line: np.ndarray) -> float:
        vec = line[:2] - line[2:]
        return np.linalg.norm(vec)
    
    @staticmethod
    def _getSlopeIntercept(line: list | np.ndarray) -> tuple[float, float]:
        slope = (line[3] - line[1]) / (line[2] - line[0])
        intercept = line[1] - slope * line[0]
        return slope, intercept    

    @staticmethod
    def _getLongestLine(lines: list[np.ndarray]) -> np.ndarray | None:
        maxLength = 0
        longestLine = None
        for line in lines:
            l = LaneDetector._getLength(line)
            if l > maxLength:
                maxLength = l
                longestLine = line
        return longestLine

    @staticmethod
    def _angle_between(slope_A: float, slope_B: float) -> float:
        return np.degrees(np.arctan2(slope_A - slope_B, 1 + slope_A * slope_B))