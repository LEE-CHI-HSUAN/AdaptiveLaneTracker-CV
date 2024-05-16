import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from rich import traceback
traceback.install(show_locals=True)


def pltImage(img: np.ndarray, color: str = None) -> None:
    '''
    Show the image in Jupyter Notebook interface.
    '''
    pic = img.copy() if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16,4))
    plt.imshow(pic, cmap=color)


def line_to_points(slope: float, intercept: float, y_start: int, y_end: int, y_interval: int = 10, x_only = True):
    '''
    Given both endpoints and an interval with respect to the y-axis,
    return a line represented by a series of points.

    Input: an array `[x1, y1, x2, y2]`.
    Return: a list of 2d points `[(x1,y1), (), (), ...]`.
    '''
    y_start = (y_start // y_interval) * y_interval
    y_end = (y_end // y_interval) * y_interval - 1 # -1 makes the range inclusive
    y_interval = -np.abs(y_interval) if y_start > y_end else np.abs(y_interval)
    y_array = np.arange(y_start, y_end, y_interval)
    x_array = (y_array - intercept) / slope
    
    if x_only:
        return x_array
    return np.array(list(zip(x_array, y_array)))


def drawLines(img: np.ndarray, lines: list[np.ndarray], color=(0,0,255)) -> None:
    for line in lines:
        if line is None: continue
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)


def readLines(filename: str):
    '''
    A generator that returns a list of 2D point of a lane line

    Input: file path of the CULane-formated lane line annotations for a frame
    '''
    with open(filename) as fin:
        for line in fin:
            numbers = line.split() # a list of strings (digits)
            yield [(int(float(x)), int(y)) for x, y in zip(numbers[::2], numbers[1::2])]


def readPredictions(filename: str):
    '''
    A generator that returns 2 tuples representing left and right lane line respectively.

    Input: file path of the CULane-formated lane line annotations for a frame
    '''
    with open(filename) as fin:
        for line in fin:
            parameters = line.split() # left_slope left_intercept right_slope right_intercept
            yield (float(parameters[0]), float(parameters[1])), (float(parameters[2]), float(parameters[3]))


def getFrames(videoFolder: str) -> list[str]:
    '''
    Return a list of .jpg file names in a MP4 folder.
    not sorted, cuz I currently assume they are already sorted.
    '''
    return [fname for fname in os.listdir(videoFolder) if fname[-1] == 'g']


def readImage(path: str, annotation: bool=False) -> np.ndarray:
    '''
    Use OpenCV to read a image; adding annotation is optional.
    `annotation`: whether to draw ground truth lane lines.
    '''
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if annotation:
        for line in readLines(str(path)[:-4] + ".lines.txt"):
            for point in line:
                cv2.circle(img, point, 4, (0, 255, 0))
    return img


def playMP4(videoFolder: str, annotation: bool=False) -> None:
    '''
    Taylor made for CULane dataset.
    Read a MP4 DIR, and then open a cv2 window to play the given video.
    '''
    for frameName in getFrames(videoFolder):
        img = readImage(videoFolder + "\\" + frameName, True)
        cv2.imshow("mp4 player", img)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
