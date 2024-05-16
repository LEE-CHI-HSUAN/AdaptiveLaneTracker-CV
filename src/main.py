import cv2
import sys
import time
import numpy as np
from pathlib import Path

import tools
from LaneDetector import LaneDetector


def CUL_predict_one_video() -> None:
    help = '''CUL_predict_one_video() -> None
    Taylor made for CULane dataset.
    usage: `python ./main.py <frame DIR>`
    example: `python ./main.py ./CULane/driver_193_90frame/driver_193_90frame/06042010_0511.MP4`
    '''
    if sys.argv[1] is None:
        print("Please provide video path.")
        exit(0)
    videoFolder = Path(sys.argv[1])
    
    height = 590
    width = 1640
    polygon = np.array([[(width // 2, int(height * 0.4)), (0, 460), (540, 460), (820, 410), (1000, 415), (1250, 455), (width, 550)]])
    detector = LaneDetector(ROI=polygon)
    fout = open(videoFolder.name + ".prediction.txt", "w")

    for frameName in tools.getFrames(str(videoFolder)):
        img = cv2.imread(str(videoFolder / frameName))
        lanelines, org_lines = detector.detect(img)

        tools.drawLines(img, lanelines)
        tools.drawLines(img, org_lines, color=(255,0,0))
        if detector.VP[0] is not None:
            cv2.circle(img, detector.VP, 10, (255,255,0), 2)
        cv2.imshow("mp4 player", img)

        fout.write(f"{detector.l_lane_para[0]} {detector.l_lane_para[1]} {detector.r_lane_para[0]} {detector.r_lane_para[1]}\n")
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break


def CUL_predict_a_list() -> None:
    help = '''CUL_predict_a_list() -> None
    Taylor made for CULane dataset.
    Predict a list of video.
    usage: `python ./main.py <text file> <DIR storing output>`
    example: `python ./main.py ./list.txt ./prediction`

    The text file contains a list of DIR, looking like:
    ./CULane/driver_23_30frame/driver_23_30frame/05151640_0419.MP4
    ./CULane/driver_23_30frame/driver_23_30frame/05151643_0420.MP4
    ...
    '''
    if len(sys.argv) < 3:
        print(help)
        exit(0)
    outputDir = Path(sys.argv[2])
    videoDirs = []
    with open(sys.argv[1], "r") as fin:
        videoDirs= [Path(line.strip()) for line in fin]

    height = 590
    width = 1640
    polygon = np.array([[(width // 2, int(height * 0.4)), (0, 460), (540, 460), (820, 410), (1000, 415), (1250, 455), (width, 550)]])
    detector = LaneDetector(ROI=polygon)

    for videoDir in videoDirs:
        fout = open(str(outputDir / (videoDir.name + ".prediction.txt")), "w")
        for frameName in tools.getFrames(str(videoDir)):
            img = cv2.imread(str(videoDir / frameName))
            lanelines, org_lines = detector.detect(img)
            if detector.r_lane_para is None or detector.l_lane_para is None:
                fout.write("0.01 0 0.01 0\n")
            else:
                fout.write(f"{detector.l_lane_para[0]} {detector.l_lane_para[1]} {detector.r_lane_para[0]} {detector.r_lane_para[1]}\n")
            
            tools.drawLines(img, lanelines)
            tools.drawLines(img, org_lines, color=(255,0,0))
            if detector.VP[0] is not None:
                cv2.circle(img, detector.VP, 10, (255,255,0), 2)
            cv2.imshow("Detecting a list of video", img)

            if cv2.waitKey() & 0xFF == ord('q'):
                break
        fout.close()


def predict_video(auto_play: bool = True) -> None:
    help = '''predict_video(auto_play: bool = True) -> None
    Usable for general video files that OpenCV can handle.
    usage: `python ./main.py <video name>`
    example: `python ./main.py ./video.mp4`

    ## Parameter:
    `auto_play`: if set to `False`, press any button to go to the next frame.
    '''
    if sys.argv[1] is None:
        print(help)
        exit(0)
    
    video = cv2.VideoCapture(sys.argv[1])
    mspf = int(1000 / video.get(cv2.CAP_PROP_FPS)) if auto_play else 0
    _, img = video.read()
    height, width = img.shape[:2]
    polygon = np.array([[(width // 2, height // 3), (0, 670), (width, 670)]])    
    detector = LaneDetector(720, 1280, ROI=polygon, ref_height=500)

    while True:
        ret, img = video.read()
        if not ret:
            return
        
        lanelines, org_lines = detector.detect(img)
        tools.drawLines(img, lanelines)
        tools.drawLines(img, org_lines, color=(255,0,0))
        if detector.VP[0] is not None:
            cv2.circle(img, detector.VP, 10, (255,255,0), 2)
        cv2.imshow("mp4 player", img)

        if cv2.waitKey(mspf) & 0xFF == ord('q'):
            break


def estimate_frame_rate():
    if sys.argv[1] is None:
        print("Please provide video path.")
        exit(0)
    
    video = cv2.VideoCapture(sys.argv[1])
    height = 720
    width = 1280
    polygon = np.array([[(width // 2, height // 2), (0, 670), (width, 670)]])    
    detector = LaneDetector(720, 1280, ROI=polygon, ref_height=500)

    start_time = time.time()
    n_frames = 0
    while True:
        ret, img = video.read()
        if not ret:
            break
        
        detector.detect(img)
        n_frames += 1
    
    period = time.time() - start_time
    print(f"{period = }s, {1000 * period / n_frames}ms/frame")
    return 


if __name__ == "__main__":
    # predict_video(auto_play=True)
    CUL_predict_a_list()

'''
night, illumination: CULane/driver_193_90frame/driver_193_90frame/06042010_0511.MP4
'''