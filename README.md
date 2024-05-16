>This repo is still under construction.

# Abstract

This project involves creating a lane line detector using traditional image processing methods, with OpenCV as the main module. The detector utilizes edge information to extract lines and incorporates a key algorithm involving line filtering and lane tracking processes. I have introduced the index of confidence, a value between 0 and 1, to regulate the tolerance of line filtering and lane tracking. The line filtering process initially utilizes an adaptively adjusted region of interest to eliminate most of the noise, then employs the angles of lines and the vanishing point to filter lines. The line tracking algorithm takes into account the width of the lane and the displacement of the lane lines to determine whether to update lane lines.

# Quick Start

First, pull the repo from the cloud, and then `cd AdaptiveLaneTracker-CV`.

## Virtual Environment (Optional)

Using Anaconda:

```bash
conda create --name laneTracking python=3.12
conda activate laneTracking
```

Or Python's `venv` module:

```bash
python -m venv .
.\Scripts\activate
```

## Install packages

Install all required packages by running:

```
pip install -r requirements.txt
```

## Execution

```bash
python .\src\main.py .\example_video.mp4
```

Replace `.\example_video.mp4` with any video you want to test.

# Disclaimer

This program is sensitive to frame rate and hence needs further parameter tuning for different videos.
The current parameters are tuned for the CULane dataset, which has a very low frame rate (they only pick out a fraction of recorded frames every second to reduce the size of the dataset, I guess). It works quite decently on `example_video.mp4`, which is a 24-FPS video, but it would be better if the reduction rate of confidence indices and the tolerance range were lower. This phenomenon is worse if you try a 60-FPS video unless you know how to adjust the parameters.