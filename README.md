# Weapon Detection and Alert System

This repository contains a YOLOv8-powered weapon detection project tailored for video-based security and surveillance applications.

The system focuses on:

- **Video Weapon Detection**: Processes video streams (such as CCTV footage or live webcam feeds) to detect weapons in real-time. When a weapon is detected, bounding boxes are drawn around the object, and an instant email alert is sent to notify security personnel.  
- **Image Weapon Detection**: Processes individual images to detect and blur weapons. An email alert is also sent when a weapon is detected, making it useful for still photos or frames from recorded footage.

## Demo

Here is an example output of the system:

<img src="https://github.com/riyeba/weapon-detection-alert-system/blob/main/detected_weapon.png?raw=true" alt="Detected Weapon" width="600">  


![Email Alert](https://github.com/riyeba/weapon-detection-alert-system/blob/main/message_receivedd.png?raw=true)  


## Features

- Real-time detection of weapons in video streams  
- Detection of multiple weapon types: guns, pistols, rifles, knives, grenades  
- Sends instant email alerts when a weapon is detected  
- Draws bounding boxes around detected weapons for visualization  
- Blurs detected weapons in images for privacy and analysis  

## Applications

- ✅ Surveillance systems  
- ✅ Security monitoring  
- ✅ Restricted area monitoring  
- ✅ Public safety  
- ✅ Automated security alerts  
- ✅ Law enforcement tools  
- ✅ Retail security systems  
- ✅ Workplace safety monitoring  
- ✅ Event security management  
- ✅ Access control systems  

## Requirements

- Python 3.7+  
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- [Roboflow](https://roboflow.com/)  
- [yagmail](https://github.com/kootenpv/yagmail)  
- [OpenCV-Python](https://pypi.org/project/opencv-python/)  
- [NumPy](https://numpy.org/)  

## Installation

Install the required packages using:

```bash
pip install ultralytics opencv-python yagmail numpy
```

## Usage

### 1. Video Weapon Detection

The `video.py` script processes video streams, detects weapons, and sends email alerts.

#### Steps:

1. **Update the model path:**  
   Edit the `model_path` variable in the script to point to your YOLOv8 weights file (e.g., `modelweapon.pt`).

2. **Set up email alerts:**  
   Update the `email` and `email_password` variables in the script with your email address and app password.

3. **Run the script:**  
   ```bash
   python video.py




