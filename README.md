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
   ```

   ### Example code : Video Weapon Detection





```python
import cv2
from ultralytics import YOLO
import yagmail
import torch

class WeaponVideoProcessor:
  # code for processing video

```
## Email Notification Setup

Both scripts (`video.py` and `image.py`) use the `yagmail` library to send email alerts when a weapon is detected.

### Steps to Set Up:

1. **Create an email account with an app-specific password:**  
   - If you are using Gmail, generate an [app password](https://support.google.com/accounts/answer/185833) for your account.  
   - Other email providers may have similar app-password features or require enabling SMTP access.

2. **Update the scripts with your email credentials:**  
   - In both scripts, set the following variables:  
     ```python
     email = "your_email@gmail.com"
     email_password = "your_app_password"
     to_email = "recipient_email@example.com"
     ```

3. **Verify email sending:**  
   - Run the script on a test video or image to ensure that an email is received whenever a weapon is detected.

### Notes:

- Using an app-specific password is **recommended** for security.  
- Ensure that your email provider allows SMTP access for sending emails.  
- The `yagmail` library simplifies sending emails without manually configuring SMTP settings.



## Model

This project uses a **YOLOv8 model trained for weapon detection**. The model can detect multiple types of weapons, including guns, pistols, rifles, knives, and grenades.

### Details:

- The model is trained on a weapon detection dataset (using your own data or datasets from Roboflow).  
- YOLOv8 provides real-time detection, making it suitable for video streams from CCTV or live webcams.  
- Both the `video.py` and `image.py` scripts load this pre-trained model for inference.

### Using Your Model:

1. Place your trained weights file (`modelweapon.pt`) in an accessible folder.  
2. Update the `model_path` variable in the scripts to point to this file:  
   ```python
   model = YOLO('path/to/modelweapon.pt')

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for providing the YOLOv8 detection framework.  
- [Roboflow](https://roboflow.com/) for assisting with dataset management and annotation.  
- [Yagmail](https://github.com/kootenpv/yagmail) for handling email notifications.  
- Open-source Python community for libraries like OpenCV and NumPy that make real-time video and image processing possible.  
- All contributors and researchers whose work inspired and supported this project.  



