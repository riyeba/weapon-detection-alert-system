import cv2
from ultralytics import YOLO
import yagmail
import torch



class WeaponVideoProcessor:
    """
    A class for processing video frames to detect weapons using YOLOv8 models.
    It loads a YOLOv8 model, detects weapons, annotates the frames, and sends email alerts.
    """

    def __init__(self, model_path: str, email: str, email_password: str) -> None:
        """
        Initialize with model path and email credentials.
        """
        self.model_path = model_path
        self.email = email
        self.email_password = email_password
        self.model = YOLO(self.model_path)  

    def send_email(self, to_email: str, object_detected: str) -> None:
        """
        Sends an email alert when a weapon is detected.
        """
        yag = yagmail.SMTP(self.email, self.email_password)
        yag.send(
            to="taakinpennu@gmail.com",
            subject="Security Alert",
            contents=f'ALERT - {object_detected} object(s) detected!'
        )
        print("Email sent successfully!")

    def process_video(self, video_source: str) -> None:
        """
        Detects weapons from video stream and displays results with annotations.
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame or the analysis has ended")
                break

            results = self.model.track(frame, persist=True, verbose=False, conf=0.54)
            detected_objects = []

            if results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu()
                class_ids = results[0].boxes.cls.cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()
                names = self.model.names

                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    if confidence >= 0.2:
                        detected_object = names[int(class_id)]
                        detected_objects.append(detected_object)

                        label = f"{detected_object} {confidence:.2f}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                      (x1 + label_size[0], y1 + base_line - 10), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, label, (x1, y1 - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        cv2.putText(frame, "Weapon Detected! Email Sent", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        self.send_email(self.email, label)
           
            cv2.imshow("Gun Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "modelweapon.pt"  
    email = "nsuksu92@gmail.com"
    email_password = "nnoqmllamzoqvpsl"

 
    video_source = 0

    processor = WeaponVideoProcessor(model_path, email, email_password)
    processor.process_video(video_source)
