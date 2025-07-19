import cv2
from ultralytics import YOLO
import yagmail
import torch


class WeaponImageProcessor:
   
    def __init__(self) -> None:
        self.to_email = "takinpenu@gmail.com"
       
        self.sender_email = "nsuksu92@gmail.com"
        self.app_password = "nnoqmllamzoqvpsl"  

    def send_email(self, to_email: str, object_detected: str) -> None:
        
        yag = yagmail.SMTP(self.sender_email, self.app_password)
        yag.send(
            to=to_email,
            subject="Security Alert",
            contents=f"ALERT - {object_detected} object has been detected!"
        )
        print("Email sent successfully!")

    def load_yolo_model(self, model_path: str) -> YOLO:
        self.model = YOLO(model_path)
        return self.model

    def process_image(self, image_path, model_path) -> None:
       
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Could not read image {image_path}.")
            return

        model = self.load_yolo_model(model_path)

        results = model.track(frame, persist=True, verbose=True)

        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            scores = results[0].boxes.conf.cpu().tolist()

            for box, cls, score in zip(boxes, clss, scores):
                if score >= 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    detected_object = model.names[int(cls)]
                    label = f"{detected_object} {score:.2f}"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1 + base_line - 10), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Email notification
                    self.send_email(self.to_email, detected_object)

                    # Alert message on image
                    text = "Weapon Detected and Email Sent"
                    cv2.putText(frame, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show the output
        cv2.imshow("detected object:", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path ="aleksey-kashmar-rxoKm1qeJ78-unsplash.jpg"
    model_path="modelweapon.pt" 
    processor = WeaponImageProcessor()
    processor.process_image(image_path,model_path)