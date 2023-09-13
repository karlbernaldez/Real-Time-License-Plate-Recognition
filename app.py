import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from util import write_csv
import os, uuid
import matplotlib.pyplot as plt

# Initialize necessary variables and models
lp_folder_path = "./licenses_plates_imgs_detected/"
vehicle_folder_path = "./vehicles/"
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=False)
vehicles = {2: "Car", 3: "Motorcycle", 5: "Bus", 6: "Truck"}

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

class VideoProcessor:
    def recv(self, img):
        return img

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop)

    width = img.shape[1]
    height = img.shape[0]
    
    if detections == [] :
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

    plate = [] 

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > 0.17:
            bbox, text, score = result
            text = result[1]
            text = text.upper()
            scores += score
            plate.append(text)
    
    if len(plate) != 0 : 
        return " ".join(plate), scores/len(plate)
    else :
        return " ".join(plate), 0

def model_prediction(img, frame_number):
    license_numbers = 0
    results = {}
    licenses_texts = []

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    vehicle_detected = False
    vehicle_bboxes = []

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
                vehicle_detected = True
                vehicle_bboxes.append([xcar1, ycar1, xcar2, ycar2, car_score, vehicles[int(class_id)]])  # store class name instead of class id

        if vehicle_detected:
            license_detections = license_plate_detector(img)[0]
            
            if len(license_detections.boxes.cls.tolist()) != 0:
                license_plate_crops_total = []
                for license_plate in license_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    # Check if this license plate is inside any vehicle bounding box
                    for veh_bbox in vehicle_bboxes:
                        xcar1, ycar1, xcar2, ycar2, car_score, veh_class_id = veh_bbox
                        if x1 > xcar1 and x2 < xcar2 and y1 > ycar1 and y2 < ycar2:
                            # This license plate is inside this vehicle bounding box
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                            
                            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                            img_name = '{}.jpg'.format(uuid.uuid1())
                            cv2.imwrite(os.path.join(lp_folder_path, img_name), license_plate_crop)

                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                            licenses_texts.append(license_plate_text)

                            if license_plate_text is not None and license_plate_text_score is not None:
                                # Drawing the recognized license plate number on the image
                                cv2.rectangle(img, (int(x1), int(y1) - 40), (int(x2), int(y1)), (255, 255, 255), cv2.FILLED)
                                cv2.putText(img, str(license_plate_text), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                                # Adding class name and score to the preview
                                cv2.putText(img, f"Class: {veh_class_id}, Score: {round(car_score, 2)}", (int(x1), int(y1) - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                                # Adding license number score to the preview
                                cv2.putText(img, f"LP Score: {round(license_plate_text_score, 2)}", (int(x1), int(y1) - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                                license_plate_crops_total.append(license_plate_crop)
                                
                                # Save a cropped image of the car
                                car_crop = img[int(ycar1):int(ycar2), int(xcar1):int(xcar2), :]
                                car_img_name = f'{veh_class_id}{frame_number}_{license_numbers}.jpg'
                                cv2.imwrite(os.path.join(vehicle_folder_path, car_img_name), car_crop)

                                results[license_numbers] = {
                                                                license_numbers: {
                                                                    'car': {
                                                                        'bbox': [xcar1, ycar1, xcar2, ycar2], 
                                                                        'car_score': car_score,
                                                                        'vehicle_img': car_img_name,  # Add the path to the vehicle image
                                                                        'vehicle_class': veh_class_id  # Add the vehicle class
                                                                    },
                                                                    'license_plate': {
                                                                        'bbox': [x1, y1, x2, y2], 
                                                                        'text': license_plate_text, 
                                                                        'bbox_score': score, 
                                                                        'text_score': license_plate_text_score
                                                                    }
                                                                }
                                                            }
                                license_numbers += 1
                                write_csv(results, f"./results/detection_results.csv")
              
                img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return [img_wth_box, licenses_texts, license_plate_crops_total, results]
    else:
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

    img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [img_wth_box, [], [], {}]  # Include an empty dictionary for results



def main():
    cap = cv2.VideoCapture(0)
    processor = VideoProcessor()
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = processor.recv(frame)
        result = model_prediction(processed_frame, frame_number)
        processed_frame = result[0]

        # Increment frame number
        frame_number += 1

        # Display the processed frame
        plt.figure('Tech Titan - Realtime License Plate Recognition System')
        plt.imshow(processed_frame)
        plt.title('Tech Titan - Realtime License Plate Recognition System')
        plt.pause(0.01)  # Adjust the pause time as needed
        plt.clf()  # Clear the plot for the next frame
        
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
