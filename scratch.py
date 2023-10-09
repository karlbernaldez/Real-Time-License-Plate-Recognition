import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from util import write_csv, char2int, int2char
import os
import time
import re
import torch
import asyncio

# Constants
LP_FOLDER_PATH = "./licenses_plates_imgs_detected/"
VEHICLE_FOLDER_PATH = "./vehicles/"
LICENSE_MODEL_DETECTION_DIR = './models/best.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
THRESHOLD = 0.15
VEHICLES = {2: "Car", 3: "MC", 5: "Bus", 6: "Truck"}

# Initialize models and devices
reader = easyocr.Reader(['en'], gpu=True)
coco_model = YOLO(COCO_MODEL_DIR).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
results_buffer = []

class VideoProcessor:
    async def recv(self, img):
        return img

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.readtext(license_plate_crop, allowlist= '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

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

def format_license_plate_text(license_plate_text, class_name):
    """
    Formats the license plate text based on vehicle class.
    
    Args:
    - license_plate_text (str): The recognized license plate text.
    - class_name (str): The class of the vehicle.

    Returns:
    - str: Formatted license plate text.
    """
    lp_length = len(license_plate_text)

    if lp_length >= 6 and lp_length <= 7:
        if class_name in ["Car", "Bus", "Truck"]:
            l = license_plate_text[:3]
            n = license_plate_text[3:lp_length]

            # Check if 'l' contains a number
            if any(char.isdigit() for char in l):
                l = int2char(l)
            
            # Check if 'n' contains a letter
            if any(char.isalpha() for char in n):
                n = char2int(n)

            license_plate_text = l + "-" + n

        elif class_name == "MC" and lp_length == 6:
            n = license_plate_text[:3]
            l = license_plate_text[3:6]

            # Check if 'n' contains a number
            if any(char.isalpha() for char in n):
                n = char2int(n)

            # Check if 'l' contains a letter
            if any(char.isdigit() for char in l):
                l = int2char(l)

            license_plate_text = n + "-" + l

    return license_plate_text

def detect_objects_and_license_plates(img, frame_number):
    license_numbers = 0
    results = {}
    licenses_texts = []
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model.track(img, persist=True)[0]
    vehicle_detected = False
    vehicle_bboxes = []

    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            if len(detection) == 7:
                xvehicle1, yvehicle1, xvehicle2, yvehicle2, track_id, vehicle_score, class_id = detection

                if int(class_id) in VEHICLES:
                    class_name = VEHICLES[int(class_id)]
                    label = f"{class_name}-{int(track_id)} Score:{round(vehicle_score, 2)}"
                    cv2.rectangle(img, (int(xvehicle1), int(yvehicle1)), (int(xvehicle2), int(yvehicle2)), (0, 0, 255), 3)
                    cv2.putText(img, label, (int(xvehicle1), int(yvehicle1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    vehicle_detected = True
                    start_time = time.time()
                    vehicle_bboxes.append([track_id, xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_score, class_name])

        if vehicle_detected:
            license_detections = license_plate_detector.track(img, persist=True)[0]
            if len(license_detections.boxes.cls.tolist()) != 0:
                for license_plate in license_detections.boxes.data.tolist():
                    if len(license_plate) == 7:
                        x1, y1, x2, y2, lp_track_id, lp_score, class_name = license_plate

                        for veh_bbox in vehicle_bboxes:
                            track_id, xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_score, veh_class_name = veh_bbox
                            # Check if this license plate is inside any vehicle bounding box
                            if x1 > xvehicle1 and x2 < xvehicle2 and y1 > yvehicle1 and y2 < yvehicle2:
                                # This license plate is inside this vehicle bounding box
                                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            
                                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)
                                licenses_texts.append(license_plate_text)

                                if license_plate_text and license_plate_text_score >= 0.3:
                                    license_plate_text = re.sub(r'[^A-Za-z0-9]', '', license_plate_text)
                                    license_plate_text = format_license_plate_text(license_plate_text, veh_class_name)

                                    end_time = time.time()
                                    inference_time = end_time - start_time

                                    lp_crop_name = f'{license_plate_text}_{lp_track_id}.jpg'
                                    cv2.imwrite(os.path.join(LP_FOLDER_PATH, lp_crop_name), license_plate_crop_gray)
                                    
                                    cv2.rectangle(img, (int(x1), int(y1) - 40), (int(x2)+20, int(y1)), (0, 0, 0), cv2.FILLED)
                                    cv2.putText(img, str(license_plate_text), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                    cv2.putText(img, f"LP Score: {round(license_plate_text_score, 2)}", (int(x1), int(y1) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                            
                                    # Save a cropped image of the car
                                    car_crop = img[int(yvehicle1):int(yvehicle2), int(xvehicle1):int(xvehicle2), :]
                                    car_img_name = f'{veh_class_name}{track_id}_{license_plate_text}.jpg'
                                    cv2.imwrite(os.path.join(VEHICLE_FOLDER_PATH, car_img_name), car_crop)
                                    results[license_numbers] = {
                                                                    license_numbers: {
                                                                        'Vehicle': {
                                                                            'vehicle_id': track_id,
                                                                            'vehicle_class': veh_class_name,
                                                                            'vehicle_score': vehicle_score,
                                                                            'vehicle_img': car_img_name,  # Add the path to the vehicle image
                                                                        },
                                                                        'license_plate': {
                                                                            'lp_id': lp_track_id,
                                                                            'text': license_plate_text, 
                                                                            'inference_time': inference_time,
                                                                            'lp_img': lp_crop_name, 
                                                                            'text_score': license_plate_text_score,

                                                                        }
                                                                    }
                                                                }
                                    license_numbers += 1
                                    write_csv(results, f"./results/LPR_results.csv")

    img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [img_wth_box, licenses_texts, results]

async def process_frame(processor, frame, frame_number):
    processed_frame = await processor.recv(frame)
    img_with_box, licenses_texts, results = detect_objects_and_license_plates(processed_frame, frame_number)
    results_buffer.extend(results)
    return img_with_box

async def main():
    cap = cv2.VideoCapture(0)
    processor = VideoProcessor()
    frame_number = 0
    width, height = 640, 360

    ret, frame = cap.read()
    next_frame = asyncio.ensure_future(process_frame(processor, frame, frame_number))

    while ret:
        frame_number += 1
        if frame_number % 1 == 0:
            processed_frame = await next_frame
            processed_frame = cv2.resize(processed_frame, (width, height))
            cv2.imshow("Tech Titans Realtime License Plate Recognition", processed_frame)

        ret, frame = cap.read()
        if ret:
            next_frame = asyncio.ensure_future(process_frame(processor, frame, frame_number))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Write results to the CSV file after processing all frames
    write_csv(results_buffer, f"./results/LPR_results.csv")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
