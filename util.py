import easyocr, os, csv


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {
        'O': '0', 'I': '1', 'Z': '2', 'J': '3', 
        'A': '4', 'S': '5', 'G': '6', 'T': '7', 
        'B': '8', 'Q': '0', 'D': '0'
    }

dict_int_to_char = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'J', 
        '4': 'A', '5': 'S', '6': 'G', '7': 'T', 
        '8': 'B', '9': 'g',  # or 'q' depending on the common misreadings
    }

def char2int(l):
        for char, replacement in dict_char_to_int.items():
            l = l.replace(char, replacement)
        return l

    # Correct integers to characters
def int2char(n):
        for char, replacement in dict_int_to_char.items():
            n = n.replace(char, replacement)
        return n




import csv
import os

import os
import csv

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    file_exists = os.path.isfile(output_path)
    
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'Vehicle_ID', 'Vehicle Class', 'Vehicle_Image', 
                'License_Plate_ID', 'LP Image', 'License Plate Number', 'LP_Score'
            ])
        
        for track_id, data in results.items():
            vehicle_details = data.get('vehicle_details', {})
            lp_details = data.get('license_plate_details', {})
            
            writer.writerow([
                track_id,
                vehicle_details.get('class_name', 'N/A'),
                vehicle_details.get('car_img_name', 'N/A'),
                lp_details.get('lp_track_id', 'N/A'),
                lp_details.get('img_name', 'N/A'),
                lp_details.get('license_plate_text', 'N/A'),
                lp_details.get('lp_score', 'N/A')
            ])




def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)
    print(detections)

    if detections == [] :
        return None, None

    for detection in detections:
        bbox, text, score = detection

        #text = text.upper().replace(' ', '')
        text = text.upper()
        print(text)

        if text is not None and score is not None and bbox is not None and len(text) >= 6:
        #if license_complies_format(text):
        #    return format_license(text), score
            return text, score

    return None, None
