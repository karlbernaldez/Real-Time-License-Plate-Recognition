import easyocr, os, csv


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


import csv
import os

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
            writer.writerow(['frame_nmr', 'vehicle_img', 'vehicle_class', 'license_number', 'license_number_score'])
        
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    writer.writerow([
                        frame_nmr,
                        results[frame_nmr][car_id]['car'].get('vehicle_img', 'N/A'),  # Get the vehicle image file path
                        results[frame_nmr][car_id]['car'].get('vehicle_class', 'N/A'), # Get the vehicle class
                        results[frame_nmr][car_id]['license_plate']['text'],  # Get the license number
                        results[frame_nmr][car_id]['license_plate']['text_score']  # Get the license number score
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
