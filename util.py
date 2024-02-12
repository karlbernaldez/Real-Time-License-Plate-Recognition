
import os, csv
# import xlwings as xw
import datetime


highest_lp_scores = {}
timestamp = datetime.datetime.now()

# Format the timestamp as a string (if needed)
timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

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

#def write_excel(results, output_path):
#     """
#     Append the results to an existing Excel file or create a new one if it doesn't exist.

#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output Excel file.
#     """
#     # Start xlwings in silent mode (Excel won't be visible)
#     xw.App(visible=False)

#     # Check if the output Excel file exists
#     if os.path.exists(output_path):
#         # Open the existing Excel file
#         wb = xw.Book(output_path)
#         ws = wb.sheets[0]
#         # Find the next empty row
#         row = ws.range("A" + str(ws.cells.last_cell.row)).end('up').row + 1
#     else:
#         # Create a new Excel workbook if it doesn't exist
#         wb = xw.Book()
#         ws = wb.sheets[0]
#         # Define column headers
#         headers = [
#             'Timestamp', 'Vehicle ID', 'Vehicle Class', 'Vehicle Image',
#             'License Plate ID', 'LP Image Path', 'License Plate Number', 'LP Score', 'Inference Time'
#         ]
#         ws.range('A1').value = headers
#         # Start from row 2 to leave space for headers
#         row = 2

#     # Insert data into Excel (including image paths)
#     for license_numbers, data in results.items():
#         vehicle_details = data.get(license_numbers, {}).get('Vehicle', {})
#         lp_details = data.get(license_numbers, {}).get('license_plate', {})

#         timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         license_plate_number = lp_details.get('text', 'N/A')
#         lp_score = lp_details.get('text_score', 0.0)

#         # Check if this license plate has been processed before and if the new score is higher
#         if license_plate_number in highest_lp_scores and lp_score <= highest_lp_scores[license_plate_number]:
#             continue  # Skip adding this entry

#         # Update the highest LP score for this license plate
#         highest_lp_scores[license_plate_number] = lp_score

#         row_data = [
#             timestamp_str,
#             vehicle_details.get('vehicle_id', 'N/A'),
#             vehicle_details.get('vehicle_class', 'N/A'),
#             vehicle_details.get('vehicle_img', 'N/A'),
#             lp_details.get('lp_id', 'N/A'),
#             lp_details.get('img_name', 'N/A'),  # Image path for LP Image
#             license_plate_number,
#             lp_score,
#             lp_details.get('inference_time', 'N/A')
#         ]

#         ws.range(f'A{row}').value = row_data
#         row += 1

#     # Save the Excel file
#     wb.save(output_path)
#     wb.close()

def write_csv(results, output_path):
    """
    Append the results to an existing CSV file or create a new one if it doesn't exist.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    # Check if the output CSV file exists
    is_existing_file = os.path.exists(output_path)

    with open(output_path, mode='a', newline='') as csvfile:
        fieldnames = [
            'Timestamp', 'Vehicle ID', 'Vehicle Class', 'Vehicle Image',
            'License Plate ID', 'LP Image Path', 'License Plate Number', 'LP Score', 'Inference Time'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If it's a new file, write the header row
        if not is_existing_file:
            writer.writeheader()

        # Insert data into CSV (including image paths)
        for license_numbers, data in results.items():
            vehicle_details = data.get(license_numbers, {}).get('Vehicle', {})
            lp_details = data.get(license_numbers, {}).get('license_plate', {})

            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            license_plate_number = lp_details.get('text', 'N/A')
            lp_score = lp_details.get('text_score', 0.0)

            # Check if this license plate has been processed before and if the new score is higher
            if license_plate_number in highest_lp_scores and lp_score <= highest_lp_scores[license_plate_number]:
                continue  # Skip adding this entry

            # Update the highest LP score for this license plate
            highest_lp_scores[license_plate_number] = lp_score

            row_data = {
                'Timestamp': timestamp_str,
                'Vehicle ID': vehicle_details.get('vehicle_id', 'N/A'),
                'Vehicle Class': vehicle_details.get('vehicle_class', 'N/A'),
                'Vehicle Image': vehicle_details.get('vehicle_img', 'N/A'),
                'License Plate ID': lp_details.get('lp_id', 'N/A'),
                'LP Image Path': lp_details.get('lp_img', 'N/A'),  # Image path for LP Image
                'License Plate Number': license_plate_number,
                'LP Score': lp_score,
                'Inference Time': lp_details.get('inference_time', 'N/A')
            }

            writer.writerow(row_data)

# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image.

#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score.
#     """

#     detections = reader.readtext(license_plate_crop)
#     print(detections)

#     if detections == [] :
#         return None, None

#     for detection in detections:
#         bbox, text, score = detection

#         #text = text.upper().replace(' ', '')
#         text = text.upper()
#         print(text)

#         if text is not None and score is not None and bbox is not None and len(text) >= 6:
#         #if license_complies_format(text):
#         #    return format_license(text), score
#             return text, score

#     return None, None
util.py
Ipinapakita ang util.py.
