import cv2

# Get the list of available capture devices
def list_capture_devices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

# Print the list of available capture devices
devices = list_capture_devices()
print("Available capture devices: ", devices)
