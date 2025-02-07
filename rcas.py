import cv2
import pytesseract
import re

# Set Tesseract path (Windows users must do this)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract speed from detected text
def extract_speed(text):
    match = re.search(r"(\d+)\s*km/h", text)  # Look for speed format "xx km/h"
    return int(match.group(1)) if match else None

# Load video
video_path = "C:/Users/Abhay/OneDrive/Desktop/Projects/Rearview_collision_warning/rearview_collision_warning/testing/Copy of NO20230924-121041-000402B.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
output_path = "demo_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_interval = fps * 2  # Process every 2 seconds
frame_number = 0
previous_speed = None  # Store last detected speed
status_message = "RCAS: OFF"  # Default status

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Process every 2 seconds
    if frame_number % frame_interval == 0:
        # Crop bottom-right region where speed is displayed
        h, w, _ = frame.shape
        crop_x_start = int(w * 0.75) - 250  
        crop_y_start = int(h * 0.80)  
        cropped_region = frame[crop_y_start:h, crop_x_start:w]

        # Resize and preprocess for better OCR accuracy
        cropped_region_resized = cv2.resize(cropped_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(cropped_region_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, processed_frame = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

        # Perform OCR
        text = pytesseract.image_to_string(processed_frame, config="--psm 7 -c tessedit_char_whitelist=0123456789kmh/")
        print(f"Frame {frame_number}: Detected Text: {text}")

        # Extract speed value
        detected_speed = extract_speed(text)
        if detected_speed is not None:
            # Determine if speed is increasing or decreasing
            if previous_speed is not None:
                if detected_speed < previous_speed:
                    status_message = "RCAS: Active"
                elif detected_speed > previous_speed:
                    status_message = "RCAS: OFF"

            previous_speed = detected_speed  # Update previous speed

    # Overlay speed text on the frame
    if previous_speed is not None:
        text_display = f"Speed: {previous_speed} km/h"
        status_display = status_message

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 255, 0)  # Green
        status_color = (0, 255, 255)  # Yellow

        text_x = frame_width - 250  # Position at top-right
        text_y = 50  
        status_y = 80  # Below speed text

        cv2.putText(frame, text_display, (text_x, text_y), font, font_scale, text_color, font_thickness)
        cv2.putText(frame, status_display, (text_x, status_y), font, font_scale, status_color, font_thickness)

    # Write frame to output video
    out.write(frame)

    # Optional: Display the frame while processing
    cv2.imshow('Video Processing', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Saved as 'demo_video.mp4'.")
