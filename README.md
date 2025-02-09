# Rearview collision warning

The Rear-End Collision Warning System uses the Indian Roads ADAS Perception System Dataset (IRAPSD), which includes 60 video files from front and rear dash cameras recorded with the 70mai dashcam (36.2 GB)(https://www.kaggle.com/datasets/surya498/irapsd-v2b-v01?resource=download) . The system focuses on detecting vehicles approaching from behind and predicts potential collisions in the region of penetration.
Leveraging YOLO (You Only Look Once) for real-time vehicle detection and Kalman Filter for tracking and prediction, the system calculates the Time to Collision (TTC). When TTC is less than 5-10 seconds, an alert is triggered, warning the driver of an impending rear-end collision.
By combining object detection and predictive tracking, this system enhances road safety by providing timely warnings for rear-end collision risks.
Video details:
Total Frames: 4501
Frame Rate: 24 FPS
Duration: 187.54 seconds

Create a virtual environment → python -m venv rcas
Activate it → rcas\Scripts\activate
Install dependencies → pip install opencv-python easyocr pytesseract numpy
Download dataset → https://www.kaggle.com/datasets/surya498/irapsd-v2b-v01?resource=download
Run the script → python your_script.py (both script)
Deactivate → deactivate

