
# Face Detection and Alert System

## Description
This is a Python-based face detection and drowsiness alert system that uses webcam input to detect user drowsiness or sleep and sends an alert accordingly.
This system uses Dlib, and facial landmark detection to monitor your face for signs of drowsiness or sleep. When the system detects a closed eye for a certain period, it alerts the user with a sound and saves a screenshot of the current state. The system can help prevent accidents by alerting drivers about their tiredness.

## Installation

### Prerequisites
- Python 3.7
- OpenCV
- Dlib
- imutils

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/projectname.git
    ```
2. Navigate to the project directory:
    ```bash
    cd projectname
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the facial landmark predictor file (e.g., `shape_predictor_68_face_landmarks.dat`) from [Dlib's model page](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

### Requirements
- OpenCV
- Dlib
- imutils
- winsound (for sound alerts)

## Usage

To run the project, simply execute the Python script:

```bash
python main.py

