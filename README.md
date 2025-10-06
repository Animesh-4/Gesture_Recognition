# hand-gesture-recognition-using-mediapipe
Estimate hand pose using MediaPipe (Python version).<br> This is a sample 
program that recognizes hand signs and finger gestures with a simple MLP using the detected key points.
<br> 
This repository contains the following contents.
* Sample program
* Hand sign recognition model(TFLite)
* Finger gesture recognition model(TFLite)
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

# Demo
Here's how to run the demo using your webcam.
```bash
pip install mediapipe opencv-python tensorflow scikit-learn matplotlib
python app.py
```

The following options can be specified when running the demo.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.5)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### model/point_history_classifier
This directory stores files related to finger gesture recognition.<br>
The following files are stored.
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv)
* Inference module(point_history_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.


# Training

You can add new gestures and retrain the models for both hand sign and finger gesture recognition.

***
### Hand Sign Recognition Training

#### 1. Data Collection
* Press the **k** key while the application is running to enter data collection mode (the screen will display `MODE:Logging Key Point`).
* Make a hand sign and press a number key (**0-9**) to save the hand's key points.
* The key points are saved to `model/keypoint_classifier/keypoint.csv`. The first column is the class ID (the number you pressed), and the subsequent columns are the landmark coordinates.
* The initial dataset includes three signs: **Open Hand** (ID: 0), **Close Hand** (ID: 1), and **Pointing** (ID: 2). You can add new classes or delete the existing data to create your own dataset.

#### 2. Model Training
* Open the `keypoint_classification.ipynb` file in a Jupyter Notebook.
* Run all the cells from top to bottom to train the model on the data in the `.csv` file.
* If you've added new classes, be sure to update the `NUM_CLASSES` variable in the notebook and add your new labels to `model/keypoint_classifier/keypoint_classifier_label.csv`.

***
### Finger Gesture Recognition Training

#### 1. Data Collection
* Press the **h** key while the application is running to enter data collection mode (the screen will display `MODE:Logging Point History`).
* Perform a finger gesture (like drawing a circle) and press a number key (**0-9**) to save the history of your index finger's coordinates.
* The coordinate history is saved to `model/point_history_classifier/point_history.csv`. The first column is the class ID.
* The initial dataset includes four gestures: **Stationary** (ID: 0), **Clockwise** (ID: 1), **Counter-Clockwise** (ID: 2), and **Moving** (ID: 4).

#### 2. Model Training
* Open the `point_history_classification.ipynb` file in a Jupyter Notebook.
* Run all the cells from top to bottom to train the model.
* If you've added new gesture classes, update the `NUM_CLASSES` variable in the notebook and add your new labels to `model/point_history_classifier/point_history_classifier_label.csv`.