import cv2
import dlib
import numpy as np
import youtube_dl
import matplotlib.pyplot as plt


# function to download YouTube video
def download_video(url):
    ydl_opts = {'outtmpl': 'video.mp4'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# function to detect faces in video frames
def detect_faces(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    faces, _ = hog.detectMultiScale(frame)
    return faces

# function to detect facial landmarks
def detect_landmarks(frame, rects):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = dlib.full_object_detections()
    for rect in rects:
        shape = predictor(frame, rect)
        faces.append(shape)
    return faces

def extract_features(jaw, right_eyebrow, left_eyebrow, nose, right_eye, left_eye, mouth, chin):
    # create an empty list to store the features
    features = []
    # calculate the distance between certain landmarks 
    # or the angle between certain lines and append it to the list
    distance = np.linalg.norm(jaw[0]-jaw[-1])
    features.append(distance)
    # you can append more features as you see fit
    return features

# function to analyze expressions
def analyze_expressions(frame, shape_predictor):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    expressions = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # extract the coordinates for the facial landmarks
        jaw = shape[0:17]
        right_eyebrow = shape[17:22]
        left_eyebrow = shape[22:27]
        nose = shape[27:36]
        right_eye = shape[36:42]
        left_eye = shape[42:48]
        mouth = shape[48:60]
        chin = shape[6:17]
        # use the coordinates to determine the facial expression
        expression = determine_expression(jaw, right_eyebrow, left_eyebrow, nose, right_eye, left_eye, mouth, chin)
        expressions.append(expression)
    return expressions

import numpy as np
from sklearn.externals import joblib

# Determines the expressions by pose estimation of facial features
def determine_expression(jaw, right_eyebrow, left_eyebrow, nose, right_eye, left_eye, mouth, chin):
    # extract features from the facial landmarks
    features = extract_features(jaw, right_eyebrow, left_eyebrow, nose, right_eye, left_eye, mouth, chin)
    # load the classifier
    clf = joblib.load('expression_classifier.pkl') 
    # use the classifier to predict the expression
    expression = clf.predict(features.reshape(1,-1))[0]
    return expression

# function to analyze video frame by frame
def analyze_expressions(youtube_url):
    # download the video
    video = download_video(youtube_url)
    # extract the frames
    frames = extract_frames(video)
    # create an empty list to store the expressions
    expressions = []
    # create an empty list to store the number of detected objects
    detected_objects = []
    # loop through the frames
    for frame in frames:
        # detect the facial landmarks
        landmarks = detect_landmarks(frame)
        # determine the expression
        expression = determine_expression(landmarks)
        # add the expression to the list
        expressions.append(expression)
        # detect the number of objects in the frame
        num_objects = detect_objects(frame)
        # add the number of detected objects to the list
        detected_objects.append(num_objects)
    # plot the expressions over time
    plt.plot(expressions)
    plt.xlabel('Frame')
    plt.ylabel('Expression')
    # add the number of detected objects to the plot
    plt.plot(detected_objects)
    plt.xlabel('Frame')
    plt.ylabel('Number of Detected Objects')
    plt.show()

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_count = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # detect faces in frame
            faces = detect_faces(frame)
            face_count.append(len(faces))
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                analyze_expressions(frame, faces)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Gets input from the user for the video to be analyzed
url = input("Enter the youtube video url: ")
# download video from YouTube
download_video(url)
# analyze the video
analyze_video('video.mp4')
