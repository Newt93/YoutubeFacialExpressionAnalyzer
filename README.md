# YoutubeFacialExpressionAnalyzer
A Youtube Facial Expression Analyzer that uses a neural network and advanced facial detection and facial pose estimation to visualize facial expressions in a Youtube video.

This project can be customized to let a user input a video URL from Youtube. Then it takes it, detects the objects and faces in the video using HOGDescriptor (OpenCV), and estimates the facial landmarks using advanced facial recognition. Afterwards it loads in a classifier that can be customized and trained on labeled data of different facial expressions. The classifier then uses keras to run it through an advanced neural network and then pickles the entire classifier. You can then import the pickled file into the Youtube Expression Analyzer (via the code).

Must have extensive knowledge of OpenCV, Machine Learning, Neural Networks, and Classifiers to be able to implement this program. Be sure to check with Youtube's ToS to make sure you don't violate anything! For research purposes only.

If you'd like a custom program built please contact leakzgggaming@gmail.com
