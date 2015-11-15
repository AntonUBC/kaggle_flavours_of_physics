# Contains paths to data and submission folders
import os


PROJECT_PATH = "D:/Data_Science_Projects/Flavours_of_Physics/"

# data and submission folders
DATA_PATH = os.path.join(PROJECT_PATH, "Data")
SUBMISSION_PATH = os.path.join(PROJECT_PATH, "Submission")
DATA_TRAIN_PATH = os.path.join(DATA_PATH, "training.csv")
DATA_TEST_PATH = os.path.join(DATA_PATH, "test.csv")
DATA_SUBMISSION_PATH = os.path.join(SUBMISSION_PATH, "submission.csv")