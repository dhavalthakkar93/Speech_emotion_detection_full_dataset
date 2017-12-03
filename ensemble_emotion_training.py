"""ensemble_emotion_training.py: purpose of this script is to train ensemble model based on  Toronto emotion speech
dataset to predict the emotion from speech """

__author__ = "Dhaval Thakkar"


import glob
import librosa
import librosa.display
import numpy as np
import _pickle as pickle
from sklearn import svm
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve

# Method to extract features from speech using librosa


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

# Method to extract label name and extract features from audio file


def parse_audio_files(path):
    features, labels = np.empty((0, 193)), np.empty(0)
    labels = []
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split("_")[3].split(".")[0])
        print(fn)
    return np.array(features), np.array(labels)


# Get labels and features of audion file of specified path
tr_features, tr_labels = parse_audio_files('./training_sounds1/*.wav')

print(tr_labels)


# Convert features and labels to the pandas Series data type
tr_features = np.array(tr_features, dtype=pd.Series)
tr_labels = np.array(tr_labels, dtype=pd.Series)

# Model1:- support vector classifier using linear kernel
model1 = svm.SVC(kernel='linear', C=1000, gamma='auto')
model1.fit(X=tr_features.astype(int), y=tr_labels.astype(str))

# Model2:- support vector classifier using RBF kernel
model2 = svm.SVC(kernel='rbf', C=1000, gamma='auto')
model2.fit(X=tr_features.astype(int), y=tr_labels.astype(str))

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

# Ensemble the linear and RBF kernel model
estimators = [('linear', model1), ('rbf', model2)]
# Voting Classifier
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, tr_features.astype(int), tr_labels.astype(str), cv=kfold)
ensemble.fit(X=tr_features.astype(int), y=tr_labels.astype(str))

# Learning curve plotting
train_sizes, train_scores, test_scores = learning_curve(ensemble, tr_features, tr_labels, n_jobs=-1, cv=kfold,
                                                        train_sizes=np.linspace(.1, 1.0, 5), verbose=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Genetic Exported Pipeline Model")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.gca().invert_yaxis()

plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                 color="g")
line_up = plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
line_down = plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.ylim(-.1, 1.1)
plt.legend(loc="lower right")
plt.show()

# File name to store the trained model
filename = 'Ensemble_Model_protocol2.sav'

# Store the trained model
pickle.dump(ensemble, open(filename, 'wb'), protocol=2)

print('Model Saved..')



