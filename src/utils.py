import keras.preprocessing.image
from scipy.io import loadmat
from datetime import datetime
import numpy as np
import os
import keras.utils as ut


def calc_age(taken, dob, face_score, second_face_score):
    if 'n' in str(face_score) or 'a' not in str(second_face_score) or face_score < 1:
        return np.nan
    birth = datetime.fromordinal(dob)
    age = int(taken - birth.year)
    if age < 0 or age > 100:
        return np.nan
    return age
    # assume the photo was taken in the middle of the year
    #if birth.month < 7:
        #return taken - birth.year
    #else:
        #return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def get_age(mat_path, db):
    meta = loadmat(mat_path)
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    age = [calc_age(photo_taken[i], dob[i], face_score[i], second_face_score[i]) for i in range(len(dob))]
    return age


def load_data(mat_path):
    d = loadmat(mat_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def tensorToNumpy(a):
    return a.numpy()

