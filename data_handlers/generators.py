import tensorflow as tf
import os.path
import pandas as pd
from pathlib import Path
import math
from src.utils import get_age
import numpy as np


# Defining the ImageDataGenerator and what Preprocessing should be done to the images

def getDataRegression(db):
    image_dir = Path('./data/{}_crop'.format(db))
    # image_dir = Path('./data/age_prediction')
    mat_path = './data/{}_crop/{}.mat'.format(db, db)

    ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
    filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages = pd.Series(get_age(mat_path, db), name="Age")
    # ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name="Age").astype(int)
    images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
        drop=True)
    return images.sample(15000, random_state=np.random.randint(1000)).reset_index(drop=True)


def getDataClassification(db, nr_classes):
    image_dir = Path('./data/{}_crop'.format(db))
    # image_dir = Path('./age_prediction')
    mat_path = './data/{}_crop/{}.mat'.format(db, db)

    ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
    filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages = pd.Series(get_age(mat_path, db), name="Age")
    images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
        drop=True)

    ## Calculation of the Age Groups depending on the Nr of Classes declared earlier
    #max_age = ages.max()
    #interval = max_age / nr_classes
    #floored_interval = math.floor(interval)
    #i = 0
    #classes = []
    #while len(classes) < nr_classes:
    #    classes.append('{}-{}'.format(i, i + floored_interval - 1))
    #    i += floored_interval
    #images["Age"] = pd.Series(images["Age"].apply(
    #    lambda x: "{}-{}".format(int((x // interval) * floored_interval),
    #                             int((x // interval) * floored_interval + floored_interval - 1))))
    #return images.sample(20000, random_state=np.random.randint(1000)).reset_index(drop=True), classes

    #max_age = ages.max()
    #interval = max_age // nr_classes
    #floored_interval = math.floor(interval)
    #i = 0
    #classes = []
    #while len(classes) < nr_classes:
    #    classes.append(i + 0.5*interval)
    #    i += interval
    #images["Age"] = pd.Series(images["Age"].apply(
    #    lambda x: (x // nr_classes)*nr_classes + 0.5*interval))
    return images.sample(10000, random_state=np.random.randint(1000)).reset_index(drop=True), ages.max()
