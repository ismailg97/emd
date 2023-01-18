import tensorflow as tf
import os.path
import pandas as pd
from pathlib import Path
import math
from src.utils import get_age
import numpy as np
from matplotlib import pyplot as plt


image_dir_age_prediction = Path('./data/age_prediction')
image_dir_utk = Path('./data/UTKFace')

# Defining the ImageDataGenerator and what Preprocessing should be done to the images

def getDataRegression(db):
    image_dir = Path('./data/{}_crop'.format(db))
    mat_path = './data/{}_crop/{}.mat'.format(db, db)

    ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
    filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages = pd.Series(get_age(mat_path, db), name="Age")
    images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
        drop=True)

    filepaths_age_prediction = pd.Series(list(image_dir_age_prediction.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages_age_prediciton = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name="Age")
    images_age_prediction = pd.concat([filepaths_age_prediction, ages_age_prediciton], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
        drop=True)

    filepaths_utk = pd.Series(list(image_dir_utk.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages_utk = pd.Series(filepaths_utk.apply(lambda x: os.path.split(x.split('_')[0])[-1]), name="Age").astype(float)
    images_utk = pd.concat([filepaths_utk, ages_utk], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
        drop=True)

    images_df = pd.concat([images, images_age_prediction, images_utk])

    plt.title('Age Distribution')
    plt.ylabel('amount')
    plt.xlabel('age')
    plt.hist(images_df['Age'], orientation='vertical')
    plt.savefig(fname="./checkpoints/regressorMSE/age_distribution")
    plt.show()

    return images_df.sample(25000, random_state=np.random.randint(1000)).reset_index(drop=True)


def getDataClassification(db, nr_classes):
    image_dir = Path('./data/{}_crop'.format(db))
    mat_path = './data/{}_crop/{}.mat'.format(db, db)


    ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
    filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages = pd.Series(get_age(mat_path, db), name="Age")
    images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
        drop=True)

    filepaths_age_prediction = pd.Series(list(image_dir_age_prediction.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages_age_prediciton = pd.Series(filepaths_age_prediction.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name="Age").astype(float)
    images_age_prediction = pd.concat([filepaths_age_prediction, ages_age_prediciton], axis=1).sample(frac=1.0,random_state=1).dropna().reset_index(drop=True)

    filepaths_utk = pd.Series(list(image_dir_utk.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages_utk = pd.Series(filepaths_utk.apply(lambda x: os.path.split(x.split('_')[0])[-1]),name="Age").astype(float)
    images_utk = pd.concat([filepaths_utk, ages_utk], axis=1).sample(frac=1.0,random_state=1).dropna().reset_index(drop=True)

    #images_df = pd.concat([images, images_age_prediction, images_utk])

    #images_df = pd.concat([images, images_utk])

    #images_df = pd.concat([images, images_age_prediction])

    images_df = images

    #print(images_utk)
    #print(images_df)
    #exit()

    #plt.title('IMDB Verteilung')
    #plt.ylabel('Anzahl der Daten')
    #plt.xlabel('Alter')
    #plt.hist(images['Age'], orientation='vertical')
    #plt.savefig(fname="./checkpoints/imdb_verteilung")
    #plt.show()

    #plt.title('WIKI Verteilung')
    #plt.ylabel('Anzahl der Daten')
    #plt.xlabel('Alter')
    #plt.hist(images['Age'], orientation='vertical')
    #plt.savefig(fname="./checkpoints/wiki_verteilung")
    #plt.show()

    #plt.title('Age_Prediction Verteilung')
    #plt.ylabel('Anzahl der Daten')
    #plt.xlabel('Alter')
    #plt.hist(images_age_prediction['Age'], orientation='vertical')
    #plt.savefig(fname="./checkpoints/age_prediction_verteilung")
    #plt.show()

    #plt.title('UTKFace Verteilung')
    #plt.ylabel('Anzahl der Daten')
    #plt.xlabel('Alter')
    #plt.hist(images_utk['Age'], orientation='vertical')
    #plt.savefig(fname="./checkpoints/utkface120_distribution")
    #plt.show()

    images_df = images_df[images_df['Age'] < 100]

    #plt.figure()
    #images['Age'].plot()
    #images.plot(x='Age', y=images['Age'].nunique(), kind='scatter')
    #exit()

    plt.title('{}+Age_Prediction Verteilung'.format(db))
    plt.ylabel('Anzahl der Samples')
    plt.xlabel('Alter')
    plt.hist(images_df['Age'], orientation='vertical')
    plt.savefig(fname="./checkpoints/classificatorXE/{}/{}+age_prediction_distribution".format(nr_classes, db))
    plt.show()

    #plt.title('IMDB+UTKFace Dataset Distribution')
    #plt.ylabel('Amount of Samples')
    #plt.xlabel('Age')
    #plt.hist(images_df['Age'], orientation='vertical')
    #plt.savefig(fname="./checkpoints/classificatorXE/{}/imdb+utk_age_distribution".format(nr_classes))
    #plt.show()


    return images_df.sample(25000, random_state=np.random.randint(1000)).reset_index(drop=True), ages.max()


