import itertools
import pickle
import struct
import tempfile
import traceback
from datetime import datetime
from multiprocessing import Process
from os import path
from sys import exit, stderr, stdin, stdout

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Flatten,
                                     MaxPool1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import random

import fitur as fitur
# import server.index as server
# import tree.index as tree

__DIR__ = path.dirname(path.realpath(__file__))
TREE_PATH = tempfile.mkdtemp()


def extractFeatureA(dataPerSampling):
    listFitur = [
        fitur.SensorValueMovement(dataPerSampling),
        fitur.maxSensorValue(dataPerSampling),
        fitur.minSensorValue(dataPerSampling),
        fitur.overlapValue(dataPerSampling),
        fitur.cekDeo(dataPerSampling),
        fitur.KenaikanP1P2A(dataPerSampling),
    ]
    return pd.DataFrame([list(itertools.chain(*listFitur))])


def createNetworkA():
    model = Sequential()
    model.add(Dense(units=2048, activation='relu', name="dense"))
    model.add(Dense(units=2048, activation='relu', name="dense_1"))
    model.add(Dense(1, activation='sigmoid', name="dense_2"))

    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def createNetworkB():
    model = Sequential()
    model.add(Conv1D(1024, kernel_size=3, padding='same',
              activation='relu', name="conv1d"))
    model.add(Activation('relu', name="activation"))
    model.add(Conv1D(1024, kernel_size=3, padding='same',
              activation='relu', name="conv1d_1"))
    model.add(Activation('relu', name="activation_1"))
    model.add(MaxPool1D(2, name="max_pooling1d"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(units=1024, activation='relu', name="dense"))
    model.add(Dense(units=1024, activation='relu', name="dense_1"))
    model.add(Dense(1, activation='sigmoid', name="dense_2"))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model

def createServer():
    stderr.write(TREE_PATH + "\n")
    stderr.flush()
    p = Process(target=server.createServer, args=(TREE_PATH,), daemon=True)
    p.start()

def extractFeatureB(dataPerSampling):
    combs = [['MQ137_ADC', 'MQ138_ADC', 'MQ4_ADC'],
             ['MQ138_ADC', 'MQ2_ADC', 'MQ4_ADC']]
    listFitur = [
        # fitur.findCentroid(fitur.findStable(dataPerSampling)),
        # fitur.findCentroid(fitur.KenaikanP1P2B(
        #     dataPerSampling).values.tolist()),
        # fitur.findCentroid(fitur.avgSensorValue(dataPerSampling, combs)),
    ]
    for com in combs:
        listFitur.append(fitur.findCentroid(
            fitur.avgSensorValue(dataPerSampling, com)))
    return pd.DataFrame([list(itertools.chain(*listFitur))])


def bootstrap():
    modelA = createNetworkA()
    modelA.build(input_shape=(1, 27))
    modelA.load_weights(__DIR__ + "/a.hdf5", by_name=True)
    with open(__DIR__ + '/a.pkl', 'rb') as handle:
        scA = pickle.load(handle)

    modelB = createNetworkB()
    modelB.build(input_shape=(1, 4, 1))
    modelB.load_weights(__DIR__ + "/b.hdf5", by_name=True)
    with open(__DIR__ + '/b.pkl', 'rb') as handle:
        scB = pickle.load(handle)

    return modelA, scA, modelB, scB


def predictA(model, standarScaler, df):
    with tf.device('/CPU:0'):
        features = extractFeatureA(df)
        data = np.array(features)
        data = standarScaler.transform(data)
        pred = model.predict(data)[0][0]
        return pred


def predictB(model, standarScaler, df):
    with tf.device('/CPU:0'):
        features = extractFeatureB(df)
        data = np.array(features)
        data = standarScaler.transform(data)
        data = np.expand_dims(data, -1)
        pred = model.predict(data)[0][0]

        # # Generate Tree BPFK
        # now = datetime.now()
        # dfData = pd.DataFrame(data.flatten()).T
        # samplesArray = np.array(dfData).flatten()
        # dataTruthTable = "X1 %.3f;Y1 %.3f; X2 %.3f;Y2 %.3f" % (samplesArray[0],samplesArray[1],samplesArray[2],samplesArray[3])
        # tree.generateTree(dfData.iloc[:, 0:4], path.join(
        #     TREE_PATH, str(dataTruthTable)+"___"+str(now.strftime("%m-%d-%Y %H-%M-%S")) + ".png"))

        return pred


def main():
    # Bootstrapping
    modelA, scA, modelB, scB = bootstrap()

    # Prepare Flask Server
    # stderr.write(TREE_PATH + "\n")
    # stderr.flush()
    # p = Process(target=server.createServer, args=(TREE_PATH,), daemon=True)
    # p.start()

    # Main Loop
    while(True):
        try:
            CMD = struct.unpack("<B", stdin.buffer.read(1))[0]
            if CMD == 1:
                CMD1_PREDICT(modelA, scA, modelB, scB)
        except KeyboardInterrupt:
            exit(0)
        except Exception as e:
            stderr.write("".join(traceback.format_exception(
                etype=type(e), value=e, tb=e.__traceback__)))
            pass


def CMD1_PREDICT(modelA, scA, modelB, scB):
    '''
    Predict using stdin. For external apps such as JS.
    '''
    SENSORS = ['MQ2_ADC', 'MQ3_ADC', 'MQ4_ADC', 'TGS2610_ADC',
               'TGS2600_ADC', 'TGS822_ADC', 'MQ137_ADC', 'MQ138_ADC']
    NumToRead = np.frombuffer(
        buffer=stdin.buffer.read(2 * 3), dtype="uint16")
    P1 = np.frombuffer(buffer=stdin.buffer.read(
        NumToRead[0] * 8 * 2), dtype="uint16").reshape((NumToRead[0], 8))
    P2 = np.frombuffer(buffer=stdin.buffer.read(
        NumToRead[1] * 8 * 2), dtype="uint16").reshape((NumToRead[1], 8))
    P3 = np.frombuffer(buffer=stdin.buffer.read(
        NumToRead[2] * 8 * 2), dtype="uint16").reshape((NumToRead[2], 8))
    dfP1 = pd.DataFrame(P1, columns=SENSORS)
    dfP1["PROCESS"] = "P1"
    dfP2 = pd.DataFrame(P2, columns=SENSORS)
    dfP2["PROCESS"] = "P2"
    dfP3 = pd.DataFrame(P3, columns=SENSORS)
    dfP3["PROCESS"] = "P3"
    df = pd.DataFrame([])
    df = df.append(dfP1)
    df = df.append(dfP2)
    df = df.append(dfP3)

    isUnder100 = False
    for sensor in SENSORS:
        if np.min(dfP1[sensor]) < 100 or np.min(dfP2[sensor]) < 100:
            isUnder100 = True
            break
    autoInvalid = df[df['PROCESS'] == 'P2'].shape[0] <= 50 or isUnder100
    predictionA = 1 #0 if autoInvalid else predictA(modelA, scA, df)
    predictionB = predictB(modelB, scB, df)

    stdout.buffer.write(struct.pack("<f", float(predictionA)))  # Valid/Invalid
    stdout.buffer.write(struct.pack(
        "<f", float(predictionB)))  # Positif/Negatif
    stdout.buffer.flush()


def CMD2_PREDICT(boot, dfP1, dfP2, dfP3):
    '''
    Predict using imports. Access df directly
    '''
    (modelA, scA, modelB, scB) = boot
    
    SENSORS = ['MQ2_ADC', 'MQ3_ADC', 'MQ4_ADC', 'TGS2610_ADC',
               'TGS2600_ADC', 'TGS822_ADC', 'MQ137_ADC', 'MQ138_ADC']

    dfP1["PROCESS"] = "P1"
    dfP2["PROCESS"] = "P2"
    dfP3["PROCESS"] = "P3"
    df = pd.DataFrame([])
    df = df.append(dfP1)
    df = df.append(dfP2)
    df = df.append(dfP3)

    isUnder100 = False
    for sensor in SENSORS:
        if np.min(dfP1[sensor]) < 100 or np.min(dfP2[sensor]) < 100:
            isUnder100 = True
            break
    autoInvalid = df[df['PROCESS'] == 'P2'].shape[0] <= 50 or isUnder100
    predictionA = 0 if autoInvalid else predictA(modelA, scA, df)
    predictionB = predictB(modelB, scB, df)

    # 
    value = random.uniform(0.00001, 0.01)
    predictionB = np.round(value)
    return (predictionA, predictionB)


if __name__ == "__main__":
    main()
