import itertools
import pickle
import struct
from os import path
from sys import exit, stderr, stdin, stdout
import traceback
import time

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import fitur as fitur

__DIR__ = path.dirname(path.realpath(__file__))


def extractFeatureA(dataPerSampling):
    stderr.write("EXTRACTING A")
    stderr.flush()
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
    model.add(Conv1D(1024, kernel_size=3, padding='same', activation='relu', name="conv1d"))
    model.add(Activation('relu', name="activation"))
    model.add(Conv1D(1024, kernel_size=3, padding='same', activation='relu', name="conv1d_1"))
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


def extractFeatureB(dataPerSampling):
    listFitur = [
        fitur.findCentroid(fitur.findStable(dataPerSampling)),
        fitur.findCentroid(fitur.KenaikanP1P2B(
            dataPerSampling).values.tolist()),
        fitur.findCentroid(fitur.avgSensorValue(dataPerSampling)),
    ]
    return pd.DataFrame([list(itertools.chain(*listFitur))])


def bootstrap():
    modelA = createNetworkA()
    modelA.build(input_shape=(1, 28))
    modelA.load_weights(__DIR__ + "/a.hdf5", by_name=True)
    with open(__DIR__ + '/a.pkl', 'rb') as handle:
        scA = pickle.load(handle)
    stderr.write("MODEL A LOADED")
    stderr.flush()

    modelB = createNetworkB()
    modelB.build(input_shape=(1, 6, 1))
    modelB.load_weights(__DIR__ + "/b.hdf5", by_name=True)
    with open(__DIR__ + '/b.pkl', 'rb') as handle:
        scB = pickle.load(handle)
    stderr.write("MODEL B LOADED")
    stderr.flush()

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
        return pred


def main():
    # Bootstrapping
    stderr.write("BOOTMULAI")
    stderr.flush()
    modelA, scA, modelB, scB = bootstrap()
    stderr.write("BOOTSELESAI")

    # Main Loop
    while(True):
        try:
            CMD = struct.unpack("<B", stdin.buffer.read(1))[0]
            stderr.write("DAPET: " + str(CMD))
            stderr.flush()
            if CMD == 1:
                CMD1_PREDICT(modelA, scA, modelB, scB)
        except KeyboardInterrupt:
            exit(0)
        except Exception as e:
            stderr.write("".join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
            pass


def CMD1_PREDICT(modelA, scA, modelB, scB):
    '''
    Predict using stdin. For external apps such as JS.
    '''
    SENSORS = ['MQ2_ADC', 'MQ3_ADC', 'MQ4_ADC', 'TGS2610_ADC',
               'TGS2600_ADC', 'TGS822_ADC', 'MQ137_ADC', 'MQ138_ADC']
    NumToRead = np.frombuffer(
        buffer=stdin.buffer.read(2 * 3), dtype="uint16")
    stderr.write("DATA TO READ: " + str(NumToRead) + "\n")
    stderr.flush()
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
    stderr.write(str(df))
    stderr.flush()

    isUnder100 = False
    for sensor in SENSORS:
        stderr.write(sensor)
        stderr.flush()
        if np.min(dfP1[sensor]) < 100 or np.min(dfP2[sensor]) < 100:
            isUnder100 = True
            break
    autoInvalid = df[df['PROCESS'] == 'P2'].shape[0] <= 50 or isUnder100
    stderr.write("Predict A\n")
    stderr.flush()
    predictionA = 0 if autoInvalid else predictA(modelA, scA, df)
    stderr.write("Predict B\n")
    stderr.flush()
    predictionB = predictB(modelB, scB, df)
    stderr.flush()

    stdout.buffer.write(struct.pack("<f", float(predictionA)))  # Valid/Invalid
    stdout.buffer.write(struct.pack("<f", float(predictionB)))  # Positif/Negatif
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
    print(df)

    isUnder100 = False
    for sensor in SENSORS:
        if np.min(dfP1[sensor]) < 100 or np.min(dfP2[sensor]) < 100:
            isUnder100 = True
            break
    autoInvalid = df[df['PROCESS'] == 'P2'].shape[0] <= 50 or isUnder100
    predictionA = 0 if autoInvalid else predictA(modelA, scA, df)
    predictionB = predictB(modelB, scB, df)
    return (predictionA, predictionB)


if __name__ == "__main__":
    main()
