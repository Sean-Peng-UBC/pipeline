from matplotlib import pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math


pi = math.pi
scaler = preprocessing.MinMaxScaler()


def create_vector(df):
    semipi = pi/2
    scope = pi/16
    dataset = []
    ext = []
    length = df.max().Cluster+1
    for cluster in range(0, length/4):
        data = df[df.Cluster == cluster].copy()
        vector = np.zeros((2, 16))
        var = [0]*3
        num = len(data)

        if num > 1:
            for i in range(0, num - 1):
                orgin = data.iloc[i]
                clock = orgin.Clock
                dist = orgin.Dist
                for j in range(i + 1, num):
                    end = data.iloc[j]
                    dclock = end.Clock - clock
                    ddist = end.Dist - dist
                    distance = np.sqrt(np.square(ddist) + np.square(dclock))
                    if ddist:
                        angle = math.atan(1.0*dclock/ddist) + semipi
                        index = int(angle/scope)
                    else:
                        index = 0
                    vector[0][index] += 1
                    vector[1][index] += distance
            vector = vector/np.square(num)

            var[0] = data.var().Length
            var[1] = data.var().Width
            var[2] = data.var().Peak

        else:
            var[0] = 1.0*data.iloc[0].Length/data.iloc[0].Width
            var[1] = 1.0*data.iloc[0].Width/data.iloc[0].Length
            var[2] = 1.0*data.iloc[0].Peak/data.iloc[0].Length

        dataset.append(vector)
        ext.append(var)
        if cluster%200 == 0:
            print cluster

    dataset = np.array(dataset)
    ext = np.array(ext)
    for array in dataset:
        temp = array.copy()
        for i in range(0, 15):
            array[:, i] = temp[:, i-1]*0.3 + temp[:, i]*0.4 + temp[:, i+1]*0.3
        array[:, 15] = temp[:, 14] * 0.3 + temp[:, 15] * 0.4 + temp[:, 0] * 0.3

    dataset = dataset.reshape(len(dataset), -1)
    dataset = np.hstack((dataset, ext))
    dataset = scaler.fit_transform(dataset)

    return dataset


def compare(corrosion, dataset):
    dd = dataset - corrosion
    dd = np.linalg.norm(dd, ord=2, axis=1)
    return dd.argsort()


def show(outindex, df):
    for i in range(0, len(outindex)):
        box = outindex[i]
        data = df[df.Cluster == box].copy()
        data.Dist += 500
        data.Clock += 500
        segment = np.zeros((1000, 1000))
        for j in range(0, len(data)):
            corrosion = data.iloc[j]
            segment[corrosion.Clock - corrosion.Width:corrosion.Clock + corrosion.Width,
            corrosion.Dist - corrosion.Length:corrosion.Dist + corrosion.Length] = corrosion['Peak']
        segment = segment[250:750, 250:750]
        if i == 0:
            fig = plt.figure()
        if len(outindex) > 1:
            plt.subplot(2, 2, i+1)
            plt.title(str(i+1))
        plt.imshow(segment, vmin=0, vmax=40, cmap='jet', aspect='auto')
        plt.colorbar()


def main():
    dataset1 = pd.read_csv('dataset1_500.csv')
    dataset2 = pd.read_csv('dataset2_500.csv')
    vector1 = create_vector(dataset1)
    vector2 = create_vector(dataset2)

    i = 5
    corrosion = vector2[i]
    outindex = compare(corrosion, vector1)

    show([i], dataset2)
    show(outindex[:4], dataset1)


if __name__ == '__main__':
    main()