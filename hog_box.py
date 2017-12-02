from matplotlib import pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import math


xl = pd.ExcelFile('CDG.xlsx')
df = xl.parse(xl.sheet_names[0], skiprows=11)
df = df[df.Identifier0 == 'MELO-ANOM']
df = df[df['Cluster\nMethod'] == '3t x 3t']
df = df.loc[:, ['Clock', 'Dist', 'Parent\nCluster-ID']]
df.Dist = df.Dist * 1000
for x in df.index:
    df.loc[x, 'Clock'] = ((df.loc[x, 'Clock'].hour-1) * 30 + df.loc[x, 'Clock'].minute * 0.5) / 360 * 914.4 * 3.14

scaler = preprocessing.MinMaxScaler()
semipi = math.pi/2
scope = math.pi/16
dataset = []
var = []
for cluster in range(1, 13284):
    vector = np.zeros((4, 16))
    vec = [0]*5
    data = df[df['Parent\nCluster-ID'] == 'CLUSTER '+str(cluster)]
    num = len(data)

    if data.Clock.max() - data.Clock.min() > 2000:
        for i in range(0, num):
            if data.iloc[i].Clock > 2000:
                data.iloc[i].Clock -= 2871
    if data.Dist.max() - data.Dist.min() > 5000:
        for i in range(0, num):
            if data.iloc[i].Dist < 1000:
                data.iloc[i].Dist += data.Dist.max()

    vec[0] = data.var().Clock
    vec[1] = data.var().Dist
    vec[2] = data.max().Dist - data.min().Dist
    vec[3] = data.max().Clock - data.min().Clock
    if vec[2] == 0:
        vec[2] = 1
    vec[4] = vec[3]/vec[2]

    for i in range(0, num-1):
        clock = data.iloc[i].Clock
        dist = data.iloc[i].Dist
        for j in range(i+1, num):
            dclock = data.iloc[j].Clock - clock
            ddist = data.iloc[j].Dist - dist
            distance = np.sqrt(abs(ddist)+abs(dclock))
            if ddist:
                angle = math.atan(dclock/ddist) + semipi
                index = int(angle/scope)
            else:
                index = 0
            vector[0][index] += 1
            vector[1][index] += distance

    vector = vector/num

    data = scaler.fit_transform(data[['Clock', 'Dist']])
    for i in range(0, num):
        for j in [0, 1]:
            index = int(data[i, j]*16)
            if index == 16:
                index = 15
            vector[j+2][index] += 1

    dataset.append(vector)
    var.append(vec)

var = np.array(var)
dataset = np.array(dataset)
for array in dataset:
    temp = array.copy()
    for i in range(0, 15):
        array[:, i] = temp[:, i-1]*0.3 + temp[:, i]*0.4 + temp[:, i+1]*0.3
    array[:, 15] = temp[:, 14] * 0.3 + temp[:, 15] * 0.4 + temp[:, 0] * 0.3

dataset = dataset.reshape(len(dataset), -1)
dataset = np.hstack((dataset, var))
dataset = scaler.fit_transform(dataset)

for i in [25, 28, 35, 39, 40, 45, 46, 47, 60, 69, 98, 103, 116, 129, 131, 132, 133, 141, 143, 145, 147, 153, 157]:
    dd = dataset-dataset[i]
    dd = np.linalg.norm(dd, ord=2, axis=1)
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    plt.axis('equal')
    image = df[df['Parent\nCluster-ID'] == 'CLUSTER '+str(dd.argsort()[0]+1)]
    plt.scatter(image.Dist, image.Clock)
    a.set_title(str(dd.argsort()[0]+1))
    a = fig.add_subplot(2, 2, 2)
    plt.axis('equal')
    image = df[df['Parent\nCluster-ID'] == 'CLUSTER '+str(dd.argsort()[1]+1)]
    plt.scatter(image.Dist, image.Clock)
    a.set_title(str(dd[dd.argsort()[1]]))
    a = fig.add_subplot(2, 2, 3)
    plt.axis('equal')
    image = df[df['Parent\nCluster-ID'] == 'CLUSTER '+str(dd.argsort()[2]+1)]
    plt.scatter(image.Dist, image.Clock)
    a.set_title(str(dd[dd.argsort()[2]]))
    a = fig.add_subplot(2, 2, 4)
    plt.axis('equal')
    image = df[df['Parent\nCluster-ID'] == 'CLUSTER '+str(dd.argsort()[3]+1)]
    plt.scatter(image.Dist, image.Clock)
    a.set_title(str(dd[dd.argsort()[3]]))
