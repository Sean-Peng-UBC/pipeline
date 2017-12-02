#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


def create_dataset():
    xl = pd.ExcelFile('CDG.xlsx')
    df = xl.parse(xl.sheet_names[0], skiprows=11)
    df = df[df.Identifier0 == 'MELO-ANOM']
    df = df.loc[:, ['Number', 'Clock', 'Dist', 'Length', 'Width', 'Joint\nLength', 'Peak\nDepth']]
    df.Dist = df.Dist * 1000 + 500
    df.Number = df.Number/10
    df.Length = df.Length/2
    df.Width = df.Width/2
    df['Joint\nLength'] = df['Joint\nLength'] * 1000
    for x in df.index:
        df.loc[x, 'Clock'] = (df.loc[x, 'Clock'].hour * 30 + df.loc[x, 'Clock'].minute * 0.5) / 360 * 914.4 * 3.14 + 500
    df = df.astype('uint16', copy=False)

    temp = np.zeros((0, 500*500), dtype=np.uint8)
    dataset = csr_matrix(temp)
    spatemp = csr_matrix(temp)
    gp = df.groupby('Number')
    j = 0
    for value in gp.indices.itervalues():
        data = df.iloc[value]
        length = data.iloc[0]['Joint\nLength']
        segment = np.zeros([2872 + 1000, length + 1000], dtype=np.uint8)
        for i in range(0, len(data)):
            corrosion = data.iloc[i]
            segment[corrosion.Clock - corrosion.Width:corrosion.Clock + corrosion.Width,
            corrosion.Dist - corrosion.Length:corrosion.Dist + corrosion.Length] = corrosion['Peak\nDepth']
        bound = segment[:1000]
        segment[:1000] += segment[-1000:]
        segment[-1000:] += bound

        for i in range(0, len(data)):
            corrosion = data.iloc[i]
            region = csr_matrix(segment[corrosion.Clock - 250:corrosion.Clock + 250,
            corrosion.Dist - 250:corrosion.Dist + 250].reshape(-1,))
            spatemp = sparse.vstack((spatemp, region))
            j += 1
            if j%5000 == 0:
                dataset = sparse.vstack((dataset, spatemp))
                spatemp = csr_matrix(temp)

    dataset = sparse.vstack((dataset, spatemp))

    sparse.save_npz('dataset.npz', dataset)


def eigenface():
    from sklearn.decomposition import TruncatedSVD

    trainset = sparse.load_npz('trainset.npz')
    svd = TruncatedSVD(n_components=4000, n_iter=10)
    svd.fit(trainset)
    components = svd.components_
    ratio = svd.explained_variance_ratio_

    np.save('train_components.npy', components)
    np.save('train_ratio.npy', ratio)


def match():
    from matplotlib import pyplot as plt

    output = np.load('output.npy')
    testoutput = np.load('testoutput.npy')
    components = np.load('component.npy')
    ratio = np.load('ratio.npy')
    dataset = sparse.load_npz('dataset.npz')
    testset = sparse.load_npz('test.npz')

    i = 0
    test = testoutput[i]
    bias = np.absolute(output - test)
    bias = bias.dot(ratio)
    argmin = bias.argmin()

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(testset[i].toarray().reshape(500, 500))
    a.set_title('Input '+str(i))
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(dataset[argmin].toarray().reshape(500, 500))
    a.set_title('Retrieval '+str(argmin))
    a = fig.add_subplot(2, 2, 3)
    plt.imshow(testoutput[i].dot(components).reshape(500, 500))
    #a.set_title('Input parameterization')
    a = fig.add_subplot(2, 2, 4)
    plt.imshow(output[argmin].dot(components).reshape(500, 500))
    #a.set_title('Retrieval parameterization')


def main():
    create_dataset()


if __name__ == '__main__':
    main()