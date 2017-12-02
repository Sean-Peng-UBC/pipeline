#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import sys
import pandas as pd
import numpy as np
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D


class Example(QtGui.QMainWindow):
    def __init__(self):
        super(Example, self).__init__()
        self.set_data()
        self.initUI()

    def set_data(self):
        xl1 = pd.ExcelFile('AFD.xlsx')
        df1 = xl1.parse(xl1.sheet_names[0], skiprows=11)
        df1 = df1[df1.Identifier0 == 'metal loss-corrosion']
        df1 = df1.loc[:, ['Number', 'Clock', 'Dist', 'Length', 'Width', 'AvgDepth', 'Peak\nDepth']]
        df1.Dist = df1.Dist*1000
        for x in df1.index:
            df1.loc[x, 'Clock'] = (df1.loc[x, 'Clock'].hour * 30 + df1.loc[x, 'Clock'].minute * 0.5) / 360 * 914.4 * 3.14
        self._data1 = df1

        xl2 = pd.ExcelFile('CDG.xlsx')
        df2 = xl2.parse(xl2.sheet_names[0], skiprows=11)
        df2 = df2[df2.Identifier0 == 'MELO-ANOM']
        df2 = df2.loc[:, ['Number', 'Clock', 'Dist', 'Length', 'Width', 'AvgDepth', 'Peak\nDepth']]
        df2.Dist = df2.Dist*1000
        for x in df2.index:
            df2.loc[x, 'Clock'] = (df2.loc[x, 'Clock'].hour * 30 + df2.loc[x, 'Clock'].minute * 0.5) / 360 * 914.4 * 3.14
        self._data2 = df2

    def initUI(self):
        btn1 = QtGui.QPushButton("Location", self)
        btn1.move(30, 20)
        btn1.clicked.connect(self.buttonClicked1)

        btn2 = QtGui.QPushButton("Depth", self)
        btn2.move(120, 20)
        btn2.clicked.connect(self.buttonClicked2)

        btn3 = QtGui.QPushButton("Size", self)
        btn3.move(30, 100)
        btn3.clicked.connect(self.buttonClicked3)

        btn4 = QtGui.QPushButton("Distribution", self)
        btn4.move(120, 100)
        btn4.clicked.connect(self.buttonClicked4)

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('Data Visualization')
        self.show()

    def buttonClicked1(self):
        data1 = self._data1.copy()
        data2 = self._data2.copy()
        text, ok = QtGui.QInputDialog.getText(self, 'Visualization', 'Corrosion:')
        if ok & (text != ''):
            pattern = re.compile(r'\d+')
            number = int(re.findall(pattern, text)[0])
            segment_number = data1.iloc[number].Number
            data1 = data1[data1.Number == segment_number]
            data2 = data2[data2.Number == segment_number]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.axis('equal')
            ax.set_xlim([0, 13000])
            ax.set_xlabel('Distance[mm]')
            ax.set_ylabel('Clock Position[mm]')

            for x in xrange(0, len(data1.index)):
                data11 = data1.iloc[x]
                rect = patches.Rectangle((data11.Dist, data11.Clock),
                                         data11.Length, data11.Width, color='b')
                ax.add_patch(rect)
            for x in xrange(0, len(data2.index)):
                data21 = data2.iloc[x]
                rect = patches.Rectangle((data21.Dist, data21.Clock),
                                         data21.Length, data21.Width, color='r')
                ax.add_patch(rect)
            fig.savefig('The ' + str(segment_number) + ' segment picture1.png', dpi=460, bbox_inches='tight')
            fig.show()

    def buttonClicked2(self):
        data1 = self._data1.copy()
        data2 = self._data2.copy()

        com = []

        for x in data1.index:
            data11 = data1.loc[x]
            segment_number = data11.Number
            data21 = data2[data2.Number == segment_number]
            data21 = data21[data21.Dist < data11.Dist + 1.5 * data11.Length]
            data21 = data21[data21.Dist > data11.Dist - 1.5 * data11.Length]
            data21 = data21[data21.Clock < data11.Clock + 1.5 * data11.Width]
            data21 = data21[data21.Clock > data11.Clock - 1.5 * data11.Width]
            for i in range(0, len(data21)):
                if abs(data21.iloc[i].AvgDepth - data11.AvgDepth) < 4:
                    temp = np.vstack((data11, data21.iloc[i]))
                    com.append(temp)

        temp = [x[0] for x in com]
        temp = np.asarray(temp)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []

        for i in range(0, 100):
            data11 = temp[temp[:, 3] == i]
            for j in range(0, 100):
                data12 = data11[data11[:, 4] == j]
                if len(data12) > 4:
                    x.append(i)
                    y.append(j)
                    z.append(len(data12)*1.0/len(temp))

        dx = 0.5 * np.ones_like(x)
        dy = 0.5 * np.ones_like(y)
        dz = z
        z = np.zeros_like(z)

        ax.bar3d(x, y, z, dx, dy, dz, color='r')

        ax.set_xlabel('Length(mm)')
        ax.set_ylabel('Width(mm)')
        ax.set_zlabel('Amplitude')
        plt.show()

                # text, ok = QtGui.QInputDialog.getText(self, 'Visualization', 'Corrosion:')
        # if ok & (text != ''):
        #     pattern = re.compile(r'\d+')
        #     number = int(re.findall(pattern, text)[0])
        #     segment_number = data1.iloc[number].Number
        #     data1 = data1[data1.Number == segment_number]
        #     data2 = data2[data2.Number == segment_number]
        #
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     plt.axis('equal')
        #     ax.set_xlim([0, 12])
        #     ax.set_ylim([0, 4])
        #     ax.set_xlabel('Distance[m]')
        #     ax.set_ylabel('Clock Position[m]')
        #     min_val = 0
        #     max_val = 8
        #     my_cmap = cm.get_cmap('jet')  # return the colormap
        #     norm = colors.Normalize(min_val, max_val)
        #
        #     for x in xrange(0, len(data1.index)):
        #         rect = patches.Rectangle((data1.iloc[x].Dist, (
        #         data1.iloc[x].Clock.hour * 30 + data1.iloc[x].Clock.minute * 0.5) / 360 * 0.9144 * 3.14),
        #                                  data1.iloc[x].Length * 0.001, data1.iloc[x].Width * 0.001,
        #                                  color=my_cmap(norm(data1.iloc[x].AvgDepth)))
        #         ax.add_patch(rect)
        #     for x in xrange(0, len(data2.index)):
        #         rect = patches.Rectangle((data2.iloc[x].Dist, (
        #         data2.iloc[x].Clock.hour * 30 + data2.iloc[x].Clock.minute * 0.5) / 360 * 0.9144 * 3.14),
        #                                  data2.iloc[x].Length * 0.001, data2.iloc[x].Width * 0.001,
        #                                  color=my_cmap(norm(data2.iloc[x].AvgDepth)))
        #         ax.add_patch(rect)
        #     # cb1 = colorbar.ColorbarBase(ax=ax.get_xlim, cmap=my_cmap, norm=norm, orientation='horizontal')
        #     ax.set_label('Avg.Depth[%]')
        #     fig.savefig('The ' + str(segment_number) + ' segment picture2.png', dpi=460, bbox_inches='tight')
        #     fig.show()

    def buttonClicked3(self):
        data1 = self._data1.copy()
        data2 = self._data2.copy()

        data1 = data1[data1.Width < 100]
        data2 = data2[data2.Width < 100]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []
        color = []

        for i in range(0, 100):
            data11 = data1[data1.Length == i]
            data21 = data2[data2.Length == i]
            for j in range(0, 100):
                data12 = data11[data11.Width == j]
                data22 = data21[data21.Width == j]
                if len(data12) > 4:
                    x.append(i)
                    y.append(j)
                    z.append(len(data12)*1.0/len(data1))
                    color.append('b')
                if len(data22) > 4:
                    x.append(i+0.5)
                    y.append(j+0.5)
                    z.append(len(data22)*1.0/len(data2))
                    color.append('r')

        dx = 0.5 * np.ones_like(x)
        dy = 0.5 * np.ones_like(y)
        dz = z
        z = np.zeros_like(z)

        ax.bar3d(x, y, z, dx, dy, dz, color=color)

        ax.set_xlabel('Length(mm)')
        ax.set_ylabel('Width(mm)')
        ax.set_zlabel('Amplitude')
        plt.show()

                # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # # plt.axis('equal')
        # ax.set_xlim([0, 100])
        # ax.set_ylim([0, 100])
        # ax.set_xlabel('Length[mm]')
        # ax.set_ylabel('Width[mm]')
        #
        # plt.scatter(data2.Length, data2.Width, color='b', s=0.1)
        # plt.scatter(data1.Length, data1.Width, color='r', s=0.1, marker='x')
        # fig.savefig('Size.png', dpi=460, bbox_inches='tight')
        # fig.show()

    def buttonClicked4(self):
        data1 = self._data1.copy()
        data2 = self._data2.copy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0, 120])
        ax.set_xlabel('Length[mm]')
        ax.set_ylabel('Number')

        Numx0 = []
        Numy0 = []
        Numx1 = []
        Numy1 = []

        for i in xrange(0, 120):
            Numx0.append(1.0*len(data1[data1.Length == i])/len(data1))
            Numy0.append(1.0*len(data2[data2.Length == i])/len(data2))
            Numx1.append(1.0*len(data1[data1.Width == i])/len(data1))
            Numy1.append(1.0*len(data2[data2.Width == i])/len(data2))

        plt.scatter(range(0, len(Numx0)), Numx0, color='b', s=1)
        plt.scatter(range(0, len(Numy0)), Numy0, color='r', s=1)
        plt.scatter(range(0, len(Numx1)), Numx1, color='b', s=1, marker='x')
        plt.scatter(range(0, len(Numy1)), Numy1, color='r', s=1, marker='x')

        fig.savefig('Distribution.png', dpi=460, bbox_inches='tight')
        fig.show()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()