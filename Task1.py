import json
import math
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# grid 1km*1km
class Grid:
    def __init__(self, MaxBound):
        self.boundary = MaxBound # the boundary: topleft,bottomleft,topright,bottomright

    def createGrid(self):
        # this is a function to identify number of rows and columns and total grids

        # how many rows; the max row distance
        self.rows = max(int(np.ceil(self.computeDistance(self.boundary[0], self.boundary[1]))),int(np.ceil(self.computeDistance(self.boundary[2], self.boundary[3]))))
        print('number of rows is: ', self.rows)
        # how many columns; the max columns distance
        self.columns = max(int(np.ceil(self.computeDistance(self.boundary[0], self.boundary[2]))),int(np.ceil(self.computeDistance(self.boundary[1], self.boundary[3]))))
        print('number of columns is: ', self.columns)
        self.numofGrids = int(self.rows * self.columns)
        print('number of grids is: ', self.numofGrids)
        self.rowPoints = []
        self.colPoints = []
        self.lonOffset = (self.boundary[3][1] - self.boundary[1][1])/self.columns
        self.latOffset = (self.boundary[0][0] - self.boundary[1][0])/self.rows
        for i in range(self.rows):
            self.rowPoints.append(51.261318 + i*self.latOffset)
        for j in range(self.columns):
            self.colPoints.append(-0.563 + j*self.lonOffset) #0.0143
        return(self.colPoints, self.rowPoints, self.latOffset, self.lonOffset)

    def computeDistance(self, coor1, coor2):
        # Haversine Formula
        # hav(theta) = sin(theta / 2)^2 = (1-cos(theta))/2
        # haversine: a = sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlong / 2)^2
        # formula: c = 2 * atan2(sqrt(a), sqrt(1-a))
        #          d = R * c
        lat1, long1, lat2, long2 = coor1[0], coor1[1], coor2[0], coor2[1]
        R = 6371.0
        phi1 = lat1 * (math.pi / 180)
        phi2 = lat2 * (math.pi / 180)
        # Delta phi
        delta1 = (lat1 - lat2) * (math.pi / 180)
        # Delta Lambda
        delta2 = (long1 - long2) * (math.pi / 180)

        a = math.sin(delta1 / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta2 / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = R * c
        return d


MaxBound = [-0.563,51.261318,0.28036,51.686031]
Mat_MaxBound = [[MaxBound[3],MaxBound[0]], [MaxBound[1],MaxBound[0]], [MaxBound[3],MaxBound[2]],[MaxBound[1],MaxBound[2]]] # topleft,downleft,topright,downright
km_MaxBound = Grid(Mat_MaxBound)
x,y, latOffset, longOffset = km_MaxBound.createGrid()

if __name__ == '__main__':
    ## loading data geoLondonJan
    cor_geoLondonJan = []
    for line in open('geoLondonJan', 'r', encoding="utf-8"):
        line_json = json.loads(line)
        cor_geoLondonJan.append(line_json['coordinates']['coordinates'])
    # print(cor_geoLondonJan)

    ## create dataframe
    dict_coor  = {}

    for coor in cor_geoLondonJan:
        row = np.floor((coor[1] - 51.261318)/latOffset)
        col = np.floor((coor[0] + 0.563)/longOffset)
        dict_coor[(row,col)] = dict_coor.get((row,col), 0) + 1

    print(max(dict_coor))

    df_coor = pd.DataFrame(0,columns=list(range(59)), index=list(range(48)))
    for key,value in dict_coor.items():
        df_coor.iloc[int(key[0]),int(key[1])] = value

    ## heatmap
    sns.set_context({"figure.figsize":(10,8)})
    sns.heatmap(data=df_coor,square=True, cmap="Oranges", vmax = 50)
    plt.title("Distribution of tweets location(max = 50)")
    plt.savefig('T1_heatmap.png')
    plt.show()
    sns.set_context({"figure.figsize":(10,8)})
    sns.heatmap(data=df_coor,square=True, cmap="Oranges")
    plt.title("Distribution of tweets location")
    plt.savefig('T1_heatmap_origin.png')
    plt.show()

    ## histogram
    fig, axis = plt.subplots(figsize =(10, 5))
    axis.hist(list(dict_coor.values()), bins = list(range(0,1353,5)))
    plt.xlabel("Number of tweets")
    plt.ylabel("Number of grids")
    plt.title("Distribution of number")
    plt.savefig('T1_hist.png')
    plt.show()