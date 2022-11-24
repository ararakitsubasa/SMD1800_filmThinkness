import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import argparse

parser = argparse.ArgumentParser()
 
parser.add_argument("-s", "--sheet", metavar="sheet name", help = "Path to your sheet")
#parser.add_argument("-3d", "--3d", metavar="3D plot", help = "type of the plot (default is 2d)")
args = vars(parser.parse_args())

#print(args)

if args["sheet"]:
    print("read sheet" + args["sheet"])
else:
    print('Catching an argumentError')


file = pd.ExcelFile("./20220927-8_Thickness_and_Rs_Result_ITO_100nm_.xlsx")
#data = pd.read_excel(file, "Rs(As Depo)分布結果", usecols="E:AC")
dataRaw = pd.read_excel(file, "Rs(As Depo)分布結果")
#dataRaw = pd.read_excel(file, "%d", args["sheet"] )

data = dataRaw.to_numpy()
print(data.shape)
print(data)
a = data.shape
print(a)
print(type(a))
row = a[0]
print(row)
cols =a[1]
print(cols)
#a = np.array(data)
#data = a.astype(np.float64)
"""
find 0

"""
table = np.array([[[0,0], [0, 0], [0, 0]]])

for i in range(row):
    for j in range(cols):
        if data[i][j] == 0:
 #          print("find 0 in data")
  #          print("row:%d col:%d" %(i, j) )
            for r in range(row):
                if data[i][r] == "max":
  #                  print("find max in row")
  #                  print("row:%d col:%d" %(i, r) )
                    for c in range(row):
                        if data[c][j] == "max":
 #                           print("find max in cols")
  #                          print("row:%d col:%d" %(c, j))
  #                          print("--------------------")
   #                         print("find a distribution from:")
   #                         print("0:(%d, %d) max_row:(%d, %d) max_col:(%d, %d)" %(i, j, i, r, c, j))
                            table = np.append(table,[[[i, j], [i, r], [c, j]]], axis=0)
                            print(table)
                        else:
                            continue
print('table: -----------------------------------------')
print(table)
print(table.shape)
print(type(table.shape))
print(table.shape[0])

#datalist = np.array([])
#n = 1
for n in range(table.shape[0]-1):
    try:
        datalist = data[table[n][0][0]+1:table[n][2][0], table[n][0][1]+1:table[n][1][1]]
        print(datalist)
        print(type(datalist[0][0]))
        print("-----------------------------")
        print(datalist.shape)
        print(datalist.shape[0])
        print(datalist.shape[1])
        datalist = datalist.astype(np.float64)
        fig, ax = plt.subplots(figsize=(6, 7))
        im = ax.imshow(datalist)
        for i in range(datalist.shape[0]):
            for j in range(datalist.shape[1]):
                text = ax.text(j, i, datalist[i, j], ha="center", va="center", color="w", fontsize=6)
        fig.colorbar(im, ax=ax)

    except:
        print("no table found, to next")
    else:
        print("got table!")

fig.tight_layout()
plt.show()


"""
        for i in range([table[n][0][0]+1 - table[n][2][0])):
            for j in range(table[n][0][1]+1 - table[n][1][1]):
                text = ax.text(j, i, datalist[i, j], ha="center", va="center", color="w")
"""