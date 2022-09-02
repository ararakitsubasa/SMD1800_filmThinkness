import matplotlib.pyplot as plt
import numpy as np


sample = 5
power = np.array([10000, 10000, 11000, 10000, 10000, 10000, 10000, 10000]) * sample

Sigma = 0.6

TM = np.array([[70.0, 70.0, 70.0, 70.0],
               [70.0, 70.0, 70.0, 70.0],
               [70.0, 70.0, 70.0, 70.0],
               [70.0, 70.0, 70.0, 70.0],
               [70.0, 70.0, 70.0, 70.0],
               [70.0, 70.0, 70.0, 70.0],
               [70.0, 70.0, 70.0, 70.0],
               [70.0, 70.0, 70.0, 70.0]]) * Sigma

#np.random.seed(19680801)
"""
M = 100
d = np.linspace(-5, 5, M)

magSwing = np.array([np.cos(7 * d), np.sin(d)])

s1 = np.dot(np.random.normal(0, TM, size=(power[], 2)), C) + magSwing

"""

C = np.array([[-1.0, 10.0], [1.0, 10.0]])

s11 = np.dot(np.random.normal(0, TM[0][0], size=(power[0], 2)), C)
s12 = np.dot(np.random.normal(0, TM[0][1], size=(power[0], 2)), C)+ np.array([90, 0])
s13 = np.dot(np.random.normal(0, TM[0][2], size=(power[0], 2)), C)+ np.array([0, 1500])
s14 = np.dot(np.random.normal(0, TM[0][3], size=(power[0], 2)), C)+ np.array([90, 1500])

s21 = np.dot(np.random.normal(0, TM[1][0], size=(power[1], 2)), C)+ np.array([210, 0])
s22 = np.dot(np.random.normal(0, TM[1][1], size=(power[1], 2)), C)+ np.array([210, 0])+ np.array([90, 0])
s23 = np.dot(np.random.normal(0, TM[1][2], size=(power[1], 2)), C)+ np.array([210, 0])+ np.array([0, 1500])
s24 = np.dot(np.random.normal(0, TM[1][3], size=(power[1], 2)), C)+ np.array([210, 0])+ np.array([90, 1500])

s31 = np.dot(np.random.normal(0, TM[2][0], size=(power[2], 2)), C)+ np.array([420, 0])
s32 = np.dot(np.random.normal(0, TM[2][1], size=(power[2], 2)), C)+ np.array([420, 0])+ np.array([90, 0])
s33 = np.dot(np.random.normal(0, TM[2][2], size=(power[2], 2)), C)+ np.array([420, 0])+ np.array([0, 1500])
s34 = np.dot(np.random.normal(0, TM[2][3], size=(power[2], 2)), C)+ np.array([420, 0])+ np.array([90, 1500])

s41 = np.dot(np.random.normal(0, TM[3][0], size=(power[3], 2)), C)+ np.array([630, 0])
s42 = np.dot(np.random.normal(0, TM[3][1], size=(power[3], 2)), C)+ np.array([630, 0])+ np.array([90, 0])
s43 = np.dot(np.random.normal(0, TM[3][2], size=(power[3], 2)), C)+ np.array([630, 0])+ np.array([0, 1500])
s44 = np.dot(np.random.normal(0, TM[3][3], size=(power[3], 2)), C)+ np.array([630, 0])+ np.array([90, 1500])

s51 = np.dot(np.random.normal(0, TM[4][0], size=(power[4], 2)), C)+ np.array([840, 0])
s52 = np.dot(np.random.normal(0, TM[4][1], size=(power[4], 2)), C)+ np.array([840, 0])+ np.array([90, 0])
s53 = np.dot(np.random.normal(0, TM[4][2], size=(power[4], 2)), C)+ np.array([840, 0])+ np.array([0, 1500])
s54 = np.dot(np.random.normal(0, TM[4][3], size=(power[4], 2)), C)+ np.array([840, 0])+ np.array([90, 1500])

s61 = np.dot(np.random.normal(0, TM[5][0], size=(power[5], 2)), C)+ np.array([1050, 0])
s62 = np.dot(np.random.normal(0, TM[5][1], size=(power[5], 2)), C)+ np.array([1050, 0])+ np.array([90, 0])
s63 = np.dot(np.random.normal(0, TM[5][2], size=(power[5], 2)), C)+ np.array([1050, 0])+ np.array([0, 1500])
s64 = np.dot(np.random.normal(0, TM[5][3], size=(power[5], 2)), C)+ np.array([1050, 0])+ np.array([90, 1500])

s71 = np.dot(np.random.normal(0, TM[6][0], size=(power[6], 2)), C)+ np.array([1260, 0])
s72 = np.dot(np.random.normal(0, TM[6][1], size=(power[6], 2)), C)+ np.array([1260, 0])+ np.array([90, 0])
s73 = np.dot(np.random.normal(0, TM[6][2], size=(power[6], 2)), C)+ np.array([1260, 0])+ np.array([0, 1500])
s74 = np.dot(np.random.normal(0, TM[6][3], size=(power[6], 2)), C)+ np.array([1260, 0])+ np.array([90, 1500])

s81 = np.dot(np.random.normal(0, TM[7][0], size=(power[7], 2)), C)+ np.array([1470, 0])
s82 = np.dot(np.random.normal(0, TM[7][1], size=(power[7], 2)), C)+ np.array([1470, 0])+ np.array([90, 0])
s83 = np.dot(np.random.normal(0, TM[7][2], size=(power[7], 2)), C)+ np.array([1470, 0])+ np.array([0, 1500])
s84 = np.dot(np.random.normal(0, TM[7][3], size=(power[7], 2)), C)+ np.array([1470, 0])+ np.array([90, 1500])

#data = np.random.random((2, 2, 2))
data = np.vstack([s11, s12, s13, s14, s21, s22, s23, s24, s31, s32, s33, s34, s41, s42, s43, s44, s51, s52, s53, s54, s61, s62, s63, s64, s71, s72, s73, s74, s81, s82, s83, s84])
   
#data3 = data[-1, :]
data = data.reshape((32, -1, 2))

#print(data.reshape(-1, 2))

data = data.reshape(-1, 2)

N = 100
t = np.linspace(-5, 5, N)

#targetSwing = np.array([10 * np.cos(3 * t), np.sin(t)])

toData = data

for j in range(len(t)):
    targetSwing = np.array([67.5 * np.cos(3 * t[j]), 35 * np.sin(t[j])])
    data2 = data + targetSwing
    data3 = np.vstack([data, data2])
    data3 = data3.reshape(2, -1, 2)
    data3 = data3[-1, :]
    toData = np.vstack([toData, data3])
    
#data = data[-1, :]
dataMap = np.histogram2d(toData[ :, 0], toData[ :, 1], bins=(np.arange(-200, 1800, 50), np.arange(-1000, 2500, 50)))

"""
fig, vax= plt.subplots(figsize=(6, 7))
vax.imshow(dataMap)
plt.show()
"""
print(dataMap)
#print(toData)
print("shape of data:")
print(dataMap[0])
print("shape##########")
print(dataMap[0].shape)
print("bins of data:")
print(dataMap[1])
print(dataMap[2])

print("filmMap:")

filmMap = dataMap[0][5:35, 18:50]
print(filmMap)
print("filmMap.shape")
print(filmMap.shape)

print("position of maximum:" + '', np.unravel_index(np.argmax(filmMap), filmMap.shape))
print("position of minimum:" + '', np.unravel_index(np.argmin(filmMap), filmMap.shape))

pMax = np.unravel_index(np.argmax(filmMap), filmMap.shape)
pMin = np.unravel_index(np.argmin(filmMap), filmMap.shape)

xScale = np.linspace(0, 1, 40)
yScale = np.linspace(0, 1, 70)

maxX = xScale[pMax[0] + 4]
maxY = yScale[pMax[1] + 18]

minX = xScale[pMin[0] + 4]
minY = yScale[pMin[1] + 18]

Uniformity = (np.max(filmMap)-np.min(filmMap))/(2 * np.mean(filmMap))
print("Uniformity:" + '', Uniformity)

textstr = '\n'.join((
r'$Uniformity=%.4f$' % (Uniformity, ),
r'$\sigma=%.4f$' % (Sigma, )))

fig, vax= plt.subplots(figsize=(6, 7))
hh = vax.hist2d(toData[ :, 0], toData[ :, 1], bins=(np.arange(-200, 1800, 50), np.arange(-1000, 2500, 50)))

vax.vlines([0, 1500], 0, 1, transform=vax.get_xaxis_transform(), colors='r')
vax.hlines([-175, 1675], 0, 1, transform=vax.get_yaxis_transform(), colors='r')
vax.axis("tight")
fig.colorbar(hh[3], ax=vax)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

vax.text(0.05, 0.95, textstr, transform=vax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

vax.text(maxX, maxY, "Max", transform=vax.transAxes, fontsize=7, verticalalignment='top', bbox=props)
vax.text(minX, minY, "Min", transform=vax.transAxes, fontsize=7, verticalalignment='top', bbox=props)

plt.show()


"""
fig, ax = plt.subplots()

for i in range(len(data)):
    ax.cla()
   # ax.imshow(data[i])
    ax.hist2d(data[i][ :, 0], data[i][ :, 1], bins=(np.arange(-10, 40, 0.5), np.arange(-80, 80, 0.5)))

    ax.set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
    plt.pause(1)
"""
