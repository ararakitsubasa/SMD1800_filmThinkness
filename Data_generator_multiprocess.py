import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import sys
from numpy import random

import multiprocessing
import time
#import random

def task(ru):
  #  ru = int(sys.argv[1])

    print(ru)

    Sigma = random.choice([0.3, 0.5, 0.7, 0.9])


    sample = 100
    power = np.array([100, 100, 100, 100, 100, 100, 100, 100]) * sample

    tsSigma = 1

    TS = np.array([70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0]) * tsSigma

    #Shunt

    shuntArray_top = np.ones((8, 9))

    shuntArray_down = np.ones((8, 9))

    sampleN = 100
    power_shunt = np.ones((8, 19)) * sampleN

    #T/S
    C = np.array([[-1.0, 1.0], [1.0, 1.0]])

    s1 = np.dot(np.random.normal(0, TS[0], size=(power[0], 2)), C)
    s2 = np.dot(np.random.normal(0, TS[0], size=(power[1], 2)), C)+ np.array([210, 0])
    s3 = np.dot(np.random.normal(0, TS[0], size=(power[2], 2)), C)+ np.array([420, 0])
    s4 = np.dot(np.random.normal(0, TS[0], size=(power[3], 2)), C)+ np.array([630, 0])
    s5 = np.dot(np.random.normal(0, TS[0], size=(power[4], 2)), C)+ np.array([840, 0])
    s6 = np.dot(np.random.normal(0, TS[0], size=(power[5], 2)), C)+ np.array([1050, 0])
    s7 = np.dot(np.random.normal(0, TS[0], size=(power[6], 2)), C)+ np.array([1260, 0])
    s8 = np.dot(np.random.normal(0, TS[0], size=(power[7], 2)), C)+ np.array([1470, 0])

    #Sigma = 0.6

    TM = np.array([[70.0, 70.0, 70.0, 70.0],
                [70.0, 70.0, 70.0, 70.0],
                [70.0, 70.0, 70.0, 70.0],
                [70.0, 70.0, 70.0, 70.0],
                [70.0, 70.0, 70.0, 70.0],
                [70.0, 70.0, 70.0, 70.0],
                [70.0, 70.0, 70.0, 70.0],
                [70.0, 70.0, 70.0, 70.0]]) * Sigma

    sample2 = 1000
    power2 = np.array([10, 10, 10, 10, 10, 10, 10, 10]) * sample2
    #T/M
    c1 = np.array([[-1.0, 10.0], [1.0, 10.0]])

    s11 = np.dot(np.random.normal(0, TM[0][0], size=(power2[0], 2)), c1)+ np.array([0,-750])
    s12 = np.dot(np.random.normal(0, TM[0][1], size=(power2[0], 2)), c1)+ np.array([90, 0])+ np.array([0,-750])
    s13 = np.dot(np.random.normal(0, TM[0][2], size=(power2[0], 2)), c1)+ np.array([0, 1500])+ np.array([0,-750])
    s14 = np.dot(np.random.normal(0, TM[0][3], size=(power2[0], 2)), c1)+ np.array([90, 1500])+ np.array([0,-750])

    s21 = np.dot(np.random.normal(0, TM[1][0], size=(power2[1], 2)), c1)+ np.array([210, 0])+ np.array([0,-750])
    s22 = np.dot(np.random.normal(0, TM[1][1], size=(power2[1], 2)), c1)+ np.array([210, 0])+ np.array([90, 0])+ np.array([0,-750])
    s23 = np.dot(np.random.normal(0, TM[1][2], size=(power2[1], 2)), c1)+ np.array([210, 0])+ np.array([0, 1500])+ np.array([0,-750])
    s24 = np.dot(np.random.normal(0, TM[1][3], size=(power2[1], 2)), c1)+ np.array([210, 0])+ np.array([90, 1500])+ np.array([0,-750])

    s31 = np.dot(np.random.normal(0, TM[2][0], size=(power2[2], 2)), c1)+ np.array([420, 0])+ np.array([0,-750])
    s32 = np.dot(np.random.normal(0, TM[2][1], size=(power2[2], 2)), c1)+ np.array([420, 0])+ np.array([90, 0])+ np.array([0,-750])
    s33 = np.dot(np.random.normal(0, TM[2][2], size=(power2[2], 2)), c1)+ np.array([420, 0])+ np.array([0, 1500])+ np.array([0,-750])
    s34 = np.dot(np.random.normal(0, TM[2][3], size=(power2[2], 2)), c1)+ np.array([420, 0])+ np.array([90, 1500])+ np.array([0,-750])

    s41 = np.dot(np.random.normal(0, TM[3][0], size=(power2[3], 2)), c1)+ np.array([630, 0])+ np.array([0,-750])
    s42 = np.dot(np.random.normal(0, TM[3][1], size=(power2[3], 2)), c1)+ np.array([630, 0])+ np.array([90, 0])+ np.array([0,-750])
    s43 = np.dot(np.random.normal(0, TM[3][2], size=(power2[3], 2)), c1)+ np.array([630, 0])+ np.array([0, 1500])+ np.array([0,-750])
    s44 = np.dot(np.random.normal(0, TM[3][3], size=(power2[3], 2)), c1)+ np.array([630, 0])+ np.array([90, 1500])+ np.array([0,-750])

    s51 = np.dot(np.random.normal(0, TM[4][0], size=(power2[4], 2)), c1)+ np.array([840, 0])+ np.array([0,-750])
    s52 = np.dot(np.random.normal(0, TM[4][1], size=(power2[4], 2)), c1)+ np.array([840, 0])+ np.array([90, 0])+ np.array([0,-750])
    s53 = np.dot(np.random.normal(0, TM[4][2], size=(power2[4], 2)), c1)+ np.array([840, 0])+ np.array([0, 1500])+ np.array([0,-750])
    s54 = np.dot(np.random.normal(0, TM[4][3], size=(power2[4], 2)), c1)+ np.array([840, 0])+ np.array([90, 1500])+ np.array([0,-750])

    s61 = np.dot(np.random.normal(0, TM[5][0], size=(power2[5], 2)), c1)+ np.array([1050, 0])+ np.array([0,-750])
    s62 = np.dot(np.random.normal(0, TM[5][1], size=(power2[5], 2)), c1)+ np.array([1050, 0])+ np.array([90, 0])+ np.array([0,-750])
    s63 = np.dot(np.random.normal(0, TM[5][2], size=(power2[5], 2)), c1)+ np.array([1050, 0])+ np.array([0, 1500])+ np.array([0,-750])
    s64 = np.dot(np.random.normal(0, TM[5][3], size=(power2[5], 2)), c1)+ np.array([1050, 0])+ np.array([90, 1500])+ np.array([0,-750])

    s71 = np.dot(np.random.normal(0, TM[6][0], size=(power2[6], 2)), c1)+ np.array([1260, 0])+ np.array([0,-750])
    s72 = np.dot(np.random.normal(0, TM[6][1], size=(power2[6], 2)), c1)+ np.array([1260, 0])+ np.array([90, 0])+ np.array([0,-750])
    s73 = np.dot(np.random.normal(0, TM[6][2], size=(power2[6], 2)), c1)+ np.array([1260, 0])+ np.array([0, 1500])+ np.array([0,-750])
    s74 = np.dot(np.random.normal(0, TM[6][3], size=(power2[6], 2)), c1)+ np.array([1260, 0])+ np.array([90, 1500])+ np.array([0,-750])

    s81 = np.dot(np.random.normal(0, TM[7][0], size=(power2[7], 2)), c1)+ np.array([1470, 0])+ np.array([0,-750])
    s82 = np.dot(np.random.normal(0, TM[7][1], size=(power2[7], 2)), c1)+ np.array([1470, 0])+ np.array([90, 0])+ np.array([0,-750])
    s83 = np.dot(np.random.normal(0, TM[7][2], size=(power2[7], 2)), c1)+ np.array([1470, 0])+ np.array([0, 1500])+ np.array([0,-750])
    s84 = np.dot(np.random.normal(0, TM[7][3], size=(power2[7], 2)), c1)+ np.array([1470, 0])+ np.array([90, 1500])+ np.array([0,-750])

    data = np.vstack([s1, s2, s3, s4, s5, s6, s7, s8])
    print(data.shape)
    data = data.reshape(-1, 2)

    toData = data

    #target geometry
    N = 20
    t = np.linspace(0,  np.pi, N)
    r = 50
    x, y = r * np.cos(t), r * np.sin(t)+800
    #xv, yv = np.meshgrid(x, y, sparse=True)

    b = np.linspace(np.pi, 2 * np.pi, N)
    R = 50
    X, Y = R * np.cos(b), R * np.sin(b)-800

    n = 200
    p = np.linspace(-800, 800, n)
    x1, y1 = np.ones(n)*50, p
    x2, y2 = np.ones(n)*-50, p

    for i in range(len(t)):
        data2 = data + np.array([x[i], y[i]])
        data3 = np.vstack([data, data2])
        data3 = data3.reshape(2, -1, 2)
        data3 = data3[-1, :]
        toData = np.vstack([toData, data3])

    for j in range(len(b)):
        data2 = data + np.array([X[j], Y[j]])
        data3 = np.vstack([data, data2])
        data3 = data3.reshape(2, -1, 2)
        data3 = data3[-1, :]
        toData = np.vstack([toData, data3])
        
    for k in range(len(p)):
        data2 = data + np.array([x1[k], y1[k]])
        data3 = np.vstack([data, data2])
        data3 = data3.reshape(2, -1, 2)
        data3 = data3[-1, :]
        toData = np.vstack([toData, data3])
        
    for l in range(len(p)):
        data2 = data + np.array([x2[l], y2[l]])
        data3 = np.vstack([data, data2])
        data3 = data3.reshape(2, -1, 2)
        data3 = data3[-1, :]
        toData = np.vstack([toData, data3])

    tagData = toData
    tagtoData = tagData

    tSwingStep = 10
    tSwingPhi = np.linspace(0, 4*np.pi, tSwingStep)

    #targetSwing = np.array([10 * np.cos(3 * t), np.sin(t)])

    for q in range(len(tSwingPhi)):
    #   targetSwing = np.array([67.5 * np.cos(3 * tSwingPhi[q]), 35 * np.sin(tSwingPhi[q])])
        data2 = tagData + np.array([67.5 * np.cos(3 * tSwingPhi[q]), 35 * np.sin(tSwingPhi[q])])
        data3 = np.vstack([tagData, data2])
        data3 = data3.reshape(2, -1, 2)
        data3 = data3[-1, :]
        tagtoData = np.vstack([tagtoData, data3])
        

    # magnet data
    magData = np.vstack([s11, s12, s13, s14, s21, s22, s23, s24, s31, s32, s33, s34, s41, s42, s43, s44, s51, s52, s53, s54, s61, s62, s63, s64, s71, s72, s73, s74, s81, s82, s83, s84])
    
    #print(data.reshape(-1, 2))
    magData = magData.reshape(-1, 2)

    magN = 100
    magt = np.linspace(-5, 5, magN)

    #targetSwing = np.array([10 * np.cos(3 * t), np.sin(t)])

    magtoData = magData
    # magnet Swing
    for m in range(len(magt)):
    #    targetSwing = np.array([67.5 * np.cos(3 * magt[m]), 35 * np.sin(magt[m])])
        data2 = magData + np.array([67.5 * np.cos(3 * magt[m]), 35 * np.sin(magt[m])])
        data3 = np.vstack([magData, data2])
        data3 = data3.reshape(2, -1, 2)
        data3 = data3[-1, :]
        magtoData = np.vstack([magtoData, data3])


    print("shape of tagtoData: " + '', tagtoData.shape)
    print("shape of magtoData: " + '', magtoData.shape)

    allData = np.vstack([tagtoData, magtoData])
    print("shape of allData: " + '', allData.shape)

    allData = allData.reshape(-1, 2)

    fig, vax= plt.subplots(figsize=(6, 7))
    hh = vax.hist2d(allData[ :, 0], allData[ :, 1], bins=(np.arange(-250, 1850, 70), np.arange(-1050, 1050, 70)))

    print(hh)
    print("shape of hist:"+ '', hh[0].shape)

    filmMap = hh[0]
    maxpoint = np.max(filmMap)
    print("maxpoint:" + '', maxpoint)

    #ru = 3
    run = 'arr_' + str(ru)

    data_file = Path("/mnt/k/SMD1800_simulation/Data_generate/histfile.npz")

    if data_file.exists():

        data = np.load("histfile.npz")
        data = dict(data)
        data[str(run)] = hh[0]
        np.savez("histfile",**data)
        datas = np.load('histfile.npz')
        print(datas.files)
    #   print(datas[run])
        
    else:
        np.savez('histfile.npz', hh[0])
        datas = np.load('histfile.npz')
        print(datas.files)
    #   print(datas['arr_0'])

    label_file = Path("/mnt/k/SMD1800_simulation/Data_generate/labelfile.npz")

    if label_file.exists():

        data = np.load("labelfile.npz")
        data = dict(data)
        data[str(run)] = Sigma
        np.savez("labelfile",**data)
        datas = np.load('labelfile.npz')
        print(datas.files)
        print(datas[run])
        
    else:
        np.savez('labelfile.npz', Sigma)
        datas = np.load('labelfile.npz')
        print(datas.files)
        print(datas['arr_0'])

    plt.show()

    #run = 2
    plt.savefig('run' + str(ru) + '.png')
   # return 0

def calculate(func, args):
    return '%s says that %s' % (
        multiprocessing.current_process().name,
        func.__name__
        )

def calculatestar(args):
    return calculate(*args)


def test():
    PROCESSES = 4
    print('Creating pool with %d processes\n' % PROCESSES)

    with multiprocessing.Pool(PROCESSES) as pool:
        #
        # Tests
        #

        TASKS = [(task, i) for i in range(10)]

        results = [pool.apply_async(calculate, t) for t in TASKS]
      #  imap_it = pool.imap(calculatestar, TASKS)
     #   imap_unordered_it = pool.imap_unordered(calculatestar, TASKS)

        print('Ordered results using pool.apply_async():')
        for r in results:
            print('\t', r.get())
        print() 

       # print('Ordered results using pool.imap():')
      #  for x in imap_it:
      #      print('\t', x)
     #   print()

                # Test error handling
        #
        #
        # Testing timeouts
        #

        print('Testing ApplyResult.get() with timeout:', end=' ')
        res = pool.apply_async(calculate, TASKS[0])
        while 1:
            sys.stdout.flush()
            try:
                sys.stdout.write('\n\t%s' % res.get(0.02))
                break
            except multiprocessing.TimeoutError:
                sys.stdout.write('.')
        print()
        print()

        print('Testing IMapIterator.next() with timeout:', end=' ')
        it = pool.imap(calculatestar, TASKS)
        while 1:
            sys.stdout.flush()
            try:
                sys.stdout.write('\n\t%s' % it.next(0.02))
            except StopIteration:
                break
            except multiprocessing.TimeoutError:
                sys.stdout.write('.')
        print()
        print()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    test()

