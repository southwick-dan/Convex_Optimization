#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import sys
import os.path

opt = input("Please input the name of data whose solution\n you want to check ('g' for 'gisette', 's' for 'spamData') :\n")

if opt == 'g':
    data_n = 'gisette'
elif opt == 's':
    data_n = 'spamData'
else:
    print("=====================")
    print("No file chosen. Exit.")
    print("=====================")
    exit(0)

solver = input("Please input the solver type('gd' or 'newton' or 'cgd'):\n")

epsilon = input("Please input the epsilon (0.01 or 0.0001 or 0.000001):\n")
epsilon = float(epsilon)

sol_path = '%s_%s_solution_epsilon_%f.dat'%(data_n,solver,epsilon)
if os.path.isfile(sol_path) == False:
    print("No solution file exits. Exit!\n")
    exit(0)

data_path = './%s.mat'%data_n
if os.path.isfile(data_path) == False:
    print("No data file exits. Exit!\n")
    exit(0)

data = sio.loadmat(data_path)
xtest = data['Xtest']
ytest = data['ytest']

N,M = np.shape(xtest)

sol = np.loadtxt(sol_path)
w = sol[0:-1]
b = sol[-1]

count = 0
for i in range(N):
    if (b+np.dot(w,xtest[i,:]))*ytest[i]>0:
        count += 1

print()
print("======================================")
print("File name: %s.mat"%data_n)
print("Method: %s"%solver)
print("Epsilon: %f"%epsilon)
print("Out of %d data points, %d are correct."%(N,count))
print("Accuracy = %.2f%%."%(100*count/N))
print("======================================")
print()
