#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import Functions.Functions as Functions
import time,os


print()
opt = input("Please input the data file ('g' for 'gisette', 's' for 'spamData') :\n")

if opt == 'g':
    data_n = 'gisette'
    stepsize = 0.5
    criterion = 0.001
elif opt == 's':
    data_n = 'spamData'
    stepsize = 0.001
    criterion = 0.00001
else:
    print("=====================")
    print("No file chosen. Exit.")
    print("=====================")
    exit(0)

data_path = './%s.mat'%data_n
if os.path.isfile(data_path) == False:
    print("No data file exits. Exit!\n")
    exit(0)

data = sio.loadmat(data_path)
epsilon = input("Please input the epsilon (0.01, 0.0001, 0.000001) :\n")
epsilon = float(epsilon)


fw = open('log.%s_cgd_epsilon_%f'%(data_n,epsilon),'w')
if fw is False:
    print("Cannot print iteration history into file. Exit!\n")
    exit(0)


print()
print("==================================================")
print("Coordinate gradient descent optimization starts...")
print("==================================================")
print()

start = time.time()

xtrain = data['Xtrain']
ytrain = data['ytrain']

N,M = np.shape(xtrain)
lambda_1 = 0.001
lambda_2 = 0.001

#w = sol_0[0:-1]
#b = sol_0[-1]

#df = np.ones(M+1)
sol = np.zeros(M+1)
norm_sol = np.sqrt(np.dot(sol,sol))

f_value = Functions.target_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
df = Functions.d_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
norm_df = np.sqrt(np.dot(df,df))

count = 0
max_count = 20000 
#max_count = 2

f_his = []
f_his.append(f_value)

fw.write("# iteration number\t f\t df's norm\t solution's norm\n")

while count < max_count and norm_df > epsilon*np.maximum(1,norm_sol):
    if count%50 == 0:
        print("Iteration %d: f = %f, df's norm = %f, solution's norm = %f"%(count,f_value,norm_df,norm_sol))
        fw.write("%d\t %f\t %f\t %f\n"%(count,f_value,norm_df,norm_sol))
    sol = Functions.search_minimum_i(sol,count,N,M,xtrain,ytrain,lambda_1,lambda_2)
    df = Functions.d_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
    f_value = Functions.target_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
    norm_df = np.sqrt(np.dot(df,df))
    norm_sol = np.sqrt(np.dot(sol,sol))
    f_his.append(f_value)
    count += 1
    
print("Iteration %d: f = %f, df's norm = %f, solution's norm = %f"%(count,f_value,norm_df,norm_sol))
fw.write("%d\t %f\t %f\t %f\n"%(count,f_value,norm_df,norm_sol))

f_his.append(f_value)
np.savetxt('%s_cgd_solution_epsilon_%f.dat'%(data_n,epsilon),sol.reshape(len(sol),1))
np.savetxt('%s_cgd_f_history_epsilon_%f.dat'%(data_n,epsilon),np.array(f_his).reshape(len(f_his),1))

if count != max_count:
    
    end = time.time()
    fw.write("Time consumption: %f seconds."%(end-start))
    fw.close()
    print()
    print("======================================================")
    print("Coordinate gradient descent optimization is completed!")
    print("Data set: %s"%data_n)
    print("Steps: %d"%count)
    print("Epsilon = %f"%epsilon)
    print("Time consumption: %f seconds."%(end-start))
    print("======================================================")
    print()

else:
    end = time.time()
    fw.write("Time consumption: %f seconds."%(end-start))
    fw.close()
    print()
    print("===========================================================")
    print("                         WARNING                           ")
    print("Coordinate gradient descent optimization fails to converge!")
    print("Data set: %s"%data_n)
    print("Steps: %d"%max_count)
    print("Epsilon = %f"%epsilon)
    print("Time consumption: %f seconds."%(end-start))
    print("======================================================")
    print()
