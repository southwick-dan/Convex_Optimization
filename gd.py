#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import Functions.Functions as Functions
import time,os


print()
opt = input("Please input the data file ('g' for 'gisette', 's' for 'spamData') :\n")

if opt == 'g':
    data_n = 'gisette'
    t0 = 1.0
elif opt == 's':
    data_n = 'spamData'
    t0 = 0.001
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


fw = open('log.%s_gd_epsilon_%f'%(data_n,epsilon),'w')
if fw is False:
    print("Cannot print iteration history into file. Exit!\n")
    exit(0)


print()
print("=======================================")
print("Gradient descent optimization starts...")
print("=======================================")
print()

start = time.time()

xtrain = data['Xtrain']
ytrain = data['ytrain']

N,M = np.shape(xtrain)
lambda_1 = 0.001
lambda_2 = 0.001

w = np.zeros(M)
b = 0
#sol_0 = np.loadtxt('final_solution_0.dat')
#w = sol_0[0:-1]
#b = sol_0[-1]

#df = np.ones(M+1)
sol = np.hstack((w,b))
norm_sol = np.sqrt(np.dot(sol,sol))
alpha = 0

f_value = Functions.target_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
df = Functions.d_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
norm_df = np.sqrt(np.dot(df,df))

count = 0
#max_count = 1000000 
max_count = 50000

t = t0

f_his = []
f_his.append(f_value)

fw.write("# iteration number\t f\t df's norm\t solution's norm\n")

while count < max_count and norm_df > epsilon*np.maximum(1,norm_sol):
    if count%50 == 0:
    #if count%1 == 0:
        #print("Iteration %d: f = %f, df's norm = %f, solution's norm = %f, t = %f"%(count,f_value,norm_df,norm_sol,t))
        print("Iteration %d: f = %f, df's norm = %f, solution's norm = %f"%(count,f_value,norm_df,norm_sol))
        fw.write("%d\t %f\t %f\t %f\n"%(count,f_value,norm_df,norm_sol))
    df = Functions.d_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
    t = Functions.back_tracking(sol,df,N,M,xtrain,ytrain,lambda_1,lambda_2,t0)
    sol += t*df
    df = Functions.d_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
    f_value = Functions.target_function(sol,N,M,xtrain,ytrain,lambda_1,lambda_2)
    norm_df = np.sqrt(np.dot(df,df))
    norm_sol = np.sqrt(np.dot(sol,sol))
    f_his.append(f_value)
    count += 1
    
print("Iteration %d: f = %f, df's norm = %f, solution's norm = %f"%(count,f_value,norm_df,norm_sol))
fw.write("%d\t %f\t %f\t %f\n"%(count,f_value,norm_df,norm_sol))


f_his.append(f_value)
np.savetxt('%s_gd_solution_epsilon_%f.dat'%(data_n,epsilon),sol.reshape(len(sol),1))
np.savetxt('%s_gd_f_history_epsilon_%f.dat'%(data_n,epsilon),np.array(f_his).reshape(len(f_his),1))

end = time.time()
fw.write("Time consumption: %f seconds."%(end-start))
fw.close()


if count != max_count:
    print()
    print("===========================================")
    print("Gradient descent optimization is completed!")
    print("Data set: %s"%data_n)
    print("Steps: %d"%count)
    print("Epsilon = %f"%epsilon)
    print("Time consumption: %f seconds."%(end-start))
    print("===========================================")
    print()

else:
    print()
    print("================================================")
    print("                    WARNING                     ")
    print("Gradient descent optimization fails to converge!")
    print("Data set: %s"%data_n)
    print("Steps: %d"%max_count)
    print("Epsilon = %f"%epsilon)
    print("Time consumption: %f seconds."%(end-start))
    print("================================================")
    print()
