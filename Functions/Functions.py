from __future__ import print_function
import numpy as np
import time

# get the value of target function
def target_function(sol,N,M,x,y,lambda_1,lambda_2):
    # there are N data sets, each has a dimension of M
    w = sol[0:-1]
    b = sol[-1]
    y = y.reshape(N,)
    exp_temp = np.exp(-y*(np.matmul(x,w)+b))
    return np.mean(np.log(1+exp_temp)) + 0.5*lambda_1*np.dot(w,w) + 0.5*lambda_2*b*b

# get the first order derivative of target function
def d_function(sol,N,M,x,y,lambda_1,lambda_2):
    y = y.reshape(N,)
    w = sol[0:-1]
    b = sol[-1]
    exp_temp = np.exp(-y*(np.matmul(x,w)+b))
    cons = exp_temp*(-y)/(N*(1+exp_temp))

    # get the derivative of w
    dw = np.sum((cons*x.transpose()).transpose().reshape(N,M),axis=0) + lambda_1*w
    # get the derivative of b
    db = np.sum(cons) + lambda_2*b

    return np.hstack((-dw,-db))

# get the first and second order derivative of target function with the specific
# parameter
def dd_f_i(sol,I,N,M,x,y,lambda_1,lambda_2):
    y = y.reshape(N,)
    w = sol[0:-1]
    b = sol[-1]

    exp_temp = np.exp(-y*(np.matmul(x,w)+b))
    cons = exp_temp*(-y)/(N*(1+exp_temp))
    cons2 = exp_temp*y**2/(N*(1+exp_temp)**2)

    # dfi stores 1st order derivative and ddfi stores 2nd order derivative
    if I<M:
        xI = x[:,I]
        dfi = np.dot(cons,xI) + lambda_1*w[I]
        xx = x*x
        xxI = xx[:,I]
        ddfi = np.dot(cons2,xxI) + lambda_1
    else:
        dfi = np.sum(cons) + lambda_2*b
        ddfi = np.sum(cons2) + lambda_2

    return np.hstack((dfi,ddfi))

# get the local minimum position of f(x_i), solve f'(x_i)=0
def search_minimum_i(sol,count,N,M,x,y,lambda_1,lambda_2):
    itera = 0
    maxit = 1000
    sol_i = sol
    I = count%(M+1)
    #f_value = target_function(sol,N,M,x,y,lambda_1,lambda_2)
    norm_sol_i = abs(sol_i[I])
    dd_d = dd_f_i(sol_i,I,N,M,x,y,lambda_1,lambda_2)
    norm_df_i = abs(dd_d[0])
    tol = 0.000001
    while itera <= maxit and norm_df_i > tol*np.maximum(1,norm_sol_i):
        itera += 1
        # get the values of f'(x_i) and f''(x_i)
        dd_d = dd_f_i(sol_i,I,N,M,x,y,lambda_1,lambda_2)
        # update the solution via Newton's method
        sol_i[I] += -1.0*dd_d[0]/dd_d[1]
        # get the norm of solution value in the ith parameter at new solution location
        norm_sol_i = abs(sol_i[I])
        # get the new overall f'(x) at new x and obtain new f'(x_i)
        df_i = -d_function(sol_i,N,M,x,y,lambda_1,lambda_2)
        norm_df_i = abs(df_i[I])

        #f_value = target_function(sol_i,N,M,x,y,lambda_1,lambda_2)

    return sol_i

def back_tracking(sol,df,N,M,x,y,lambda_1,lambda_2,t0):
    itera = 0
    maxit = 100
    alp = 0.5
    bet = 0.5
    tp = t0
    ten_sol = sol+tp*df
    dfdf = np.dot(df,df)
    f_v = target_function(sol,N,M,x,y,lambda_1,lambda_2)
    #print(f_v)
    lhs = target_function(ten_sol,N,M,x,y,lambda_1,lambda_2)
    rhs = f_v - alp*tp*dfdf
    while itera < maxit and lhs > rhs:
        tp *= bet
        ten_sol = sol+tp*df
        lhs = target_function(ten_sol,N,M,x,y,lambda_1,lambda_2)
        rhs = f_v - alp*tp*dfdf
        itera += 1
        #print("t=%f , lhs = %f , rhs = %f , %f\n"%(t,lhs,rhs,f_v))
    return tp

# get the Hessian matrix
def hessian(sol,N,M,x,y,lambda_1,lambda_2):
    w = sol[0:-1]
    b = sol[-1]
    Hess = np.zeros((M+1,M+1))
    y = y.reshape(N,)
    Hess = np.zeros((M+1,M+1))

    exp_temp = np.exp(-y*(np.matmul(x,w)+b))
    cons = y**2*exp_temp/(N*(1+exp_temp)**2)
    e = np.ones((N,1))
    lm = np.concatenate((x.transpose(),e.transpose()),axis=0)
    rm = np.concatenate((x,e),axis=1)
    mm = np.diag(cons)
    add = lambda_1*np.identity(M+1)
    add[-1,-1] = lambda_2
    #start = time.time()
    Hess = np.matmul(lm,np.matmul(mm,rm)) + add
    #print(time.time()-start)

    #start = time.time()
    #print(time.time()-start)
    return Hess
