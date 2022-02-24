import scipy.sparse as sp
from scipy.sparse.linalg import splu
import numpy as np


def logdet(A):
    # compute log determinant of a sparse matrix A
    
    if not sp.issparse(A):
        LU = splu( sp.csc_matrix(A) )
    else:   
        LU = splu(A)
        
    logdetl = np.sum(np.log(LU.L.diagonal()))
    logdetu = np.sum(np.log(LU.U.diagonal()))
    
    return logdetu + logdetl

def calcC_useSparse(N,M,sgma):
    # compute inverse of prior covariance matrix (C) and its logdet
    # N - number of trials
    # M - number of state dimensions
    # sgma - volatility parameter: standard deviation of weights(=state) change from (t-1) to t, i.e. psi=w(t)-w(t-1)
    
    # use case notes:
    # if C is computed for the dynamic internal state (w), input M is the dimensionality of the internal state (M)
    # if C is computed for the dynamic projection matrix (A), input M is number of its elements (M*K)
    
    # precision matrix of state increments psi
    sginv = sgma**(-2) * sp.eye(N*M)
    
    # difference matrix, psi=D.dot(theta.ravel())
    D = sp.eye(N*M) + sp.diags(-1 * np.ones(M*N-M,), -M)
    
    # inverse prior covariance of theta
    Cm1 = D.transpose().dot(sginv).dot(D) 
    
    logdetCm1 = logdet(Cm1)
    Cm1 = Cm1.todense()
    
    return Cm1, logdetCm1

def calcpy(w,x,y_dyn):
    # compute p(y) likelihood of binary behavioral choices (y_dyn) given inputs (x) and weights (w) 
    
    M=x.shape[1]
    # provide a flattened w with order w=[w_t1_d1, w_t1_d2, ..., w_t2_d1, ...] (necessary for optimization)
    w.shape = (-1,M)
    wx = np.einsum('ij,ji->i',w,x.transpose())
    wx.shape=(-1) 
    py = np.exp(y_dyn*wx) / (1 + np.exp(wx) ) 
    w.shape=(-1) 
    
    return py

def calclogpost(w,Cm1,logdetCm1,py):
    # compute posterior over weights w given likelihood p(y) and prior precision matrix Cm1
    
    w.shape = (-1,1)
    prior = 0.5*(logdetCm1 - w.transpose().dot(Cm1).dot(w)) # +const(pi..)
    logpost = prior + np.sum(np.log(py(w))) # +const(evid.)
    w.shape=(-1)
    
    return logpost 

def calcdLdw(w,x,y_dyn,Cm1):
    # compute derivative of posterior wrt w given inputs (x), weights (w), choices (y_dyn), and precision matrix Cm1
    
    M=x.shape[1]
    prior = -Cm1.dot(w)
    w.shape = (-1,M)
    wx = np.einsum('ij,ji->i',w,x.transpose())
    wx.shape=(-1)
    likel = x * np.expand_dims(y_dyn - np.exp(wx) / (1 + np.exp(wx)), axis=1 )
    likel.shape = (-1)
    dLdw = prior + likel
    w.shape=(-1)
    
    return dLdw


def prepInp(inp,N,T,krnsz,K):
    # zero-pad the inputs to be used for design matrix
    # krnsz - time bins of the response kernel [pre-event, post-event]
    # N - n trials; T - trial duration; K - n input channels 
    
    krnlen = krnsz[0] + krnsz[1] + 1

    inp_padded = np.zeros((N,T+krnlen-1,K))

    # leading padding should be the length of the causal part of the kernel
    inp_padded[:,krnsz[1]:T+krnsz[1],:] = np.transpose(inp, (1,0,2))

    return inp_padded


def makeX(inp_padded,T,krnlen,N,K):
    # make input matrix X
    # krnlen - n timebins in kernel
    # N - n trials; T - trial duration; K - n input channels 
    
    X = np.zeros((N*T,krnlen,K))
    
    for k in range(K):
        
        this_inp = inp_padded[:,:,k]
    
        for i, trial in enumerate(this_inp):

            for timebin in range(T):

                this_X_row = trial[timebin:timebin+krnlen]
                X[timebin + i*T,:,k] = this_X_row
    
    return X


def calcXmodX(X,a_mod,T):
    # compute X "modulated" by state a_mod that changes from trial to trial but not across timebins within a trial
    # and concatenate it with an unmodulated X, i.e. get an [Xmod X] design matrix of the dynamic convolutional model
    
    if X.ndim == 2:
        mod_size = (T,1)
    elif X.ndim == 3:
        mod_size = (T,1,1)
        a_mod = a_mod[:,np.newaxis,:] # insert tau dimension
    
    vect_mod = np.kron(a_mod,np.ones( mod_size ) )# expand N dimension by T (timebins in a trial)
    Xmod = X*vect_mod
    XmodX = np.concatenate((Xmod,X), axis=1)
    
    return XmodX