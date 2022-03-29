import numpy as np
from scipy import io as sio
from scipy.sparse import diags


def load_neur_data_alltiles(animal):
    # load original data in MATLAB format; output dF/F fluorescence traces
    
    fname = 'C:/svn2/dynamic_weights/models/m' + str(animal) +'_data_only.mat'
    data = sio.loadmat(fname)

    beh = data['beh']
    pdi = beh['pdi'][0][0][0]
    dff = data['dff']

    assert(dff.shape[0]==182)
    
    xwhl = pdi[2][:25,:,np.newaxis]

    # remove inputs and readouts on trials with NaNs
    dff0 = np.squeeze(dff[0,:,:])
    nan_trials =  np.sum( np.isnan(dff0) , axis=0)>0 
    dff[:,:,nan_trials] = 0.0
    xwhl[:,nan_trials,:] = 0.0
    
    dff_flat_alltiles = np.zeros((dff.shape[0], dff.shape[1]*dff.shape[2]))
    for i in range(dff.shape[0]):
        this_dff = np.squeeze( dff[i,:,:] )
        dff_flat_alltiles[i,:] = np.reshape( this_dff.T, (-1,) )

    dff_flat_alltiles = dff_flat_alltiles/np.nanstd( np.reshape(dff_flat_alltiles,(-1,)) )
    
    return dff_flat_alltiles


def load_neur_data(animal,tile):
    
    fname = 'C:/svn2/dynamic_weights/models/m' + str(animal) +'_data_only.mat'
    data = sio.loadmat(fname)

    beh = data['beh']
    pdi = beh['pdi'][0][0][0]
    dff = data['dff']

    if dff.shape[0]==182:
        dff = np.squeeze(dff[tile,:,:])
    
    xwhl = pdi[2][:25,:,np.newaxis]
    
    # remove inputs and readouts on trials with NaNs
    nan_trials =  np.sum( np.isnan(dff.T), axis=1 )>0 
    dff[:,nan_trials] = 0.0
    xwhl[:,nan_trials,:] = 0.0
    
    dff_flat = np.reshape(dff.T,(-1,))
    dff_flat = dff_flat/np.nanstd(dff_flat)

    T,N,_ = xwhl.shape
    K = 20
    
    
    # make an input matrix X
    X = np.zeros((T,N,K))
    for trial in range(N):
        for i in range(K):
            x0pad = np.vstack( ( np.zeros((i,1)), xwhl[:,trial,[0]] ) ) # zero-pad inputs x
            X[:,trial,[i]] = x0pad[:-i or None,:] # write into input matrix
    
    
    # note that the order is T-fastest, N-next, K-next, so the N*T dimension of xxT should loop over T the fastest hence T should be dim=1 not dim=0
    xxT = np.einsum('ijk,ijl->ijkl', X.transpose(1,0,2), X.transpose(1,0,2))
    xxT = np.reshape( xxT, (N*T, K, K) )
    xxT_block = myblk_diags(xxT)
    
    
    return X,dff_flat,xxT_block,xwhl



def myblk_diags(A):
    # input A [N,K,K], put each len(N) entry of A[:,i,j] as
    # the diagonal of an (N*K x N*K) matrix. 
    # Borrowed from N.Roy's package psytrack 
    # used e.g. in creating xxT matrix when the long weight vector is horzstack of each of its dimensions' time-vectors 
    
    # Retrieve shape of given matrix
    N, K, _ = np.shape(A)

    # Will need (2K-1) diagonals, with the longest N*K long
    d = np.zeros((2 * K - 1, N * K))

    # Need to keep track of each diagonal's offset in the final
    # matrix : (0,1,...,K-1,-K+1,...,-1)
    offsets = np.hstack((np.arange(K), np.arange(-K + 1, 0))) * N

    # Retrieve diagonal values from A to fill in d
    for i in range(K):
        for j in range(K):
            m = np.min([i, j])
            d[j - i, m * N:(m + 1) * N] = A[:, i, j]

    # After diagonals are constructed, use sparse function diags() to make
    # matrix, then blow up to full size
    return diags(d, offsets, shape=(N * K, N * K), format='csc')