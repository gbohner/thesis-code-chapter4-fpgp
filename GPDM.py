import autograd.numpy as np
import autograd.scipy as scipy
import scipy.linalg
import scipy.cluster
import scipy.stats
from autograd import grad, elementwise_grad, jacobian, hessian, value_and_grad
from collections import OrderedDict

import plotly
from plotly.offline import iplot as plt
from plotly import graph_objs as plt_type
plotly.offline.init_notebook_mode(connected=True)


# Kernel func

def square_dist(x, x2=None, lengthscales=None):
    if lengthscales is None:
        lengthscales=np.ones((x.shape[0], 1))
    
    x = x / lengthscales
    xs = np.sum(np.square(x), axis=0)
    if x2 is None:
        return -2 * np.dot(x.T, x) + \
               np.reshape(xs, (-1, 1)) + np.reshape(xs, (1, -1))
    else:
        x2 = x2 / lengthscales
        x2s = np.sum(np.square(x2), axis=0)
        return -2 * np.dot(x.T, x2) + \
               np.reshape(xs, (-1, 1)) + np.reshape(x2s, (1, -1))

def euclid_dist(x, x2, lengthscales=None):
    if lengthscales is None:
        lengthscales=np.ones((x.shape[0], 1))
    r2 = square_dist(x, x2, lengthscales)
    return np.sqrt(r2 + 1e-12)

def RBF(x, x2=None, lengthscales=None, kernel_variance=1):
    if lengthscales is None:
        lengthscales=np.ones((x.shape[0], 1))
    
    return kernel_variance*np.exp(-square_dist(x, x2, lengthscales=lengthscales)/2)

def dRBF(x, x2=None, *args,**kwargs):
    if x2 is None:
        x2 = x
        
    D = x.shape[0]
    N_x1 = x.shape[1]
    
    # The jacobian returns a shape
    # (N_x1, N_x2, D, 1)
    jRBF = jacobian(RBF)
    gradRBF = np.concatenate([jRBF(x[:,i:(i+1)], x2, *args, **kwargs) for i in range(N_x1)], axis=0)
    # For every x1 input point, compute a 1xN_x2xD jacobian, then stack them by point getting N_x1, N_x2, D, 1
    
    # We want to stack it to 2D to have shape 
    # (D*N_x1,  N_x2)
    
    # Here the derivative is with respect to the first argument, and it is ANTI-SYMMETRIC (Transpose -> minus sign)
    return np.reshape(gradRBF.swapaxes(1,2).swapaxes(0,1), (D*N_x1, -1), order='F')
    
    
def ddRBF(x, x2=None, *args, **kwargs):
    if x2 is None:
        x2 = x
    
    D = x.shape[0]
    N_x1 = x.shape[1]
    N_x2 = x2.shape[1]
    
    # The hessian defined here returns a shape
    # (D*N_x1, N_x2, D, 1)
    hRBF = jacobian(dRBF, argnum=1)
    
    hessRBF = np.concatenate([hRBF(x, x2[:,j:(j+1)], *args, **kwargs) for j in range(N_x2)], axis=1)
    
    # We want to stack it to 2D to have shape 
    # (D*N_x1, D*N_x2)
    
    return np.reshape(hessRBF.swapaxes(1,2), (D*N_x1, -1), order='F')













# Expected kernels

def RBF_eK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [ k(x, X)], a 1 x M array
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
    return RBF(x=mu, 
               x2=X, 
               lengthscales=np.sqrt(lengthscales**2 + sigma), 
               kernel_variance=kernel_variance*np.sqrt(np.prod(lengthscales**2)/np.prod(lengthscales**2 + sigma))
              )


def RBF_exK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [ x * k(x, X)], a D x M array
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
        
    kern = RBF_eK(mu, sigma, X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    mean_gauss = (X/(lengthscales**2) + mu/sigma)/(1/(lengthscales**2)+(1/sigma))
    
    return kern*mean_gauss

def RBF_edK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [ dk(x, X) ], an 1 x (D x M) array
    We want it differentiated with respect to the second argument X -> No minus sign (minus signs cancel)
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
        
    exK = RBF_exK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    eK = RBF_eK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    
    return np.reshape((exK - X * eK)/(lengthscales**2),(1,-1), order='F')


def RBF_eKK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [k(X, x) * k(x, X) ], an M x M array
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
        
        
    kXX_scaled = RBF(
                       x=X, 
                       x2=X, 
                       lengthscales=np.sqrt(2*(lengthscales**2)), 
                       kernel_variance=kernel_variance*np.sqrt(np.prod(lengthscales**2)/np.prod(2*(lengthscales**2)))
                    )
    
    X_pairwise_sums = X[:,:,None] + X[:,:,None].swapaxes(1,2)

    kXpX_mu = RBF(
                       x=np.reshape(X_pairwise_sums/2,(mu.shape[0], -1), order='F'), 
                       x2=mu, 
                       lengthscales=np.sqrt((lengthscales**2)/2 + sigma), 
                       kernel_variance=kernel_variance*np.sqrt(np.prod(lengthscales**2)/np.prod((lengthscales**2)/2 + sigma))
                    )
    
    return kXX_scaled * np.reshape(kXpX_mu, (X.shape[1], X.shape[1]), order='F')


def RBF_exKK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [x * k(X, x) * k(x, X) ], a D x M x M array
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
        
    
    # M x M array
    eKK = RBF_eKK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    X_pairwise_sums = X[:,:,None] + X[:,:,None].swapaxes(1,2)

    # D x M x M array
    mean_gauss = ((X_pairwise_sums/2)/(((lengthscales**2)/2)[:,:,None]) + (mu/sigma)[:,:,None])/(
                                                (1/((lengthscales**2)/2)+(1/sigma))[:,:,None])
    
    
    
    return eKK[:,:,None].swapaxes(1,2).swapaxes(0,1) * mean_gauss


def RBF_exxKK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [ (x*x.T) * k(X, x) * k(x, X) ], a D x D x M x M array
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
        
    
    # M x M array
    eKK = RBF_eKK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    # D x D array
    var_gauss = 1/(1/((lengthscales**2)/2)+(1/sigma))
    
    
    X_pairwise_sums = X[:,:,None] + X[:,:,None].swapaxes(1,2)

    # D x M x M array
    mean_gauss = ((X_pairwise_sums/2)/(((lengthscales**2)/2)[:,:,None]) + (mu/sigma)[:,:,None])*(var_gauss[:,:,None])
    
    # D x D x M x M array
    mean_outer = np.expand_dims(mean_gauss, axis=1) * np.expand_dims(mean_gauss, axis=0)
    
    
    
    return np.expand_dims(np.expand_dims(eKK, axis=0), axis=0) * (var_gauss[:,:,None,None] + mean_outer)


def RBF_eKdK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [  k(X, x) * dk(x, X)  ], an M x (D x M) array
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
        
        
        
    # d x m1 x m2
    exKK = RBF_exKK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    # m1 x m2
    eKK = RBF_eKK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    # d x m1 x m2,
    # As exKK naturally uses the first argument and
    # X is the second argument in the derivative kernel, we should expand it, such that we iterate along m2 dimension    
    eKdK = (exKK - np.expand_dims(X, axis=1) * np.expand_dims(eKK, axis=0))/((lengthscales**2)[:,:,None])
       
    # We then finally modify the order of axis and the dimensionality to get 
    # the expected m1 - d - m2 order with M x (DM) shape
    
    return np.reshape(eKdK.swapaxes(0,1), (X.shape[1], -1), order='F')


def RBF_edKdK(mu, sigma, X, lengthscales=None, kernel_variance=1):
    """
    x ~ N(mu, sigma), Dx1
    X is DxM
    Return E_x [  dk(X, x) * dk(x, X)  ], a (D x M) x (D x M) array
    """
    if lengthscales is None:
        lengthscales=np.ones((mu.shape[0], 1))
        
        
    # d1 x d2 x m1 x m2
    exxKK = RBF_exxKK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    # d1 x m1 x m2
    exKK = RBF_exKK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    # m1 x m2
    eKK = RBF_eKK(mu=mu, sigma=sigma, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    edKdK = (exxKK # exKK
        -1.0 * np.expand_dims(np.expand_dims(X, axis=2), axis=1) * np.expand_dims(exKK, axis=0)  # X[:,None,:,None]
        -1.0 * np.expand_dims(np.expand_dims(X, axis=1), axis=0) * np.expand_dims(exKK, axis=1)  # X[None,:,None,:]
        + np.expand_dims(np.expand_dims(X, axis=2), axis=1) * np.expand_dims(np.expand_dims(X, axis=1), axis=0) 
           * np.expand_dims(np.expand_dims(eKK, axis=0), axis=0) # X[:,None,:,None] * X[None,:,None,:] * eKK[None,None,:,:]
    )
    
    # Divide with lengthscales appropriately
    edKdK = edKdK / ((lengthscales.T**2)[:,:,None,None])
    edKdK = edKdK / ((lengthscales**2)[:,:,None,None])
    
    # We then finally modify the order of axis and the dimensionality to get 
    # the expected m1 - d - m2 order with M x (DM) shape
    
    return  np.reshape(edKdK.swapaxes(1,2), (X.shape[0]*X.shape[1], X.shape[0]*X.shape[1]), order='F')









# GP functions

def create_fp_gp_funcs():
    def fp_get_static_K(eta, lengthscales, z, u, s, J, sig_eps):
        """
        Return the cholesky decomposition of the structured kernel matrix, 
        as well as structured targets, 
        so these only have to be computed once per parameter update
        """
        
        rbf = lambda x1,x2: RBF(x1, x2, lengthscales=lengthscales, kernel_variance=eta)
        drbf = lambda x1,x2: dRBF(x1, x2, lengthscales=lengthscales, kernel_variance=eta)
        ddrbf = lambda x1,x2: ddRBF(x1, x2, lengthscales=lengthscales, kernel_variance=eta)
        
        ## Diagonal blocks
        # Inducing points
        Ku_u = rbf(z,z)
        Ku_u = Ku_u + np.diag((sig_eps*np.ones((Ku_u.shape[0],1))).flatten()) # Noise is due to noisy transition

        # Fixed points
        Ks_s = rbf(s,s)
        Ks_s = Ks_s #+ np.diag((sig_eps*np.ones((Ks_s.shape[0],1))).flatten()) # Noise is due to noisy transition
        
        # Jacobians at fixed points
        KJ_J = ddrbf(s,s) # Derivatives are not affected by noisy transition ?

        ## Off-diagonal blocks
        # Fixed vs inducing
        Ks_u = rbf(s, z)
        
        # Jac vs inducing
        KJ_u = drbf(s, z)
        
        # Jac vs fixed
        KJ_s = drbf(s, s)


        ## Stack the matrices appropriately (3 x 3 blocks) # need - signs for derivative transposes?

        K_full = np.concatenate(
                           [np.concatenate([Ku_u, Ks_u, KJ_u]), 
                           np.concatenate([Ks_u.T, Ks_s, KJ_s]), 
                           np.concatenate([KJ_u.T, KJ_s.T, KJ_J])],
                           axis=1)
        
        #return K_full

        L = np.linalg.cholesky(K_full)
        
        
#         
# #         # Transform J for appropriate lengthscale (columnwise multiple)
# #         if not (lengthscales is None):
# #             J_scaled = J / np.broadcast_to( (lengthscales[:,:,np.newaxis]).swapaxes(0,1).swapaxes(1,2), J.shape)
# #             print J_scaled
# #         else:
#         J_scaled = J
    
#         # Reshape J to be appropriate dimensions for the derivative kernel, 
#         # D-dim feature column vectors stacked vertically for each fixed point (DxM) x D output features as number of columns
#         J_scaled = np.reshape(J_scaled.swapaxes(0,1),(-1, J.shape[1]))

        # J is originally M fixed points x DxD jacobians, make it into (DxM) x D
        targets = np.concatenate([u.T, s.T, np.reshape(J.swapaxes(0,1),(-1, J.shape[1]), order='F')]) 
        # Sometimes called beta, (L + M + D_inxM) x D_out, where we will use D independent GPs, one for each output dimension 
        # (output and input dimensions are both D)
        
        params = {'eta': eta, 'lengthscales': lengthscales, 'z': z, 'u': u, 's': s, 'J':J, 'sig_eps': sig_eps}
        
        return (L, targets, params)
        
    def fp_predict(xstar, L, targets, params):
        # Separate predictions per output dimension
        D = targets.shape[1]
        
        # Compute the kstar kernels:
        Kx_u = RBF(xstar, params['z'], lengthscales=params['lengthscales'], kernel_variance=params['eta'])
        Kx_s = RBF(xstar, params['s'], lengthscales=params['lengthscales'], kernel_variance=params['eta'])
        Kx_J = dRBF(params['s'], xstar, lengthscales=params['lengthscales'], kernel_variance=params['eta']).T
        Kx_x = RBF(xstar, xstar, lengthscales=params['lengthscales'], kernel_variance=params['eta'])

        Kx_pred = np.concatenate([Kx_u, Kx_s, Kx_J], axis=1).T
        
        # Compute the predictive mean per dimension then concatenate them
        Mu_star = np.concatenate([ \
                np.dot(Kx_pred.T, 
                       scipy.linalg.solve_triangular(
                            L.T, scipy.linalg.solve_triangular(L, targets[:,d:(d+1)], lower=True),
                            lower=False)
                       ) \
             for d in range(D)], axis=1)
        
        
        # Compute the predictive variance
        Sig_star = Kx_x - np.dot(Kx_pred.T, 
                                 scipy.linalg.solve_triangular(
                                        L.T, scipy.linalg.solve_triangular(L, Kx_pred, lower=True),
                                        lower=False)
                                 )
        
        return (Mu_star, Sig_star, Kx_pred)
    
    
    return (fp_get_static_K, fp_predict)
    
    
(fp_get_static_K, fp_predict) = create_fp_gp_funcs()







# Time series filtering

def update_t_t1(mu_t1_t1, Sigma_t1_t1, L, targets, kernel_variance, z, u, lengthscales, s, J):
    """
        Compute mu_t_t1 and Sigma_t_t1 given mu_t1_t1 and Sigma_t1_t1, as well as the parameters 
    """
    
    D = mu_t1_t1.shape[0]
    
    # K inverse matrix (Nu+Ns+D*Ns)^2 and targets (Nu+Ns+D*Ns)*D scaled
    #Linv = scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    Linv = np.linalg.inv(L) # Numpy inv is by far the fastest for moderate matrices that I expect L to be
    Kinv = np.dot(Linv.T, Linv)
    
    alpha = np.dot(Kinv, targets)
    
    
    # Expected kernels
    X = np.concatenate([z, s],axis=1) # Base observation locations
    dX = s # extra derivative observation locations
    
    eK_zs = RBF_eK(mu=mu_t1_t1, sigma=Sigma_t1_t1, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    edK_s = RBF_edK(mu=mu_t1_t1, sigma=Sigma_t1_t1, X=dX, lengthscales=lengthscales, kernel_variance=kernel_variance)
        
    eKK_zs_zs = RBF_eKK(mu=mu_t1_t1, sigma=Sigma_t1_t1, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)

    # We will only use a part of this, we compute uneccessarily the derivatives at locations z.
    eKdK_zs_zs = RBF_eKdK(mu=mu_t1_t1, sigma=Sigma_t1_t1, X=X, lengthscales=lengthscales, kernel_variance=kernel_variance)
    
    edKdK_s_s = RBF_edKdK(mu=mu_t1_t1, sigma=Sigma_t1_t1, X=dX, lengthscales=lengthscales, kernel_variance=kernel_variance)
 
    
    # Get blocks
    D = s.shape[0]
    Nz = z.shape[1]
    Ns = s.shape[1]
    Nzs = Nz+Ns
    NJ = D * s.shape[1]
    
    alpha_zs = alpha[0:Nzs,:]
    alpha_J = alpha[-NJ:,:]
    
    Kinv_zs = Kinv[0:Nzs, 0:Nzs]
    Kinv_J = Kinv[-NJ:, -NJ:]
    
    eKdK_zs_s = eKdK_zs_zs[:, -NJ:]
    
    # Compute the mean
    mu_t_t1 = np.zeros((D,1))
    
    
    mu_t_t1 = (np.dot( eK_zs , alpha_zs) +
            np.dot( edK_s , alpha_J)
            )
    
    mu_t_t1 = mu_t_t1.T
    
    
    # Compute output variances
    
    # Shared for each dimensions (diagonal variances only or every entry??? Yes, expected covariance of indep GPs = 0):
    e_sigma_out = (
        RBF(x=mu_t1_t1, lengthscales=lengthscales, kernel_variance=kernel_variance) +
        - np.trace(np.dot(Kinv_zs, eKK_zs_zs)) +
        - 2*np.trace(np.dot(Kinv[0:Nzs, -NJ:], eKdK_zs_s.T)) +
        - np.trace(np.dot(Kinv_J, edKdK_s_s))
    )
    
    
    # Dimension_specific (D x D matrix)
    e_mu_muT_out = (
        np.dot(np.dot(alpha_zs.T, eKK_zs_zs), alpha_zs)  
        + np.dot(np.dot(alpha_zs.T, eKdK_zs_s), alpha_J)
        + np.dot(np.dot(alpha_zs.T, eKdK_zs_s), alpha_J).T
        + np.dot(np.dot(alpha_J.T, edKdK_s_s), alpha_J)
    )
    
    m_mT = np.dot(mu_t_t1, mu_t_t1.T)
    
    Sigma_t_t1 = e_sigma_out * np.eye(D) + e_mu_muT_out - m_mT
    
    return (mu_t_t1, Sigma_t_t1)
         
    
    
def update_t_t(mu_t_t1, Sigma_t_t1, C, Sigma_nu, y_t):
    """
        Compute mu_t_t and Sigma_t_t given mu_t_t1 and Sigma_t_t1, as well as the parameters 
    """
    
    proj_var = np.dot(C, Sigma_t_t1)
    proj_inv = np.linalg.inv(np.dot(proj_var, C.T) + np.diag(Sigma_nu.flatten()))
    proj_back = np.dot(Sigma_t_t1, np.dot(C.T, proj_inv))
    
    mu_t_t = mu_t_t1 + np.dot(proj_back, (y_t - np.dot(C, mu_t_t1)))
    
    Sigma_t_t = Sigma_t_t1 - np.dot(proj_back, proj_var)
    
    return (mu_t_t, Sigma_t_t)
    
    
def nlog_marg_ll(mu_t_t1, Sigma_t_t1, C, Sigma_nu, y_t):
    """
        Computes the negative log marginal likelihood
        - log N(y_t; C*mu_t_t1, C * Sigma_t_t1 * C.T + Sigma_nu)
    """

    mu_gauss = np.dot(C, mu_t_t1) 
    var_gauss = np.dot(np.dot(C, Sigma_t_t1), C.T) + np.diag(Sigma_nu.flatten())
    
    inv_var_gauss = np.linalg.inv(var_gauss)
    
    D = mu_t_t1.shape[1]*1.0
    
    log_marg_ll = (
        - 1.0* D/2.0*np.log(2*np.pi)
        - 1.0/2.0*np.linalg.slogdet(var_gauss)[1] 
        - 1.0/2.0*np.dot(np.dot( (y_t.T - mu_gauss.T), inv_var_gauss), (y_t - mu_gauss) )  
    )
    
    return -1.0 * log_marg_ll
                                                       
    
    
    

    
    
    
# Optimisation algorithm

def init_params(y, D, Nz, Ns):
    """
    Initialise the model parameters given 
    - y (Dy x T x N): observations
    - D (scalar): latent dimensionality
    - Nz (scalar): number of inducing points
    - Ns (scalar): number of fixed points
    """    
    
    Dy = y.shape[0]
    
    # If Dy = D, then just set 
    if D==Dy:
        C = np.eye(D, Dy)
    else:
        # Do PCA and choose the first to find initial C (todo)
        C = np.eye(D, Dy)
        
    # Estimate the noisyness of y via simple autoregressive AR(1) linear regression predictability, per dim
    y_t = y[:,1:,:]
    y_t1 = y[:,:-1,:]

    y_t_rsh = np.reshape(y_t, (D, -1)).T
    y_t1_rsh = np.reshape(y_t1, (D, -1)).T

    # Get optimal linear estimate of y_t given the y_t-1 vector    
    y_t_hat = np.dot(y_t1_rsh, np.dot(np.dot(np.linalg.inv(np.dot(y_t1_rsh.T, y_t1_rsh)), y_t1_rsh.T), y_t_rsh))

    # Set the estimated variance of y_t
    Sigma_nu = np.var(y_t_rsh - y_t_hat, axis = 0)[:,None].T
    
    
    # Init GP params
    # First run basic GP regression from y_t-1 -> y_t to estimate kernel hyperparams (skip for now, do simpler things)
    
    # Set inducing point locations to span y backprojected through Cinv
    if D==Dy:
        Cinv = np.eye(D, Dy)
    else:
        Cinv = numpy.linalg.pinv(C)
        
    Cinvy = np.tensordot(Cinv, y, axes = ([1], [0]))
    grid_min = np.min(Cinvy, axis=(1,2))
    grid_max = np.max(Cinvy, axis=(1,2))
    
    grid_num = np.floor(Nz ** (1./D)) # Do an evenly spaced D-dim grid using floor(sqrt(Nz))**D points
    remaining_num = Nz - grid_num**D # Initialise the other points randomly
    
    grid_axes = []
    for d in range(D):
        grid_axes.append(np.linspace(grid_min[d], grid_max[d], num=grid_num))

    grid_dims = np.meshgrid(*grid_axes)
    grid_dims_flat = []
    for d in range(D):
        grid_dims_flat.append(grid_dims[d].flatten())
    gridpts = np.array(grid_dims_flat)
    
    np.random.seed(123)
    if remaining_num > 0:
        randompts = np.random.rand(D,remaining_num.astype(int))*(grid_max-grid_min)[:,None]+grid_min[:,None]
        z = np.concatenate([gridpts, randompts], axis=1)
    else:
        z = gridpts

    # Initialise inducing point values and noise to give a flat-ish surface
    u = z + 1e-1*np.random.randn(*z.shape)
    Sigma_eps = np.array([[1e-1]])
    
    
    # Estimate kernel lengthscale from grid size and number of inducing points
    kernel_variance = np.array([[1.0]])
    lengthscales=(grid_max-grid_min)[:,None]/grid_num * (2.*np.sqrt(D*1.)/2.) # Account for dimension scaling (sqrtD) also
    
    
    # Init p(x_0) as expected backprojection of first time step
    yrsh = Cinvy[:,0,:]
    mu_0_0 = np.mean(yrsh, axis=1)[:, None]
    Sigma_0_0 = np.var(yrsh, axis=1)[:, None]
#    # Subtract the contribution from Sigma_nu (the linear regression explained observation noise)
#     Sigma_0_0 = 1./(1./Sigma_0_0 - 1./Sigma_nu) # Need more checking
    
    
    ## Initialise fixed point locations from k-means clustering on the back-projected values (k_medians better though)
    # Transform the data appropriately, and whiten it (kmeans requires) then unwhiten
    yrsh = (np.reshape(Cinvy, (D,-1)).T)
    ystd = np.std(yrsh, axis=0,)
    s = (scipy.cluster.vq.kmeans(yrsh / np.expand_dims(ystd, axis=0), Ns)[0]*np.expand_dims(ystd, axis=0)).T

    
    Jac_matrices = []
    for n in range(Ns):
        Jac_matrices.append(np.expand_dims(np.zeros((D,D)), axis=0))
    J = np.concatenate(Jac_matrices, axis=0)
    
    
    return (mu_0_0, Sigma_0_0, C, Sigma_nu, z, u, Sigma_eps, lengthscales, kernel_variance, s, J)
    
    
def params_to_vec(mu_0_0, Sigma_0_0, C, Sigma_nu, z, u, Sigma_eps, lengthscales, kernel_variance, s, J):
    """
    Return all the parameters vectorised into a flattened vector, but also a dictionary of parameter lengths
    """
    paramvec = []
    paramdict_ind = OrderedDict()
    paramdict_shape = OrderedDict()
    
    paramdict_shape['mu_0_0'] = mu_0_0.shape
    paramdict_shape['Sigma_0_0'] = Sigma_0_0.shape
    paramdict_shape['C'] = C.shape
    paramdict_shape['Sigma_nu'] = Sigma_nu.shape
    paramdict_shape['z'] = z.shape
    paramdict_shape['u'] = u.shape
    paramdict_shape['Sigma_eps'] = Sigma_eps.shape
    paramdict_shape['lengthscales'] = lengthscales.shape
    paramdict_shape['kernel_variance'] = kernel_variance.shape
    paramdict_shape['s'] = s.shape 
    paramdict_shape['J'] = J.shape
    
    cur_ind = 0
    for key in paramdict_shape.keys():
        paramdict_ind[key] = np.arange(cur_ind, cur_ind+np.prod(paramdict_shape[key]))
        cur_ind = cur_ind+np.prod(paramdict_shape[key])
    
    paramvec.append(np.reshape(mu_0_0, (-1,1)))
    paramvec.append(np.reshape(Sigma_0_0, (-1,1)))
    paramvec.append(np.reshape(C, (-1,1)))
    paramvec.append(np.reshape(Sigma_nu, (-1,1)))
    paramvec.append(np.reshape(z, (-1,1)))
    paramvec.append(np.reshape(u, (-1,1)))
    paramvec.append(np.reshape(Sigma_eps, (-1,1)))
    paramvec.append(np.reshape(lengthscales, (-1,1)))
    paramvec.append(np.reshape(kernel_variance, (-1,1)))
    paramvec.append(np.reshape(s, (-1,1)))
    paramvec.append(np.reshape(J, (-1,1)))
    
    paramvec = np.concatenate(paramvec, axis=0).flatten()
    
    return (paramvec, paramdict_ind, paramdict_shape)

def vec_to_params(paramvec, Dy, D, Nz, Ns):
    i = 0
    
    mu_0_0 = paramvec[i:(i+D)]
    i += mu_0_0.shape[0]
    mu_0_0 = mu_0_0[:, None]
    
    Sigma_0_0 = paramvec[i:(i+D)]
    i += Sigma_0_0.shape[0]
    Sigma_0_0 = Sigma_0_0[:, None]
        
    C = paramvec[i:(i+D*Dy)]
    i += C.shape[0]
    C = np.reshape(C, (D,Dy))
    
    Sigma_nu = paramvec[i:(i+Dy)]
    i += Sigma_nu.shape[0]
    Sigma_nu = Sigma_nu[:, None]
    
    z = paramvec[i:(i+D*Nz)]
    i += z.shape[0]
    z = np.reshape(z, (D, Nz))
    
    u = paramvec[i:(i+D*Nz)]
    i += u.shape[0]
    u = np.reshape(u, (D, Nz))
    
    Sigma_eps = paramvec[i:(i+1)]
    i += Sigma_eps.shape[0]
    Sigma_eps = Sigma_eps[:, None]
    
    lengthscales = paramvec[i:(i+D)]
    i += lengthscales.shape[0]
    lengthscales = lengthscales[:, None]
    
    kernel_variance = paramvec[i:(i+1)]
    i += kernel_variance.shape[0]
    kernel_variance = kernel_variance[:, None]

    
    s = paramvec[i:(i+D*Ns)]
    i += s.shape[0]
    s = np.reshape(s, (D, Ns))
    
    J = paramvec[i:(i+Ns*D*D)]
    i += J.shape[0]
    J = np.reshape(J, (Ns, D, D))
    
    
    
    assert i == paramvec.shape[0]

    
    return (mu_0_0, Sigma_0_0, C, Sigma_nu, z, u, Sigma_eps, lengthscales, kernel_variance, s, J)



# Define a single complete iteration for vectorised parameter collection
def time_full_iter(paramvec, y, D, Nz, Ns, ret_smoothed = False):
    """
    Run a full iteration filtering forward in time, returning the total negative log likelihood (the objective)
    as a function of a vectorised collection of all our parameters, so we can take easy gradient steps
    """
    
    # The objective is to minimise the negative log likelihood
    neg_log_likelihood = 0
    
    Dy = y.shape[0]
    T = y.shape[1]
    Ny = y.shape[2]
    
    # Collect smoothed latent trajectories
    x_all_t1 = np.zeros((D, T, Ny))
    x_all_t = np.zeros((D, T, Ny))
    sig_all_t1 = np.zeros((D, T, Ny))
    sig_all_t = np.zeros((D, T, Ny))
    
    # Unpack the parameters
    (mu_0_0, Sigma_0_0, C, Sigma_nu, z, u, Sigma_eps, lengthscales, kernel_variance, s, J) = vec_to_params(paramvec, Dy, D, Nz, Ns)
    
    L, targets, params = fp_get_static_K(eta=kernel_variance, lengthscales=lengthscales, z=z, u=u, s=s, J=J, sig_eps=Sigma_eps)

    for n in range(Ny):
        
        mu_t1_t1 = mu_0_0
        Sigma_t1_t1 = Sigma_0_0
        
        for t in range(T):
            mu_t_t1, Sigma_t_t1 = update_t_t1(mu_t1_t1, Sigma_t1_t1, L, targets, kernel_variance, z, u, lengthscales, s, J)

            Sigma_t_t1 = Sigma_t_t1 + 1e-6
            
            if ret_smoothed:
                x_all_t1[:,t:(t+1),n:(n+1)] = mu_t_t1.flatten()            
                sig_all_t1[:,t:(t+1),n:(n+1)] = np.diag(Sigma_t_t1).flatten()
            
            cur_nll_term = nlog_marg_ll(mu_t_t1, Sigma_t_t1, C, Sigma_nu, y[:,t:(t+1),n])
            neg_log_likelihood = neg_log_likelihood + cur_nll_term

            mu_t1_t1, Sigma_t1_t1 = update_t_t(mu_t_t1, Sigma_t_t1, C, Sigma_nu, y[:,t:(t+1),n])
            Sigma_t1_t1 = np.diag(Sigma_t1_t1)[:,None]

            Sigma_t1_t1 = Sigma_t1_t1 + 1e-6
            
            if ret_smoothed:
                x_all_t[:,t:(t+1),n:(n+1)] = mu_t1_t1.flatten()
                sig_all_t[:,t:(t+1),n:(n+1)] = np.diag(Sigma_t1_t1).flatten()

            #print(np.concatenate([y[:,t:(t+1),0], mu_t_t1, mu_t1_t1], axis=1))
            
    return (neg_log_likelihood, x_all_t1, x_all_t, sig_all_t1, sig_all_t)
        
    