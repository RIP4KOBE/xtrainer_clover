import numpy as np
from numpy.linalg import eigh
from scipy.linalg import eig, sqrtm, inv, logm, expm, sqrtm

# For plotting
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from sklearn.mixture import GaussianMixture




# Set random seed for reproducibility
np.random.seed(42)

# checked
def init_GMM_timeBased(Data, model):
    """
    Initialization of Gaussian Mixture Model (GMM) parameters by clustering
    the data into equal bins based on the first variable (time steps).

    Parameters:
    - Data: 2D numpy array where each column is a datapoint and each row is a variable.
    - model: Dictionary containing the model parameters such as the number of states (nbStates).

    Returns:
    - model: Updated model dictionary with initialized Priors, Mu, and Sigma.
    """

    # Number of variables (dimensions)
    nbVar = Data.shape[0]

    # Optional regularization term to avoid numerical instability
    if 'params_diagRegFact' not in model:
        model['params_diagRegFact'] = 1E-4

    # Equal binning of the time steps (first row of Data)
    TimingSep = np.linspace(np.min(Data[0, :]), np.max(Data[0, :]), model['nbStates'] + 1)

    model['Priors'] = np.zeros(model['nbStates'])
    model['Mu'] = np.zeros((nbVar, model['nbStates']))
    model['Sigma'] = np.zeros((nbVar, nbVar, model['nbStates']))

    for i in range(model['nbStates']):
        # Find the data points within the current time bin
        idtmp = np.where((Data[0, :] >= TimingSep[i]) & (Data[0, :] < TimingSep[i + 1]))[0]

        # Update the Priors
        model['Priors'][i] = len(idtmp)

        # Update the means (Mu)
        model['Mu'][:, i] = np.mean(Data[:, idtmp], axis=1)

        # Update the covariance matrices (Sigma)
        if len(idtmp) > 1:
            model['Sigma'][:, :, i] = np.cov(Data[:, idtmp], bias=True)
        else:
            model['Sigma'][:, :, i] = np.zeros((nbVar, nbVar))

        # Add regularization term to avoid numerical instability
        model['Sigma'][:, :, i] += np.eye(nbVar) * model['params_diagRegFact']

    # Normalize the priors
    model['Priors'] /= np.sum(model['Priors'])

    return model

# checked
def gaussPDF(Data, Mu, Sigma):
    """
    Compute the likelihood of data points under a Gaussian distribution.

    Parameters:
    - Data:  D x N array representing N datapoints of D dimensions.
    - Mu:    D x 1 vector representing the mean of the Gaussian.
    - Sigma: D x D array representing the covariance matrix of the Gaussian.

    Returns:
    - prob:  1 x N vector representing the likelihood of the N datapoints.
    """

    # Handle cases where Data, Mu, or Sigma are scalars
    if np.isscalar(Data):
        Data = np.array([[Data]])  # Convert scalar to (1, 1)
    if np.isscalar(Mu):
        Mu = np.array([[Mu]])  # Convert scalar to (1, 1)
    if np.isscalar(Sigma) or Sigma.ndim == 1:
        Sigma = np.array([[Sigma]])  # Convert scalar to (1, 1)

    # Check if Data is 1D and reshape it to a 2D array for consistent matrix operations
    if Data.ndim == 1:
        Data = Data.reshape(-1, 1)

    nbVar, nbData = Data.shape

    # Center the data by subtracting the mean Mu
    Data = Data.T - Mu.T  # Transpose to match dimensions for broadcasting

    # print("Sigma shape for gaussPDF:", Sigma.shape)

    # Compute the likelihood using the Gaussian formula
    try:
        prob = np.sum((Data @ np.linalg.inv(Sigma)) * Data, axis=1)
    except np.linalg.LinAlgError:
        print(f"Warning: Singular matrix encountered. Using pseudo-inverse for covariance matrix Sigma.")
        prob = np.sum((Data @ np.linalg.pinv(Sigma)) * Data, axis=1)  # Use pseudo-inverse

    prob = np.exp(-0.5 * prob) / (np.sqrt((2 * np.pi) ** nbVar * np.abs(np.linalg.det(Sigma)) + np.finfo(float).eps))

    return prob


def plotGMM(Mu, Sigma, color, axs, valAlpha=1, linestyle='-', linewidth=0.5, edgeAlpha=None):
    """
    This function displays the parameters of a Gaussian Mixture Model (GMM).

    Args:
    - Mu:         2xK array representing the centers of K Gaussians.
    - Sigma:      2x2xK array representing the covariance matrices of K Gaussians.
    - color:      Array representing the RGB color to use for the display.
    - valAlpha:   Transparency factor (optional, default=1).
    - linestyle:  Line style for ellipse edges (optional, default='-').
    - linewidth:  Line width for ellipse edges (optional, default=0.5).
    - edgeAlpha:  Transparency factor for edges (optional, default=same as valAlpha).

    Returns:
    - h: List of plot handles for the generated patches and points.
    - X: 2xN array of the ellipse points in each Gaussian.
    """
    # print("Mu shape for plotGMM: ", Mu.shape)
    nbStates = Mu.shape[1]
    nbDrawingSeg = 100
    color = np.array(color)
    darkcolor = np.clip(color * 0.5 * 2, 0, 1)  # Darker color for edges
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

    if edgeAlpha is None:
        edgeAlpha = valAlpha

    h = []
    X = np.zeros((2, nbDrawingSeg, nbStates))

    for i in range(nbStates):
        # Eigenvalue decomposition (for covariance matrices)
        if Sigma.ndim == 2:
            Sigma = Sigma[:, :, np.newaxis]
        D, V = np.linalg.eig(Sigma[:, :, i])
        R = np.real(V @ np.diag(np.sqrt(D)))  # Transformation matrix for ellipse

        X[:, :, i] = R @ np.array([np.cos(t), np.sin(t)]) + Mu[:, i, np.newaxis]

        # Plot with transparency
        polygon = Polygon(X[:, :, i].T, closed=True, facecolor=color, edgecolor=darkcolor,
                          alpha=valAlpha, linestyle=linestyle, linewidth=linewidth)
        axs.add_patch(polygon)
        h.append(polygon)

        # Plot the mean point
        point, = axs.plot(Mu[0, i], Mu[1, i], '.', color=darkcolor, markersize=1)
        h.append(point)

    axs.set_aspect('equal')
    plt.draw()

    return h, X


def plotGMM3D(Mu, Sigma, color, axs, alpha=1, dispOpt=1):
    """
    This function displays the parameters of a Gaussian Mixture Model (GMM) in 3D.

    Args:
    - Mu:      3xK array representing the centers of K Gaussians.
    - Sigma:   3x3xK array representing the covariance matrices of K Gaussians.
    - color:   Array representing the RGB color to use for the display.
    - alpha:   Transparency factor (optional, default=1).
    - dispOpt: Display option for the plot (optional, default=1).

    Returns:
    - h: List of plot handles for the generated patches and points.
    """

    nbData = Mu.shape[1]
    nbPoints = 10  # Number of points to form a circular path
    nbRings = 50  # Number of circular paths following the principal direction

    pts0 = np.array([np.cos(np.linspace(0, 2 * np.pi, nbPoints, endpoint=True)),
                     np.sin(np.linspace(0, 2 * np.pi, nbPoints, endpoint=True))])

    h = []
    ax = axs

    for n in range(nbData):
        # Eigenvalue decomposition (for covariance matrices)
        if Sigma.ndim == 2:
            Sigma = Sigma[:, :, np.newaxis]
        D0, V0 = np.linalg.eigh(Sigma[:, :, n])
        U0 = np.real(V0 @ np.diag(np.sqrt(D0)))

        # Generate the circular paths (rings)
        ringpts0 = np.array([np.cos(np.linspace(0, np.pi, nbRings + 1)),
                             np.sin(np.linspace(0, np.pi, nbRings + 1))])
        ringpts = np.zeros((3, nbRings))
        ringpts[1:, :] = ringpts0[:, :-1]
        U = np.zeros((3, 3))
        U[:, 1:] = U0[:, 1:]
        ringTmp = U @ ringpts

        # Compute the circular paths
        xring = np.zeros((3, nbPoints, nbRings))
        for j in range(nbRings):
            U = np.zeros((3, 3))
            U[:, 0] = U0[:, 0]
            U[:, 1] = ringTmp[:, j]
            pts = np.zeros((3, nbPoints))
            pts[:2, :] = pts0
            xring[:, :, j] = U @ pts + Mu[:, n, np.newaxis]

        # Close the ellipsoid
        xringfull = np.concatenate([xring, xring[:, :, [0]]], axis=2)

        # Plot the ellipsoid
        for j in range(nbRings):
            for i in range(nbPoints - 1):
                verts = [
                    [xringfull[:, i, j], xringfull[:, i + 1, j], xringfull[:, i + 1, j + 1], xringfull[:, i, j + 1]]
                    for i in range(nbPoints - 1)
                ]
                color = np.array(color)
                poly = Poly3DCollection(verts, facecolor=np.minimum(color + 0.1, 1), edgecolor=color, alpha=alpha)
                if dispOpt == 1:
                    poly.set_edgecolor(color)
                    poly.set_linewidth(1)
                    poly.set_alpha(alpha)
                elif dispOpt == 2:
                    poly.set_linestyle('none')
                h.append(poly)
                ax.add_collection3d(poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()

    return h

def logmap(X, S):
    """
    This function computes the logarithmic map on the SPD manifold.

    Parameters:
        X: SPD matrix (d x d) or SPD matrices (d x d x N)
        S: Base SPD matrix (d x d)

    Returns:
        U: Symmetric matrix Log_S(X) or symmetric matrices d x d x N
    """

    # Determine if X is a single matrix or a batch of matrices
    if X.ndim == 3:
        N = X.shape[2]  # Number of matrices
        D = X.shape[0]  # Dimension of each matrix
        U = np.zeros((D, D, N), dtype=np.float64) # Initialize the output array for multiple matrices

        # Loop over each matrix in X
        for n in range(N):
            # Compute the matrix S^-1 * X[:,:,n]
            S_inv_X = np.linalg.solve(S, X[:, :, n])

            # Perform the eigendecomposition of S^-1 * X[:,:,n]
            d, v = eig(S_inv_X)

            d = np.real(d)
            eps = 1e-8
            d[d < eps] = eps

            # Take the logarithm of the eigenvalues
            # log_eigvals = np.diag(np.log(np.diag(np.real(d))))
            log_eigvals = np.diag(np.log(d))

            # Reconstruct the log map using the eigendecomposition
            tem = S @ v @ log_eigvals @ np.linalg.inv(v)
            U[:, :, n] = S @ v @ log_eigvals @ np.linalg.inv(v)

    else:
        S_inv_X = np.linalg.solve(S, X)

        # Perform the eigendecomposition of S^-1 * X
        d, v = eig(S_inv_X)

        d = np.real(d)
        eps = 1e-8
        d[d < eps] = eps

        # Take the logarithm of the eigenvalues
        # log_eigvals = np.diag(np.log(np.diag(np.real(d))))
        log_eigvals = np.diag(np.log(d))

        # Reconstruct the log map using the eigendecomposition
        U = S @ v @ log_eigvals @ np.linalg.inv(v)

    return U


def logmap_vec(x, s):
    """
    This function computes the logarithmic map on the SPD manifold with
    Mandel notation.

    Parameters:
    - x: SPD matrix in vector form or SPD matrices d x d x N in vector form.
    - s: Base SPD matrix in vector form.

    Returns:
    - u: Symmetric matrix Log_S(X) in vector form or symmetric matrices d x d x N in vector form.
    """

    # Convert vectorized SPD matrices back to symmetric matrices
    X = vec2symmat(x)
    S = vec2symmat(s)

    # Compute the logarithmic map Log_S(X) on the SPD manifold
    U = logmap(X, S)

    # Convert the result back to vector form
    u = symmat2vec(U)

    return u

def expmap(U, S):
    """
    Computes the exponential map on the SPD manifold.

    Parameters:
    - U: Symmetric matrix on the tangent space of S, shape (d, d, N)
    - S: Base SPD matrix, shape (d, d)

    Returns:
    - X: SPD matrix Exp_S(U), shape (d, d, N)
    """

    if U.ndim == 3:

        # Get number of matrices N
        N = U.shape[2]

        # Initialize the result matrix X
        X = np.zeros_like(U)

        # Loop over each U[:,:,n]
        for n in range(N):
            # Solve the system S * X = U[:,:,n] => equivalent to S\U(:,:,n)
            Sinv_U = np.linalg.solve(S, U[:, :, n])

            # Perform eigenvalue decomposition
            d, v = eig(Sinv_U)

            # Exponentiate the eigenvalues and construct the result
            exp_d = np.diag(np.exp(np.real(d)))  # Apply the exponential to eigenvalues
            X[:, :, n] = S @ v @ exp_d @ inv(v)  # Compute the SPD result

    else:
        # Solve the system S * X = U[:,:,n] => equivalent to S\U(:,:,n)
        Sinv_U = np.linalg.solve(S, U)
        # Perform eigenvalue decomposition
        d, v = eig(Sinv_U)

        # Exponentiate the eigenvalues and construct the result
        exp_d = np.diag(np.exp(np.real(d)))  # Apply the exponential to eigenvalues
        X = S @ v @ exp_d @ inv(v)  # Compute the SPD result

    return X


# checked
def expmap_vec(u, s):
    """
    Compute the exponential map on the SPD manifold with Mandel notation.

    Parameters:
    - u: Symmetric matrix in vector form (Mandel notation) or d x d x N in vector form.
    - s: Base SPD matrix in vector form (Mandel notation).

    Returns:
    - x: SPD matrix Exp_S(U) in vector form (Mandel notation) or d x d x N in vector form.
    """
    U = vec2symmat(u)
    S = vec2symmat(s)
    X = expmap(U, S)
    x = symmat2vec(X)
    return x




# checked
def symmat2vec(M):
    """
    This function computes a vectorization of SPD matrices using Mandel
    notation.

    Parameters:
    - M: SPD matrix or SPD matrices of size d x d x N

    Returns:
    - v: Vectorized SPD matrix or vectorized SPD matrices of size d' x N
    """
    # Check if M is 2D or 3D
    if M.ndim == 2:
        N = M.shape[0]
        v = np.diag(M)  # Diagonal elements

        # Loop over the upper diagonals (off-diagonal elements)
        for n in range(1, N):
            v = np.concatenate((v, np.sqrt(2) * np.diag(M, n)))  # Mandel notation
            # v = np.concatenate((v, np.diag(M, n)))  # Voigt notation (if needed)

    elif M.ndim == 3:
        D, _, N = M.shape
        v = np.empty((0, N))  # Initialize empty array

        # Loop through the third dimension (N matrices)
        for n in range(N):
            vn = np.diag(M[:, :, n])  # Diagonal elements of the nth matrix

            # Loop over the upper diagonals (off-diagonal elements)
            for d in range(1, D):
                vn = np.concatenate((vn, np.sqrt(2) * np.diag(M[:, :, n], d)))  # Mandel notation
                # vn = np.concatenate((vn, np.diag(M[:, :, n], d)))  # Voigt notation (if needed)

            v = np.column_stack((v, vn)) if v.size else vn[:, np.newaxis]

    else:
        # Handle case where M is not 2D or 3D
        raise ValueError(f"Input matrix M must be 2D or 3D, but got {M.ndim}D")

    return v


# checked
def vec2symmat(v):
    """
    This function computes SPD matrices based on a vector using Mandel notation.

    Parameters:
        v: Vectorized SPD matrix or vectorized SPD matrices (d' x N)

    Returns:
        M: SPD matrix (d x d) or SPD matrices (d x d x N)
    """
    if v.ndim == 1:
        # Case for a single vectorized SPD matrix
        n = v.shape[0]
        N = int((-1 + np.sqrt(1 + 8 * n)) // 2)
        M = np.diag(v[:N])
        id = np.cumsum(np.flip(np.arange(1, N+1)))

        for i in range(N - 1):
            # M += np.diag(v[id[i]:id[i + 1]] / np.sqrt(2), k=i)  # upper diagonal
            # M += np.diag(v[id[i]:id[i + 1]] / np.sqrt(2), k=-i)  # lower diagonal
            M += np.diag(v[id[i]:id[i+1]] / np.sqrt(2), k=i+1)  # Upper diagonal
            M += np.diag(v[id[i]:id[i+1]] / np.sqrt(2), k=-(i+1))  # Lower diagonal

    else:
        # Case for multiple vectorized SPD matrices
        d, N = v.shape
        D = int((-1 + np.sqrt(1 + 8 * d)) // 2)
        M = np.zeros((D, D, N))

        for n in range(N):
            vn = v[:, n]
            Mn = np.diag(vn[:D])
            id = np.cumsum(np.flip(np.arange(1, D+1)))

            for i in range(D - 1):
                Mn += np.diag(vn[id[i]:id[i+1]] / np.sqrt(2), k=i + 1)  # Upper diagonal
                Mn += np.diag(vn[id[i]:id[i+1]] / np.sqrt(2), k=-(i + 1))  # Lower diagonal

            M[:, :, n] = Mn

    return M


def is_spd(matrix):
    """Check if a matrix is symmetric and positive definite (SPD)."""
    if not np.allclose(matrix, matrix.T):
        return False
    eigvals = np.linalg.eigvals(matrix)
    return np.all(eigvals > 0)


def transp_operator(S1, S2):
    """
    Compute the parallel transport operator from S1 to S2 on the SPD manifold.

    A SPD matrix X is transported from S1 to S2 with Ac * X * Ac'.

    Parameters:
    - S1: SPD matrix of shape (d, d).
    - S2: SPD matrix of shape (d, d).

    Returns:
    - Ac: Parallel transport operator of shape (d, d).
    """
    # Check if S1 and S2 are SPD matrices
    # if not is_spd(S1):
    #     raise ValueError("S1 is not a valid SPD matrix.")
    # if not is_spd(S2):
    #     raise ValueError("S2 is not a valid SPD matrix.")

    try:
        # Try computing the inverse of S1
        S1_inv = np.linalg.inv(S1)
    except np.linalg.LinAlgError:
        print("Warning: S1 is singular, using pseudo-inverse.")
        S1_inv = np.linalg.pinv(S1)

    # Compute the product S2 * S1_inv
    prod = S2 @ S1_inv

    # Compute the matrix square root of the product
    Ac = sqrtm(prod)

    return Ac

# checked
def spdMean(setS, nbIt=10):
    """
    This function computes the mean of SPD matrices on the SPD manifold.

    Parameters:
    - setS: Set of SPD matrices of shape (d, d, N), where d is the dimension of each matrix,
            and N is the number of matrices.
    - nbIt: Number of iterations for the Gauss-Newton algorithm (default is 10).

    Returns:
    - M: Mean SPD matrix.
    """

    # Initialize M with the first matrix in the set
    M = setS[:, :, 0]

    # Gauss-Newton iteration
    for _ in range(nbIt):
        L = np.zeros_like(M)

        # Sum of logm of M^(-0.5) * setS_n * M^(-0.5)
        M_sqrt_inv = np.linalg.inv(sqrtm(M)) # M^(-0.5)
        M_sqrt_inv = np.real(M_sqrt_inv)
        for n in range(setS.shape[2]):
            spd_matrix = M_sqrt_inv @ setS[:, :, n] @ M_sqrt_inv
            # print("spd_matrix: ", spd_matrix)
            if is_spd(spd_matrix):
                log_spd_matrix = logm(spd_matrix)
                L += log_spd_matrix
            else:
                raise ValueError(f"Matrix {n} is not symmetric positive definite (SPD)")
            # L += np.real(logm(M_sqrt_inv @ setS[:, :, n] @ M_sqrt_inv))

        # Update the mean matrix
        M = sqrtm(M) @ expm(L / setS.shape[2]) @ sqrtm(M)

    return M

# checked
def spd_init_GMM_kbins(Data, model, nbSamples, spdDataId=None):
    """
    This function computes K-Bins initialization for GMM on the SPD manifold.

    Parameters:
    - Data:        Set of data in vector form. Some dimensions of these
                   data are SPD data expressed in Mandel notation. The
                   remaining part of the data is Euclidean.
    - model:       Model variable (dictionary).
    - nbSamples:   Number of distinct demonstrations.
    - spdDataId:   Indices of SPD data in Data.

    Returns:
    - model:       Initialized model (dictionary).
    """

    nbData = Data.shape[1] // nbSamples

    # Optional regularization term to avoid numerical instability
    if 'params_diagRegFact' not in model:
        model['params_diagRegFact'] = 1E-4

    # Delimit the cluster bins for the first demonstration
    tSep = np.round(np.linspace(0, nbData, model['nbStates'] + 1)).astype(int)

    # Initialize Priors, MuMan, and Sigma if not already initialized
    model['Priors'] = np.zeros(model['nbStates'])
    model['MuMan'] = np.zeros((Data.shape[0], model['nbStates']))
    model['Sigma'] = np.zeros((Data.shape[0], Data.shape[0], model['nbStates']))

    # Compute statistics for each bin
    for i in range(model['nbStates']):
        id_list = []

        # Collect indices for each bin
        for n in range(nbSamples):
            id_list.extend((n * nbData + np.arange(tSep[i], tSep[i + 1])).tolist())

        model['Priors'][i] = len(id_list)

        # Mean computed on SPD manifold for parts of the data belonging to the manifold
        if spdDataId is None:
            model['MuMan'][:, i] = symmat2vec(spdMean(vec2symmat(Data[:, id_list])))
        else:
            model['MuMan'][:, i] = np.mean(Data[:, id_list], axis=1)
            if isinstance(spdDataId, list):
                for c in range(len(spdDataId)):
                    model['MuMan'][spdDataId[c], i] = symmat2vec(
                        spdMean(vec2symmat(Data[spdDataId[c], id_list]), 3))
            else:
                model['MuMan'][spdDataId, i] = symmat2vec(
                    spdMean(vec2symmat(Data[np.ix_(spdDataId, id_list)]), 3))

        # Parts of data belonging to SPD manifold projected to tangent space at the mean
        DataTgt = np.zeros_like(Data[:, id_list])

        if spdDataId is None:
            DataTgt = logmap_vec(Data[:, id_list], model['MuMan'][:, i])
        else:
            DataTgt = Data[:, id_list]
            if isinstance(spdDataId, list):
                for c in range(len(spdDataId)):
                    DataTgt[spdDataId[c], :] = logmap_vec(
                        Data[spdDataId[c], id_list], model['MuMan'][spdDataId[c], i])
            else:
                DataTgt[spdDataId, :] = logmap_vec(
                    Data[np.ix_(spdDataId, id_list)], model['MuMan'][spdDataId, i])

        # Compute covariance in tangent space and regularize
        model['Sigma'][:, :, i] = np.cov(DataTgt, rowvar=True) + np.eye(model['nbVarVec']) * model['params_diagRegFact']

    # Normalize Priors
    model['Priors'] /= np.sum(model['Priors'])

    return model


def GMR(model, DataIn, in_idx, out_idx):
    """
    Gaussian Mixture Regression (GMR) implementation in Python.

    Parameters:
    - model: A dictionary containing the GMM parameters (Priors, Mu, Sigma, nbStates)
    - DataIn: Input data (D x N)
    - in_idx: Indices of input variables
    - out_idx: Indices of output variables

    Returns:
    - expData: Expected output data
    - expSigma: Expected covariance matrices of the output
    - H: Activation weights
    """

    if np.isscalar(DataIn):
        nbData =  1
    else:
        shape = np.shape(DataIn)
        if len(shape) < 2:
            nbData =  1
        else:  # 返回第二维度的大小
            nbData =  shape[1]

    # nbData = DataIn.shape[1]
    nbVarOut = len(out_idx)

    if 'params_diagRegFact' not in model:
        model['params_diagRegFact'] = 1E-8  # Regularization term

    expData = np.zeros((nbVarOut, nbData),  dtype=np.float64)
    expSigma = np.zeros((nbVarOut, nbVarOut, nbData),  dtype=np.float64)
    H = np.zeros((model['nbStates'], nbData),  dtype=np.float64)
    MuTmp = np.zeros((nbVarOut, model['nbStates']),  dtype=np.float64)

    for t in range(nbData):
        # Compute activation weight
        for i in range(model['nbStates']):
            H[i, t] = np.float64(model['Priors'][i] * gaussPDF(DataIn, model['Mu'][in_idx, i],
                                                    model['Sigma'][in_idx, in_idx, i])[0])
        H[:, t] = H[:, t] / (np.sum(H[:, t]) + np.finfo(np.float64).eps)

        # Compute conditional means
        for i in range(model['nbStates']):
            MuTmp[:, i] = (model['Mu'][out_idx, i] + model['Sigma'][out_idx, in_idx, i] / model['Sigma'][in_idx, in_idx, i] * (DataIn - model['Mu'][in_idx, i]))

            expData[:, t] += H[i, t] * MuTmp[:, i]

        # Compute conditional covariances
        for i in range(model['nbStates']):
            SigmaTmp = model['Sigma'][np.ix_(out_idx, out_idx, [i])].squeeze() - (model['Sigma'][out_idx, in_idx, i].reshape(2,1) / model['Sigma'][in_idx, in_idx, i] * (model['Sigma'][in_idx, out_idx, i].reshape(1,2)))
            expSigma[:, :, t] += H[i, t] * (SigmaTmp + np.outer(MuTmp[:, i], MuTmp[:, i]))
            # expSigma[:, :, t] += H[i, t] * (SigmaTmp + MuTmp[:, i] @ MuTmp[:, i].T)


        expSigma[:, :, t] = expSigma[:, :, t] - np.outer(expData[:, t], expData[:, t]) + np.eye(nbVarOut) * model['params_diagRegFact']

        # expSigma[:, :, t] = expSigma[:, :, t] - expData[:, t] @ expData[:, t].T + np.eye(nbVarOut) * model[
        #     'params_diagRegFact']

        expData = expData.squeeze()
        expSigma = expSigma.squeeze()

    return expData, expSigma, H

# checked
def compute_gamma(data, model):
    """
    Compute the responsibility matrix (GAMMA) and likelihood (L) for the given data and model.

    Parameters:
    data : numpy array
        The input data.
    model : dictionary
        The GMM model containing the parameters Priors, Mu, and Sigma.

    Returns:
    L : numpy array
        The likelihood of each data point for each state.
    GAMMA : numpy array
        The responsibility matrix.
    """
    nb_states = model['nbStates']
    L = np.zeros((nb_states, data.shape[1]))

    for i in range(nb_states):
        L[i, :] = model['Priors'][i] * gaussPDF(data, model['Mu'][:, i], model['Sigma'][:, :, i])

    GAMMA = L / (np.sum(L, axis=0) + np.finfo(float).eps)

    return L, GAMMA

# checked
def EM_GMM(data, model):
    """
    Training of a Gaussian mixture model (GMM) with an expectation-maximization (EM) algorithm.

    Parameters:
    data : numpy array
        The input data for the GMM.
    model : dictionary
        The GMM model containing the parameters Priors, Mu, Sigma, and other algorithm settings.

    Returns:
    model : dictionary
        The updated GMM model after training.
    GAMMA2 : numpy array
        The normalized responsibilities.
    LL : list
        Log-likelihood values over iterations.
    """

    nb_data = data.shape[1]

    # Set default parameters if not provided in the model
    model.setdefault('params_nbMinSteps', 5)  # Minimum number of iterations allowed
    model.setdefault('params_nbMaxSteps', 100)  # Maximum number of iterations allowed
    model.setdefault('params_maxDiffLL', 1e-4)  # Likelihood increase threshold to stop the algorithm
    model.setdefault('params_diagRegFact', 1e-4)  # Regularization term
    model.setdefault('params_updateComp', np.ones(3))  # pi, Mu, Sigma

    LL = []

    for nb_iter in range(model['params_nbMaxSteps']):
        print('.', end='')

        # E-step
        L, GAMMA = compute_gamma(data, model)
        # GAMMA2 = GAMMA / np.sum(GAMMA, axis=0)
        GAMMA2 = GAMMA / np.sum(GAMMA, axis=1, keepdims=True)

        # M-step
        for i in range(model['nbStates']):
            # Update Priors
            if model['params_updateComp'][0]:
                model['Priors'][i] = np.sum(GAMMA[i, :]) / nb_data

            # Update Mu
            if model['params_updateComp'][1]:
                model['Mu'][:, i] = np.dot(data, GAMMA2[i, :])

            # Update Sigma
            if model['params_updateComp'][2]:
                data_tmp = data - model['Mu'][:, i].reshape(-1, 1)
                model['Sigma'][:, :, i] = np.dot(data_tmp * GAMMA2[i, :], data_tmp.T) + np.eye(data.shape[0]) * model[
                    'params_diagRegFact']

        # Compute average log-likelihood
        LL.append(np.sum(np.log(np.sum(L, axis=0))) / nb_data)

        # Stop the algorithm if EM converged (small change of LL)
        if nb_iter > model['params_nbMinSteps']:
            if (LL[nb_iter] - LL[nb_iter - 1] < model['params_maxDiffLL']) or (
                    nb_iter == model['params_nbMaxSteps'] - 1):
                print(f'\nEM converged after {nb_iter + 1} iterations.')
                return model

def rotate_points(points, axis, angle_deg, origin):
    """
    Rotate points around a specified axis by a given angle (degrees),
    with respect to a user-defined origin.

    Parameters:
        points (ndarray): Array of shape (n, 3) representing the points to rotate.
        axis (ndarray): Array of shape (3,) representing the rotation axis.
        angle_deg (float): Rotation angle in degrees.
        origin (ndarray): Array of shape (3,) representing the rotation center.

    Returns:
        rotated_points (ndarray): Rotated points as an array of shape (n, 3).
    """
    # Ensure input shapes are correct
    points = np.array(points)  # Shape: (n, 3)
    axis = np.array(axis)  # Shape: (3,)
    origin = np.array(origin).reshape(3,1)  # Shape: (3,)

    # Step 1: Translate points to the origin
    points_centered = points - origin

    # Step 2: Create the rotation object using the axis and angle
    r = R.from_rotvec(np.radians(angle_deg) * axis)

    # Step 3: Apply the rotation
    rotated_centered = r.apply(points_centered.T).T  # Rotate around the origin

    # Step 4: Translate points back to their original position
    rotated_points = rotated_centered + origin

    return rotated_points

def create_cone_data(r=200, num_points=100):
    phi = np.arange(0, 2 * np.pi + 0.1, 0.1)
    xax = np.array([np.zeros_like(phi), r * np.ones_like(phi)])
    yax = np.array([np.zeros_like(phi), r * np.sin(phi)])
    zax = np.array([np.zeros_like(phi), r / np.sqrt(2) * np.cos(phi)])
    return xax, yax, zax


def plot_cone(ax, xax, yax, zax, direction, angle_deg):
    """

    在给定的 3D 轴上绘制操控性椭圆体的圆锥体。

    Parameters:
    - ax: matplotlib 的 3D 轴
    - xax, yax, zax: 定义圆锥的 X, Y, Z 数据
    - direction: 旋转方向向量
    - angle_deg: 旋转角度
    """
    # 绘制圆锥表面
    # 将 xax, yax, zax 转换为 3D 点的列表
    surface_points = np.vstack((xax.flatten(), yax.flatten(), zax.flatten()))

    # 旋转圆锥表面的所有点
    rotated_surface = rotate_points(surface_points, direction, angle_deg, origin=[0, 0, 0])

    # 重新调整形状到原始的网格形状
    X_rot = rotated_surface[0, :].reshape(xax.shape)
    Y_rot = rotated_surface[1, :].reshape(yax.shape)
    Z_rot = rotated_surface[2, :].reshape(zax.shape)

    # 绘制旋转后的圆锥表面
    ax.plot_surface(X_rot, Y_rot, Z_rot, color=[0.95, 0.95, 0.95], alpha=0.5)

    # 绘制圆锥边缘线
    edge_line = np.vstack((xax[1, :], yax[1, :], zax[1, :]))
    rotated_edge = rotate_points(edge_line, direction, angle_deg, origin=[0, 0, 0])
    ax.plot(rotated_edge[0, :], rotated_edge[1, :], rotated_edge[2, :], color='black', linewidth=3)

    # 绘制生成线
    for idx in [63, 40]:
        generator_line = np.vstack((xax[:, idx], yax[:, idx], zax[:, idx]))
        rotated_gen_line = rotate_points(generator_line, direction, angle_deg, origin=[0, 0, 0])
        ax.plot(rotated_gen_line[0, :], rotated_gen_line[1, :], rotated_gen_line[2, :], color='black', linewidth=3)

    # 绘制坐标轴
    ax.plot([0, 250], [0, 0], [0, 0], 'k-', linewidth=0.5)
    ax.plot([0, 0], [0, 250], [0, 0], 'k-', linewidth=0.5)
    ax.plot([0, 0], [0, 0], [0, 150], 'k-', linewidth=0.5)

    # 添加文本标签
    ax.text(280, -40, 0, r'$\mathbf{M}_{11}$', fontsize=20, ha='center')
    ax.text(15, 0, 120, r'$\mathbf{M}_{12}$', fontsize=20, ha='center')
    ax.text(5, 220, -15, r'$\mathbf{M}_{22}$', fontsize=20, ha='center')


def skew_symmetric_matrix(vector):
    """
    Compute the skew-symmetric matrix of a 3D vector.
    """
    x, y, z = vector
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])

def compute_bimanual_relative_jacobian(J_left, J_right, R_21, R_24, p_23, task_dim, scaling=False):
    """
    Compute the relative Jacobian matrix JR.
    """
    # calculate wrench transformation matrix
    S = skew_symmetric_matrix(p_23)
    I = np.eye(3)
    psi = np.block([[I, -S], [np.zeros((3, 3)), I]])

    # calculate rotation matrices
    omega_21 = np.block([[R_21, np.zeros((3, 3))], [np.zeros((3, 3)), R_21]])
    omega_24 = np.block([[R_24, np.zeros((3, 3))], [np.zeros((3, 3)), R_24]])

    # calculate Jacobian matrices
    JR = np.block([[-psi @ omega_21 @ J_left, omega_24 @ J_right]])[:task_dim, :]  # take desired task dimension of the Jacobian
    # JR = np.block([[-psi @ omega_21 @ J_left, omega_24 @ J_right]])[[0, 2], :] # take only xz translational part

    if scaling == True:
        # scale the Jacobian matrix
        auxJ = np.array([[6.6028, 2.1570, 1.2760],
                         [7.4681, 9.7562, 4.8345],
                         [0, 0, 0]])

        # Compute the scaling factor
        # Frobenius norm scaling
        scale = np.linalg.norm(auxJ, 'fro') / np.linalg.norm(JR, 'fro')

        # Apply scaling to JR
        JR = scale * JR

    return JR

def compute_bimanual_relative_manipulability(JR, W_rel):
    """
    Compute the relative manipulability ellipsoid of a dual-arm robot system given its relative Jacobian and weight matrix.
    """
    # Compute the manipulability ellipsoid of the dual-arm robot system via the inverse-based formulation
    # JR_W = JR @ W_rel @ JR.T
    # manipulability_ellipsoid = np.linalg.inv(JR_W)

    # Compute the manipulability ellipsoid of the dual-arm robot system via the direct formulation
    M = JR @ W_rel @ JR.T

    return M

def plot_ellipsoid(M, ax, center=np.zeros(3), scale=0.05, color='b'):
    """
    绘制 3D 椭球。

    参数:
    - M: 3x3 的对称正定矩阵（如可操作性椭球矩阵）。
    - ax: matplotlib 的 3D 轴对象。
    - center: 椭球中心（默认为原点）。
    - scale: 椭球缩放比例，用于控制椭球的整体大小。
    - color: 椭球颜色。
    """
    # 特征值和特征向量分解
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, 1e-6)  # 防止特征值过小，导致数值不稳定

    # 将特征值进行归一化处理，确保椭球大小受控
    eigenvalues /= np.max(eigenvalues)

    # 生成单位球体点云（用于构造三维椭球）
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # 将单位球体扩展为椭球
    unit_sphere = np.stack([x, y, z], axis=-1)  # 形状为 (50, 50, 3)
    scaling_matrix = np.diag(np.sqrt(eigenvalues)) * scale
    ellipsoid = np.einsum('ij,...j->...i', eigenvectors @ scaling_matrix, unit_sphere)

    # 平移到指定中心
    ellipsoid += center

    # 绘制椭球
    ax.plot_surface(ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2], color=color, alpha=0.2)


def visualize_trajectory_with_ellipsoids(poses_left, poses_right, manipulability_matrices, scale=0.01, step=40):
    """
    可视化双臂机器人的三维运动轨迹以及对应的可操作性椭球。

    参数:
    - poses_left: 左臂轨迹点，每个点为一个 SE(3) 位姿 (4x4 矩阵)。
    - poses_right: 右臂轨迹点，每个点为一个 SE(3) 位姿 (4x4 矩阵)。
    - manipulability_matrices: 双臂的可操作性矩阵列表，每对点包含两个 6x6 矩阵 (左臂和右臂)。
    - scale: 可操作性椭球的缩放因子。
    """
    # 提取左臂和右臂的位置信息
    left_positions = np.array([pose.t for pose in poses_left])
    right_positions = np.array([pose.t for pose in poses_right])

    # 创建 3D 图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制左臂和右臂的轨迹（虚线连接）
    ax.plot(left_positions[::step, 0], left_positions[::step, 1], left_positions[::step, 2], 'r--', linewidth=1.5, label='Left Arm Trajectory')
    ax.plot(right_positions[::step, 0], right_positions[::step, 1], right_positions[::step, 2], 'b--', linewidth=1.5,
            label='Right Arm Trajectory')

    # 绘制左臂和右臂的轨迹点（散点）
    ax.scatter(left_positions[::step, 0], left_positions[::step, 1], left_positions[::step, 2], c='r', marker='o')
    ax.scatter(right_positions[::step, 0], right_positions[::step, 1], right_positions[::step, 2], c='b', marker='o')

    # 绘制左臂和右臂的轨迹点（散点）
    ax.scatter(left_positions[0, 0], left_positions[0, 1], left_positions[0, 2], c='y', marker='^', label='Start Point')
    # ax.scatter(left_positions[-1, 0], left_positions[-1, 1], left_positions[-1, 2], c='y', marker='2', label='End Point')
    ax.scatter(right_positions[0, 0], right_positions[0, 1], right_positions[0, 2], c='y', marker='^', label='Start Point')
    # ax.scatter(right_positions[-1, 0], right_positions[-1, 1], right_positions[-1, 2], c='y', marker='2', label='End Point')



    # 绘制每个轨迹点的可操作性椭球
    for i in range(0, len(poses_left), step):
        pose_left = poses_left[i]
        pose_right = poses_right[i]

        # 计算椭球中心（左右臂轨迹点的中点）
        ellipsoid_center = (pose_left.t + pose_right.t) / 2

        # 提取对应的平移可操作性矩阵
        manipulability_matrix = manipulability_matrices[i][:3, :3]

        # 修正：归一化可操作性矩阵以防止椭球过大或过小
        manipulability_matrix /= np.linalg.norm(manipulability_matrix)

        # 绘制双臂的可操作性椭球
        plot_ellipsoid(manipulability_matrix, ax, center=ellipsoid_center, scale=scale, color='g')

    # 设置图形属性
    ax.set_title("Dual-Arm Trajectory and Relative Manipulability Ellipsoids")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


# 6x6 矩阵分解为位置和姿态部分
def decompose_manipulability_matrix(M6x6):
    """
    分解 6x6 矩阵为位置和姿态部分。
    """
    M_position = M6x6[:3, :3]  # 位置部分
    M_orientation = M6x6[3:, 3:]  # 姿态部分
    return M_position, M_orientation

def load_traj(file_path):
    """
    load the trajectory data from a pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def scale_to_range(data, new_min, new_max):
    """将数据缩放到指定范围 [new_min, new_max]"""
    old_min = np.min(data)
    old_max = np.max(data)
    scaled_data = new_min + (data - old_min) * (new_max - new_min) / (old_max - old_min)
    return scaled_data


if __name__ == "__main__":
    # vec2symmat function testing
    # M_original = np.array([[4, 1],
    #                        [1, 3]])  # a simple 2x2 SPD matrix
    #
    # print("Original matrix M:\n", M_original)
    #
    # vec_M =symmat2vec(M_original)
    # print("Vectorized M:\n", vec_M)
    #
    # M_reconstructed = vec2symmat(vec_M)
    # print("Reconstructed matrix M:\n", M_reconstructed)
    #
    # if is_spd(M_reconstructed):
    #     print("Reconstructed matrix is symmetric positive definite (SPD).")
    # else:
    #     print("Reconstructed matrix is not symmetric positive definite (SPD).")
    #
    # v1 = np.array([4, 3, 2, np.sqrt(2) * 1, np.sqrt(2) * 1, np.sqrt(2) * 1])
    # v2 = np.array([5, 6, 7, np.sqrt(2) * 2, np.sqrt(2) * 2, np.sqrt(2) * 2])
    # V = np.column_stack((v1, v2))  # 将两个矢量化矩阵合并成一个矩阵
    #
    # M_reconstructed_multi = vec2symmat(V)
    #
    # print("\nReconstructed Matrices (for multiple inputs):")
    # print(M_reconstructed_multi)

    # logmap function testing
    # # Define a 2x2 SPD matrix X
    # X = np.array([[4, 1], [1, 3]])
    #
    # # Define a base SPD matrix S (identity matrix)
    # S = np.eye(2)
    #
    # # Test logmap with a single SPD matrix
    # U_single = logmap(X, S)
    # print("Logarithmic map for a single SPD matrix:")
    # print(U_single)
    #
    # # Define multiple (2) SPD matrices X
    # X_multi = np.stack([X, 2 * X], axis=-1)  # Stack two SPD matrices along the
    # # third dimension
    # print("\nMultiple SPD matrices X:\n", X_multi[:,:,1])
    #
    # # Test logmap with multiple SPD matrices
    # U_multi = logmap(X_multi, S)
    # print("\nLogarithmic map for multiple SPD matrices:")
    # print(U_multi[:,:,1])

    # expmap function testing
    # Define a 2x2 SPD matrix S (identity matrix)
    # S = np.eye(2)
    # # Define a 2x2 matrix U
    # U = np.array([[0.5, 0.5], [0.5, 0.5]])
    # # Test expmap with a single SPD matrix
    # # X_single = expmap(U, S)
    # # print("Exponential map for a single SPD matrix:")
    # # print(X_single)
    # # Define multiple (2) matrices U
    # U_multi = np.stack([U, 2 * U], axis=-1)  # Stack two matrices along the third dimension
    # print("\nMultiple matrices U:\n", U_multi[:,:,1])
    # # Test expmap with multiple matrices
    # X_multi = expmap(U_multi, S)
    # print("\nExponential map for multiple matrices:")
    # print(X_multi[:,:,0])


    # manipulability plot 3d testing
    # 示例 6x6 矩阵
    # M6x6 = np.random.rand(6, 6)
    # M6x6 = M6x6 @ M6x6.T  # 确保对称正定
    #
    # # 分解为位置和姿态部分
    # M_position, M_orientation = decompose_manipulability_matrix(M6x6)
    #
    # # 绘制 3D 椭球
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制位置部分椭球
    # plot_ellipsoid(M_position, ax, center=np.array([0, 0, 0]), scale=1.0, color='r')
    #
    # # 设置图形样式
    # ax.set_title("Manipulability Ellipsoid (Position Part)")
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_zlabel("Z-axis")
    # plt.show()

    # plotting trajectory with ME testing
    # 示例输入数据
    # 随机生成 10 个 SE(3) 位姿（左臂和右臂）
    # 示例输入数据
    # 随机生成 10 个 SE(3) 位姿（左臂和右臂）
    np.random.seed(42)
    poses_left = [np.eye(4) + np.random.randn(4, 4) * 0.05 for _ in range(10)]
    poses_right = [np.eye(4) + np.random.randn(4, 4) * 0.05 for _ in range(10)]

    # 随机生成 10 对双臂的 6x6 可操作性矩阵
    manipulability_matrices = [(np.random.rand(6, 6) @ np.random.rand(6, 6).T) for _ in range(10)]

    # 可视化
    visualize_trajectory_with_ellipsoids(poses_left, poses_right, manipulability_matrices, scale=0.05)




