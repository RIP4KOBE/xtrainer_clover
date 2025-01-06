import numpy as np
from scipy.linalg import block_diag
import roboticstoolbox as rtb
from urdfpy import URDF
from spatialmath import SE3
from ModelTrain.manipulability.manip_utils import *
from ModelTrain.manipulability.sqrtm import sqrtm


import matplotlib.pyplot as plt
import pytorch_kinematics as pk
import pickle
import os
import torch

# Set random seed for reproducibility
np.random.seed(42)

class BimanualManipulabilityLearning:
    def __init__(self):
        """
        Parameter Initialization and Create Robots
        """
        # Parameter Initialization
        self.nbIter = 10  # Number of iterations for Gauss-Newton
        self.nbIterEM = 10  # Number of iterations for EM
        self.modelPD = {
            'nbStates': 5,  # Number of Gaussian components
            'nbVar': 4,  # Input + 3x3 SPD output
            'nbVarOut': 3,  # Dimensionality of SPD matrices (output)
            'dt': 1E-1,  # Time step
            'params_diagRegFact': 1E-4,  # Regularization factor for covariance
        }
        self.modelPD['nbVarOutVec'] = self.modelPD['nbVarOut'] + self.modelPD['nbVarOut'] * (self.modelPD['nbVarOut'] - 1) // 2
        self.modelPD['nbVarVec'] = self.modelPD['nbVar'] - self.modelPD['nbVarOut'] + self.modelPD['nbVarOutVec']
        self.modelPD['nbVarCovOut'] = self.modelPD['nbVar'] + self.modelPD['nbVar'] * (self.modelPD['nbVar'] - 1) // 2

        # Create Robots
        self.urdf = "/home/zhuoli/dobot_xtrainer/ModelTrain/manipulability/data/urdf/nova2_robot.urdf"
        self.left_arm = rtb.robot.ERobot.URDF(self.urdf)
        self.right_arm = rtb.robot.ERobot.URDF(self.urdf)

        # Relative parameters for bimanual manipulability
        self.manipulability_params = {
            "R_21": np.eye(3),  # Left arm: ee to base
            "R_24": np.eye(3),  # Relative rotation: left ee to right base
            "R_34": np.eye(3),  # Right arm: ee to base
            "p_21": np.array([0, 0, 0]),  # Left ee to base
            "p_23": np.array([0, 0, 0]),  # Left ee to right ee
            "T_14": np.eye(4),  # Left ee to right ee
            "R_14": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "p_14": np.array([1.08, 0, 0]),
            "W_rel": np.eye(12),  # Relative weight matrix
        }

        # Update T_14 based on R_14 and p_14
        self.manipulability_params["T_14"][:3, :3] = self.manipulability_params["R_14"]
        self.manipulability_params["T_14"][:3, 3] = self.manipulability_params["p_14"]
        self.manipulability_params["T_14"] = SE3(self.manipulability_params["T_14"])

        print("Bimanual manipulability learning process successfully initialized.")

    def get_param(self, name):
        """Dynamically access manipulability parameters by name."""
        return self.manipulability_params.get(name)

    def set_param(self, name, value):
        """Dynamically update manipulability parameters by name."""
        if name in self.manipulability_params:
            self.manipulability_params[name] = value
        else:
            raise KeyError(f"{name} is not a valid manipulability parameter.")

    def compute_bimanual_jacobian(self, q_left, q_right):
        '''
        Compute the Jacobians for left and right arms
        '''

        # Compute the Jacobians for left and right arms
        J_left = self.left_arm.jacob0(q_left)
        J_right = self.right_arm.jacob0(q_right)

        return J_left, J_right

    def load_data_and_generate_ellipsoids(self, data_folder):
        """
        Load Demonstration Data and Generate Bimanual Manipulability Ellipsoids

        Parameters:
            data_folder (str): Path to the folder containing demonstration data
        """
        print('Loading demonstration data...')
        self.trajs = []
        R_14 = self.manipulability_params["R_14"]
        p_14 = self.manipulability_params["p_14"]
        W_rel = self.manipulability_params["W_rel"]

        # Load trajectory data
        for folder_name in os.listdir(data_folder):
            folder_path = os.path.join(data_folder, folder_name)

            if os.path.isdir(folder_path):
                print(f"Processing trajectory folder: {folder_name}")
                trajectory_points = []

                for file_name in sorted(os.listdir(folder_path)):
                    if file_name.endswith(".pkl"):
                        file_path = os.path.join(folder_path, file_name)

                        with open(file_path, "rb") as f:
                            point = pickle.load(f)

                        trajectory_points.append(point)

                self.trajs.append(trajectory_points)

        print(f"Total number of trajectories: {len(self.trajs)}")
        for i, trajectory in enumerate(self.trajs):
            print(f"Trajectory {i + 1}: {len(trajectory)} points")

        # Preprocess data
        self.nbSamples = len(self.trajs)
        self.nbData = len(self.trajs[0])
        self.xIn = np.arange(1, self.nbData + 1) * self.modelPD['dt']
        self.X = np.zeros((self.modelPD['nbVar'], self.modelPD['nbVar'], self.nbData * self.nbSamples))
        self.X[0, 0, :] = np.tile(self.xIn, self.nbSamples)

        Data = []

        for n in range(self.nbSamples):
            poses_left = []
            poses_right = []
            manipulability_matrices = []

            for t in range(self.nbData):
                data = self.trajs[n][t]
                q_left = data.get('joint_positions')[:6]
                q_right = data.get('joint_positions')[7:13]

                J_left = self.left_arm.jacob0(q_left)
                J_right = self.right_arm.jacob0(q_right)

                T_left = self.left_arm.fkine(q_left)
                poses_left.append(T_left)
                p_21 = T_left.t  # position vector from left arm ee to base
                R_21 = T_left.R  # rotation matrix from left arm ee to base
                R_24 = R_21 @ R_14  # rotation matrix from left arm ee to right arm ee

                T_right = self.right_arm.fkine(q_right)
                poses_right.append(T_right)
                p_43 = T_right.inv().t  # position vector from right arm base to right arm ee
                p_13 = p_14 - R_14 @ p_43
                p_23 = p_21 + R_21 @ p_13  # position vector from left arm ee to right arm ee

                JR = compute_bimanual_relative_jacobian(J_left, J_right, R_21, R_24, p_23, task_dim=3, scaling=True)
                manipulability_ellipsoid = compute_bimanual_relative_manipulability(JR, W_rel)
                manipulability_matrices.append(manipulability_ellipsoid)
                self.X[1:self.modelPD['nbVar'], 1:self.modelPD['nbVar'], t + n * self.nbData] = manipulability_ellipsoid

                xIn_t = np.hstack((self.xIn[t].reshape(1, 1), np.zeros((1, 3))))
                Data.append(np.vstack((xIn_t, T_left, T_right)))

            # visualize_trajectory_with_ellipsoids(poses_left, poses_right, manipulability_matrices, scale=0.05)

        Data = np.stack(Data, axis=2)

        # Combining data from all samples
        self.x = np.zeros((self.modelPD['nbVarVec'], self.nbData * self.nbSamples))
        self.x[0, :] = self.X[0, 0, :]
        for i in range(self.nbData * self.nbSamples):
            self.x[1:, i] = symmat2vec(self.X[1:, 1:, i])

        print("Demonstration data loaded and manipulability ellipsoids generated.")

    def gmm_learning(self, output_file):
        """
        GMM Learning and Save Model

        Parameters:
            output_file (str): Path to save the trained GMM model
        """
        print('Learning SPD GMM for Bimanual Manipulability Ellipsoids...')
        # Initialize GMM
        # Initialization on the manifold
        self.in_idx = 0
        self.outMat = np.arange(1, self.modelPD['nbVar'])
        self.out = np.arange(1, self.modelPD['nbVarVec'])
        self.modelPD = spd_init_GMM_kbins(self.x, self.modelPD, self.nbSamples, self.out)
        self.modelPD['Mu'] = np.zeros_like(self.modelPD['MuMan'])
        L = np.zeros((self.modelPD['nbStates'], self.nbData * self.nbSamples), dtype=np.float32)
        xts = np.zeros((self.modelPD['nbVarVec'], self.nbData * self.nbSamples, self.modelPD['nbStates']))

        # EM for SPD matrices manifold
        for nb in range(self.nbIterEM):
            print('.', end='')
            # E-step
            for i in range(self.modelPD['nbStates']):
                xts[self.in_idx, :, i] = self.x[self.in_idx, :] - self.modelPD['MuMan'][self.in_idx, i]
                xts[self.out, :, i] = logmap_vec(self.x[self.out, :], self.modelPD['MuMan'][self.out, i])
                L[i, :] = self.modelPD['Priors'][i] * gaussPDF(xts[:, :, i], self.modelPD['Mu'][:, i], self.modelPD['Sigma'][:, :, i])

            # Responsibilities
            L_sum = np.sum(L, axis=0, keepdims=True) + np.finfo(float).eps
            GAMMA = L / L_sum
            GAMMA_sum = np.sum(GAMMA, axis=1, keepdims=True) + np.finfo(float).eps
            H = GAMMA / GAMMA_sum

            # M-step
            for i in range(self.modelPD['nbStates']):
                # Update Priors
                self.modelPD['Priors'][i] = np.sum(GAMMA[i, :]) / (self.nbData * self.nbSamples)

                # Update MuMan
                for n in range(self.nbIter):
                    # Update on the tangent space
                    uTmp = np.zeros((self.modelPD['nbVarVec'], self.nbData * self.nbSamples))
                    uTmp[self.in_idx, :] = self.x[self.in_idx, :] - self.modelPD['MuMan'][self.in_idx, i]
                    uTmp[self.out, :] = logmap_vec(self.x[self.out, :], self.modelPD['MuMan'][self.out, i])
                    uTmpTot = np.sum(uTmp * H[i, :], axis=1)

                    # Update on the manifold
                    self.modelPD['MuMan'][self.in_idx, i] = uTmpTot[self.in_idx] + self.modelPD['MuMan'][self.in_idx, i]
                    self.modelPD['MuMan'][self.out, i] = expmap_vec(uTmpTot[self.out], self.modelPD['MuMan'][self.out, i])

                # Update Sigma
                self.modelPD['Sigma'][:, :, i] = (uTmp @ np.diag(H[i, :]) @ uTmp.T +
                                             np.eye(self.modelPD['nbVarVec']) * self.modelPD['params_diagRegFact'])

        print('SPD GMM Learning Completed.')

        # Save model to file
        with open(output_file, 'wb') as f:
            pickle.dump(self.modelPD, f)
        print(f"GMM model saved to {output_file}")

    def gmr_regression(self, model_file, xIn):
        """
        GMR Regression

        Parameters:
            model_file (str): Path to the trained GMM model
            xIn (np.ndarray): Time Input for regression

        Returns:
            xhat (np.ndarray): Predicted outputs
            expSigma (np.ndarray): Conditional covariance
        """
        print('Performing GMR Regression...')
        with open(model_file, 'rb') as f:
            self.modelPD = pickle.load(f)

        # Eigendecomposition of Sigma
        self.V = np.zeros((self.modelPD['nbVarVec'], self.modelPD['nbVarVec'], self.modelPD['nbStates']))
        self.D = np.zeros((self.modelPD['nbVarVec'], self.modelPD['nbVarVec'], self.modelPD['nbStates']))

        for i in range(self.modelPD['nbStates']):
            D_matrices, V_matrices = np.linalg.eig(self.modelPD['Sigma'][:, :, i])
            self.V[:, :, i] = V_matrices
            self.D[:, :, i] = np.diag(D_matrices)

        self.in_idx = 0  # time index
        out = np.arange(1, self.modelPD['nbVarVec'])  # Output dimensions
        self.outMat = np.arange(1, self.modelPD['nbVar'])
        nbVarOut = len(out)
        outMan = np.arange(1, self.modelPD['nbVar'])

        # Initializations for GMR for manipulability ellipsoids
        uhat = np.zeros(nbVarOut)
        xhat = np.zeros(nbVarOut)
        uOut = np.zeros((nbVarOut, self.modelPD['nbStates']))
        expSigma = np.zeros((nbVarOut, nbVarOut))
        H = np.zeros(self.modelPD['nbStates'])

        # GMR for manipulability ellipsoids
        for i in range(self.modelPD['nbStates']):
            H[i] = self.modelPD['Priors'][i] * gaussPDF(xIn - self.modelPD['MuMan'][self.in_idx, i],
                                                      self.modelPD['Mu'][self.in_idx, i],
                                                      self.modelPD['Sigma'][self.in_idx, self.in_idx, i]).item()

        id_max = np.argmax(H)
        xhat = self.modelPD['MuMan'][out, id_max]  # Initial point

        # Iterative computation
        for n in range(self.nbIter):
            uhat = np.zeros(nbVarOut)
            for i in range(self.modelPD['nbStates']):
                # Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t)
                S1 = vec2symmat(self.modelPD['MuMan'][out, i])
                S2 = vec2symmat(xhat)
                Ac = block_diag(1, transp_operator(S1, S2))

                # Parallel transport of eigenvectors
                vMat = np.zeros((self.modelPD['nbVar'], self.modelPD['nbVar'], self.V.shape[1], self.modelPD['nbStates']))
                pvMat = np.zeros_like(vMat)
                pV = np.zeros((self.modelPD['nbVarVec'], self.modelPD['nbVarVec'], self.modelPD['nbStates']))
                pSigma = np.zeros((self.modelPD['nbVarVec'], self.modelPD['nbVarVec'], self.modelPD['nbStates']))

                for j in range(self.V.shape[1]):
                    vMat[:, :, j, i] = block_diag(self.V[self.in_idx, j, i], vec2symmat(self.V[out, j, i]))

                    if np.isscalar(self.D[j, j, i]):
                        D_sqrt = np.sqrt(self.D[j, j, i])
                        pvMat[:, :, j, i] = Ac @ (D_sqrt * vMat[:, :, j, i]) @ Ac.T
                    else:
                        D_sqrt = np.sqrt(self.D[j, j, i])
                        pvMat[:, :, j, i] = Ac @ D_sqrt @ vMat[:, :, j, i] @ Ac.T

                    if np.isscalar(pvMat[self.in_idx, self.in_idx, j, i]):
                        input_diag = np.array([pvMat[self.in_idx, self.in_idx, j, i]])
                    else:
                        input_diag = np.diag(pvMat[self.in_idx, self.in_idx, j, i])
                    pV[:, j, i] = np.concatenate(
                        [input_diag, symmat2vec(pvMat[np.ix_(self.outMat, self.outMat, [j], [i])][:, :, 0, 0])])

                # Parallel transported sigma (reconstruction from eigenvectors)
                pSigma[:, :, i] = pV[:, :, i] @ pV[:, :, i].T

                # Gaussian conditioning on the tangent space
                if np.isscalar(pSigma[self.in_idx, self.in_idx, i]):
                    inv_pSigma_in = 1 / pSigma[self.in_idx, self.in_idx, i]
                    uOut[:, i] = (logmap_vec(self.modelPD['MuMan'][out, i], xhat).reshape(6, 1) + \
                                     (pSigma[out, self.in_idx, i] * inv_pSigma_in).reshape(6, 1) @ \
                                     (xIn - self.modelPD['MuMan'][self.in_idx, i]).reshape(1, 1)).flatten()
                else:
                    inv_pSigma_in = np.linalg.inv(pSigma[self.in_idx, self.in_idx, i])
                    uOut[:, i] = logmap_vec(self.modelPD['MuMan'][out, i], xhat) + \
                                    np.dot(pSigma[out, self.in_idx, i], inv_pSigma_in) @ \
                                    (xIn - self.modelPD['MuMan'][self.in_idx, i])

                # Accumulate weighted result
                uhat += uOut[:, i] * H[i]

            # Projection back onto the manifold
            xhat = expmap_vec(uhat, xhat)

        # Compute conditional covariances
        for i in range(self.modelPD['nbStates']):
            SigmaOutTmp = pSigma[out, out, i] - np.dot(pSigma[out, self.in_idx, i],
                                                       inv_pSigma_in) @ pSigma[
                              self.in_idx, out, i]
            expSigma[:, :] += H[i] * (SigmaOutTmp + np.outer(uOut[:, i], uOut[:, i]))

        expSigma[:, :] -= np.outer(uhat, uhat)

        print('GMR Regression Completed.')


        return xhat, expSigma

    def plot_results(self, xhat, expSigma):
        """
        Plot regression results.

        Parameters:
            xhat (np.ndarray): Predicted means
            expSigma (np.ndarray): Conditional covariance matrices
        """
        plt.figure()
        plt.plot(xhat.T, label='Predicted Means')
        plt.title("Regression Results")
        plt.legend()
        plt.show()

class GMRModel:
    def __init__(self, model_file, nbIter=10):

        self.nbIter = nbIter
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(model_file, 'rb') as f:
            self.modelPD = pickle.load(f)

        self.modelPD['MuMan'] = torch.tensor(
            self.modelPD['MuMan'], dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.modelPD['Sigma'] = torch.tensor(
            self.modelPD['Sigma'], dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.modelPD['Priors'] = torch.tensor(self.modelPD['Priors'], dtype=torch.float32, requires_grad=True, device=self.device)
        self.modelPD['Mu'] = torch.tensor(self.modelPD['Mu'], dtype=torch.float32, requires_grad=True, device=self.device)

    def gmr_regression(self, xIn):
        """
        GMR Regression using PyTorch operations

        Parameters:
            xIn (torch.Tensor): Time Input for regression (requires_grad=True)

        Returns:
            xhat (torch.Tensor): Predicted outputs
            expSigma (torch.Tensor): Conditional covariance
        """
        print('Performing GMR Regression...')

        # Eigendecomposition of Sigma
        self.V = torch.zeros(
            (self.modelPD['nbVarVec'], self.modelPD['nbVarVec'], self.modelPD['nbStates']),
            dtype=torch.float32,
            device=xIn.device
        )
        self.D = torch.zeros_like(self.V)

        for i in range(self.modelPD['nbStates']):
            D_matrices, V_matrices = torch.linalg.eig(self.modelPD['Sigma'][:, :, i])
            self.V[:, :, i] = V_matrices.real
            self.D[:, :, i] = torch.diag(D_matrices).real

        self.in_idx = 0  # time index
        out = torch.arange(1, self.modelPD['nbVarVec'], device=xIn.device)  # Output dimensions
        self.outMat = torch.arange(1, self.modelPD['nbVar'], device=xIn.device)
        nbVarOut = len(out)
        outMan = torch.arange(1, self.modelPD['nbVar'], device=xIn.device)

        # Initializations for GMR
        uhat = torch.zeros(nbVarOut, device=xIn.device)
        xhat = torch.zeros(nbVarOut, device=xIn.device)
        uOut = torch.zeros((nbVarOut, self.modelPD['nbStates']), device=xIn.device)
        expSigma = torch.zeros((nbVarOut, nbVarOut), device=xIn.device)
        H = torch.zeros(self.modelPD['nbStates'], device=xIn.device)

        # GMR for manipulability ellipsoids
        for i in range(self.modelPD['nbStates']):
            H[i] = self.modelPD['Priors'][i] * self.gaussPDF(
                xIn - self.modelPD['MuMan'][self.in_idx, i],
                self.modelPD['Mu'][self.in_idx, i],
                self.modelPD['Sigma'][self.in_idx, self.in_idx, i]
            )

        id_max = torch.argmax(H)
        xhat = self.modelPD['MuMan'][out, id_max]

        # Iterative computation
        for n in range(self.nbIter):
            uhat = torch.zeros(nbVarOut, device=xIn.device)
            for i in range(self.modelPD['nbStates']):
                # Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t)
                S1 = self.vec2symmat(self.modelPD['MuMan'][out, i])
                S2 = self.vec2symmat(xhat)
                Ac = torch.block_diag(torch.tensor(1.0, device=xIn.device), self.transp_operator(S1, S2))

                # Parallel transport of eigenvectors
                vMat = torch.zeros(
                    (self.modelPD['nbVar'], self.modelPD['nbVar'], self.V.shape[1], self.modelPD['nbStates']), dtype=torch.float32, device=xIn.device
                )
                pvMat = torch.zeros_like(vMat)
                pV = torch.zeros(
                    (self.modelPD['nbVarVec'], self.modelPD['nbVarVec'], self.modelPD['nbStates']),
                    device=xIn.device
                )
                pSigma = torch.zeros_like(pV)

                for j in range(self.V.shape[1]):
                    vMat[:, :, j, i] = torch.block_diag(
                        self.V[self.in_idx, j, i], self.vec2symmat(self.V[out, j, i])
                    )

                    D_sqrt = torch.sqrt(self.D[j, j, i])
                    pvMat[:, :, j, i] = Ac @ (D_sqrt * vMat[:, :, j, i]) @ Ac.T
                    input_diag = torch.diag(pvMat[self.in_idx, self.in_idx, j, i].unsqueeze(0).unsqueeze(1))
                    pV[:, j, i] = torch.cat(
                        [input_diag, self.symmat2vec(pvMat.index_select(0, self.outMat).index_select(1, self.outMat)[:, :, j, i])]
                    )

                # Parallel transported sigma
                pSigma[:, :, i] = pV[:, :, i] @ pV[:, :, i].T


                if len(pSigma[self.in_idx, self.in_idx, i].shape) == 0:
                    inv_pSigma_in = 1 / pSigma[self.in_idx, self.in_idx, i]
                    uOut[:, i] = (self.logmap_vec(self.modelPD['MuMan'][out, i],
                        xhat
                    ).view(6, 1) + (pSigma[out, self.in_idx, i] * inv_pSigma_in).view(6, 1) @ (
                        xIn - self.modelPD['MuMan'][self.in_idx, i]
                    ).view(1, 1)).flatten()
                else:
                    inv_pSigma_in = torch.linalg.inv(pSigma[self.in_idx, self.in_idx, i])
                    uOut[:, i] = self.logmap_vec(
                        self.modelPD['MuMan'][out, i], xhat
                    ) + (pSigma.index_select(0, torch.tensor(out)).index_select(1, torch.tensor(self.in_idx))[:, :,
                         i] * inv_pSigma_in) @ (xIn - self.modelPD['MuMan'][self.in_idx, i])

                # Accumulate weighted result
                uhat += uOut[:, i] * H[i]

            # Projection back onto the manifold
            xhat = self.expmap_vec(uhat, xhat)

        # Compute conditional covariances
        for i in range(self.modelPD['nbStates']):
            SigmaOutTmp = pSigma[out, out, i] - torch.matmul( pSigma[out, self.in_idx, i] * inv_pSigma_in.unsqueeze(0),
                pSigma[self.in_idx, out, i]
            )
            expSigma[:, :] += H[i] * (SigmaOutTmp + torch.outer(uOut[:, i], uOut[:, i]))

        expSigma[:, :] -= torch.outer(uhat, uhat)

        print('GMR Regression Completed.')
        return xhat, expSigma

    def gaussPDF(self, Data, Mu, Sigma):
        """
        Compute the likelihood of data points under a Gaussian distribution using PyTorch,
        with support for automatic differentiation.

        Parameters:
        - Data:  torch.Tensor of shape (D, N), representing N datapoints of D dimensions.
        - Mu:    torch.Tensor of shape (D, 1), representing the mean of the Gaussian.
        - Sigma: torch.Tensor of shape (D, D), representing the covariance matrix of the Gaussian.

        Returns:
        - prob:  torch.Tensor of shape (N,), representing the likelihood of the N datapoints.
        """

        # Ensure inputs are tensors and move them to the same device
        Data = torch.as_tensor(Data)
        Mu = torch.as_tensor(Mu)
        Sigma = torch.as_tensor(Sigma)

        # Handle scalar cases
        if Data.ndim == 0:
            Data = Data.view(1, 1)  # Convert scalar to (1, 1)
        if Mu.ndim == 0:
            Mu = Mu.view(1, 1)  # Convert scalar to (1, 1)
        if Sigma.ndim == 0:
            Sigma = Sigma.view(1, 1)  # Convert scalar to (1, 1)

        # Check if Data is 1D and reshape it to a 2D array for consistent matrix operations
        if Data.ndim == 1:
            Data = Data.view(-1, 1)

        nbVar, nbData = Data.shape

        # Center the data by subtracting the mean Mu (broadcasting works in PyTorch)
        Data = Data.T - Mu.T  # Shape: (N, D)

        # Compute the inverse and determinant of the covariance matrix
        try:
            Sigma_inv = torch.linalg.inv(Sigma)
            Sigma_det = torch.linalg.det(Sigma)
        except RuntimeError:
            print(f"Warning: Singular matrix encountered. Using pseudo-inverse for covariance matrix Sigma.")
            Sigma_inv = torch.linalg.pinv(Sigma)  # Use pseudo-inverse
            Sigma_det = torch.linalg.det(Sigma + torch.eye(Sigma.size(0)) * 1e-6)  # Regularized determinant

        # Compute the Mahalanobis distance
        # Data @ Sigma_inv @ Data.T computes the quadratic form
        prob = torch.sum((Data @ Sigma_inv) * Data, dim=1)  # Shape: (N,)

        # Compute the Gaussian probability density function
        prob = torch.exp(-0.5 * prob) / (
            torch.sqrt((2 * torch.pi) ** nbVar * torch.abs(Sigma_det) + torch.finfo(torch.float32).eps))

        return prob

    def vec2symmat(self, v):
        """
        This function computes SPD matrices based on a vector using Mandel notation.
        Supports PyTorch operations for automatic differentiation.

        Parameters:
            v: torch.Tensor
               Vectorized SPD matrix (d',) or vectorized SPD matrices (d', N).

        Returns:
            M: torch.Tensor
               SPD matrix (d x d) or SPD matrices (d x d x N).
        """
        if v.ndim == 1:  # Case for a single vectorized SPD matrix
            n = v.shape[0]
            N = int((-1 + torch.sqrt(torch.tensor(1 + 8 * n, dtype=torch.float32))) // 2)
            M = torch.diag(v[:N].to(torch.float32)) # Initialize diagonal elements of the SPD matrix
            id = torch.cumsum(torch.flip(torch.arange(1, N + 1), dims=[0]), dim=0)

            for i in range(N - 1):
                M += torch.diag(v[id[i]:id[i + 1]] / torch.sqrt(torch.tensor(2.0)), diagonal=i + 1)  # Upper diagonal
                M += torch.diag(v[id[i]:id[i + 1]] / torch.sqrt(torch.tensor(2.0)), diagonal=-(i + 1))  # Lower diagonal

        else:  # Case for multiple vectorized SPD matrices
            d, N = v.shape
            D = int((-1 + torch.sqrt(torch.tensor(1 + 8 * d, dtype=torch.float32))) // 2)
            M = torch.zeros((D, D, N), dtype=torch.float32, device=v.device)  # Initialize SPD matrices

            for n in range(N):
                vn = v[:, n]
                Mn = torch.diag(vn[:D])  # Initialize diagonal elements of the SPD matrix
                id = torch.cumsum(torch.flip(torch.arange(1, D + 1, device=v.device), dims=[0]), dim=0)

                for i in range(D - 1):
                    Mn += torch.diag(
                        vn[id[i]:id[i + 1]] / torch.sqrt(torch.tensor(2.0, dtype=v.dtype, device=v.device)),
                        diagonal=i + 1)  # Upper diagonal
                    Mn += torch.diag(
                        vn[id[i]:id[i + 1]] / torch.sqrt(torch.tensor(2.0, dtype=v.dtype, device=v.device)),
                        diagonal=-(i + 1))  # Lower diagonal

                M[:, :, n] = Mn

        return M

    def symmat2vec(self, M):
        """
        This function computes a vectorization of SPD matrices using Mandel
        notation with PyTorch, supporting automatic differentiation.

        Parameters:
        - M: torch.Tensor
             SPD matrix or SPD matrices of size (d, d) or (d, d, N).

        Returns:
        - v: torch.Tensor
             Vectorized SPD matrix or vectorized SPD matrices of size (d',) or (d', N).
        """
        if M.ndim == 2:  # Case for a single SPD matrix
            D = M.shape[0]
            v = torch.diag(M)  # Extract diagonal elements

            # Extract upper diagonal elements (off-diagonal) and scale using Mandel notation
            for d in range(1, D):
                v = torch.cat((v, torch.sqrt(torch.tensor(2.0)) * torch.diag(M, d)))

        elif M.ndim == 3:  # Case for multiple SPD matrices
            D, _, N = M.shape
            v_list = []  # List to store vectorized matrices

            for n in range(N):  # Loop through each matrix in the batch
                Mn = M[:, :, n]
                vn = torch.diag(Mn)  # Diagonal elements of the nth matrix

                # Extract upper diagonal elements (off-diagonal) and scale using Mandel notation
                for d in range(1, D):
                    vn = torch.cat((vn, torch.sqrt(torch.tensor(2.0)) * torch.diag(Mn, d)))

                v_list.append(vn.unsqueeze(1))  # Add dimension for stacking

            v = torch.cat(v_list, dim=1)  # Stack all vectorized matrices along the last dimension

        else:
            # Handle case where M is neither 2D nor 3D
            raise ValueError(f"Input matrix M must be 2D or 3D, but got {M.ndim}D")

        return v

    import torch

    def logmap(self, X, S):
        """
        Compute the logarithmic map on the SPD manifold using PyTorch,
        with support for automatic differentiation.

        Parameters:
            X: torch.Tensor
               SPD matrix (d x d) or SPD matrices (d x d x N).
            S: torch.Tensor
               Base SPD matrix (d x d).

        Returns:
            U: torch.Tensor
               Symmetric matrix Log_S(X) (d x d) or symmetric matrices (d x d x N).
        """
        # Determine if X represents a single matrix or a batch of matrices
        if X.ndim == 3:  # Case: Batch of SPD matrices
            N = X.shape[2]  # Number of matrices
            D = X.shape[0]  # Dimension of each matrix
            U = torch.zeros((D, D, N), dtype=X.dtype, device=X.device)  # Initialize output tensor

            # Loop over each matrix in the batch
            for n in range(N):
                # Compute S^-1 * X[:, :, n]
                S_inv_X = torch.linalg.solve(S, X[:, :, n])  # Equivalent to S^-1 @ X

                # Perform eigendecomposition of S^-1 * X[:, :, n]
                eigvals, eigvecs = torch.linalg.eig(S_inv_X)

                # Ensure numerical stability by clamping eigenvalues to avoid log(0)
                eps = 1e-8
                eigvals = torch.clamp(eigvals, min=eps)

                # Take the logarithm of the eigenvalues
                log_eigvals = torch.diag(torch.log(eigvals))

                # Reconstruct the log map using the eigendecomposition
                U[:, :, n] = S @ eigvecs @ log_eigvals @ torch.linalg.inv(eigvecs)

        else:  # Case: Single SPD matrix
            # Compute S^-1 * X
            S_inv_X = torch.linalg.solve(S, X)  # Equivalent to S^-1 @ X

            # Perform eigendecomposition of S^-1 * X
            eigvals, eigvecs = torch.linalg.eig(S_inv_X)
            eigvals = torch.real(eigvals)
            eigvecs = torch.real(eigvecs)

            # Ensure numerical stability by clamping eigenvalues to avoid log(0)
            eps = 1e-8
            eigvals = torch.clamp(eigvals, min=eps)

            # Take the logarithm of the eigenvalues
            log_eigvals = torch.diag(torch.log(eigvals))

            # Reconstruct the log map using the eigendecomposition
            U = S @ eigvecs @ log_eigvals @ torch.linalg.inv(eigvecs)

        return U

    def logmap_vec(self, x, s):
        """
        This function computes the logarithmic map on the SPD manifold with Mandel notation,
        using PyTorch to support automatic differentiation.

        Parameters:
            x: torch.Tensor
               SPD matrix in vector form (d',) or SPD matrices in vector form (d', N).
            s: torch.Tensor
               Base SPD matrix in vector form (d',).

        Returns:
            u: torch.Tensor
               Logarithmic map result in vector form (d',) or (d', N).
        """
        # Convert vectorized SPD matrices back to symmetric matrices
        X = self.vec2symmat(x)  # Convert x from vector to symmetric matrix form
        S = self.vec2symmat(s)  # Convert s from vector to symmetric matrix form

        # Compute the logarithmic map Log_S(X) on the SPD manifold
        U = self.logmap(X, S)  # Compute the logmap in matrix form

        # Convert the resulting symmetric matrices back to vector form
        u = self.symmat2vec(U)  # Convert U from matrix form back to vector form

        return u

    import torch

    def expmap(self, U, S):
        """
        Computes the exponential map on the SPD manifold using PyTorch,
        with support for automatic differentiation.

        Parameters:
        - U: torch.Tensor
             Symmetric matrix on the tangent space of S, shape (d, d) or (d, d, N).
        - S: torch.Tensor
             Base SPD matrix, shape (d, d).

        Returns:
        - X: torch.Tensor
             SPD matrix Exp_S(U), shape (d, d) or (d, d, N).
        """
        if U.ndim == 3:  # Batch of tangent space matrices
            # Get the number of matrices
            N = U.shape[2]
            D = U.shape[0]

            # Initialize the result matrix X
            X = torch.zeros((D, D, N), dtype=U.dtype, device=U.device)

            # Loop over each U[:, :, n]
            for n in range(N):
                # Compute S^-1 * U[:, :, n]
                Sinv_U = torch.linalg.solve(S, U[:, :, n])  # Equivalent to S^-1 @ U

                # Perform eigenvalue decomposition
                eigvals, eigvecs = torch.linalg.eig(Sinv_U)

                # Exponentiate the eigenvalues
                exp_eigvals = torch.diag(torch.real(torch.exp(eigvals)))

                # Reconstruct the SPD matrix using the exponential map
                X[:, :, n] = S @ eigvecs @ exp_eigvals @ torch.linalg.inv(eigvecs)

        else:  # Single tangent space matrix
            # Compute S^-1 * U
            Sinv_U = torch.linalg.solve(S, U)  # Equivalent to S^-1 @ U

            # Perform eigenvalue decomposition
            eigvals, eigvecs = torch.linalg.eig(Sinv_U)
            eigvals = torch.real(eigvals)
            eigvecs = torch.real(eigvecs)

            # Exponentiate the eigenvalues
            exp_eigvals = torch.diag(torch.real(torch.exp(eigvals)))

            # Reconstruct the SPD matrix using the exponential map
            X = S @ eigvecs @ exp_eigvals @ torch.linalg.inv(eigvecs)

        return X

    import torch

    def expmap_vec(self, u, s):
        """
        Compute the exponential map on the SPD manifold with Mandel notation, using PyTorch
        to support automatic differentiation.

        Parameters:
        - u: torch.Tensor
             Symmetric matrix in vector form (Mandel notation), shape (d',) or (d', N).
        - s: torch.Tensor
             Base SPD matrix in vector form (Mandel notation), shape (d',).

        Returns:
        - x: torch.Tensor
             SPD matrix Exp_S(U) in vector form (Mandel notation), shape (d',) or (d', N).
        """
        # Step 1: Convert vectorized SPD matrices to symmetric matrix form
        U = self.vec2symmat(u)  # Convert tangent vector from Mandel vector form to symmetric matrix form
        S = self.vec2symmat(s)  # Convert base matrix from Mandel vector form to symmetric matrix form

        # Step 2: Compute the exponential map on the SPD manifold
        X = self.expmap(U, S)  # Perform the exponential map in matrix form

        # Step 3: Convert the resulting symmetric matrix back to vector form
        x = self.symmat2vec(X)  # Convert the result back to Mandel vector form

        return x

    def transp_operator(self, S1, S2):
        """
        Compute the parallel transport operator from S1 to S2 on the SPD manifold using PyTorch,
        with support for automatic differentiation.

        A SPD matrix X is transported from S1 to S2 with Ac * X * Ac'.

        Parameters:
        - S1: torch.Tensor
             SPD matrix of shape (d, d).
        - S2: torch.Tensor
             SPD matrix of shape (d, d).

        Returns:
        - Ac: torch.Tensor
             Parallel transport operator of shape (d, d).
        """
        # Ensure S1 and S2 are square matrices
        if S1.shape[0] != S1.shape[1] or S2.shape[0] != S2.shape[1]:
            raise ValueError("S1 and S2 must be square matrices.")

        # Step 1: Compute the inverse of S1 (S1^-1)
        S1_inv = torch.linalg.inv(S1)  # PyTorch equivalent of NumPy's np.linalg.inv

        # Step 2: Compute the product S2 * S1^-1
        prod = S2 @ S1_inv

        Ac = sqrtm(prod)

        return Ac

class BimanualRobot:
    def __init__(self, urdf_path):
        """
        Initialize the bimanual robot with left and right arms based on URDF.
        """
        with open(urdf_path, "rb") as f:  # 以字节模式读取文件
            urdf_data = f.read()

        # Extract links and joints for both arms
        # Assuming the URDF contains information for both arms
        self.left_arm_chain = pk.build_serial_chain_from_urdf(urdf_data, "Link6")
        self.right_arm_chain = pk.build_serial_chain_from_urdf(urdf_data, "Link6")

        # Set the device and data type for the kinematic chains
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        self.left_arm_chain.to(dtype=dtype, device=self.device)
        self.right_arm_chain.to(dtype=dtype, device=self.device)

        # Relative parameters for bimanual manipulability calculation
        self.R_21 = torch.eye(3).to(self.device)  # Left arm rotation matrix: end-effector (ee) to base
        self.R_24 = torch.eye(3).to(self.device)  # Relative rotation matrix: left arm ee to right arm base
        self.R_34 = torch.eye(3).to(self.device)   # Right arm rotation matrix: ee to base

        self.p_21 = torch.tensor([0.0, 0.0, 0.0]).to(self.device)   # Absolute position vector: left arm ee to base
        self.p_23 = torch.tensor([0.0, 0.0, 0.0]).to(self.device)   # Relative position vector: left arm ee to right arm ee

        # Relative transformation matrix: left arm ee to right arm ee
        self.T_14 = torch.eye(4).to(self.device)   # Start with an identity matrix
        self.R_14 = torch.tensor([[-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, 1.0]]).to(self.device)   # Rotation part
        self.p_14 = torch.tensor([1.08, 0.0, 0.0]).to(self.device)   # Translation part

        # Update T_14 with rotation and position
        self.T_14[:3, :3] = self.R_14
        self.T_14[:3, 3] = self.p_14

        # Relative weight matrix
        self.W_rel = torch.eye(12).to(self.device)   # Relative weight matrix (12x12)

        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def compute_bimanual_jacobian(self, q_left, q_right):
        """
        Compute the Jacobians for left and right arms.

        Parameters:
            q_left (torch.Tensor): Joint angles for the left arm.
            q_right (torch.Tensor): Joint angles for the right arm.

        Returns:
            torch.Tensor, torch.Tensor: Jacobians for left and right arms.
        """
        J_left = self.left_arm_chain.jacobian(q_left)
        J_right = self.right_arm_chain.jacobian(q_right)

        if not isinstance(J_left, torch.Tensor):
            J_left = torch.tensor(J_left, dtype=torch.float32)  # 转为 torch.Tensor
        if not isinstance(J_right, torch.Tensor):
            J_right = torch.tensor(J_right, dtype=torch.float32)  # 转为 torch.Tensor

        return J_left, J_right

    def skew_symmetric_matrix(self, vector):
        """
        Compute the skew-symmetric matrix of a 3D vector using PyTorch.

        Parameters:
            vector (torch.Tensor): A 3D vector (shape: [3]).

        Returns:
            torch.Tensor: The 3x3 skew-symmetric matrix.
        """
        x, y, z = vector
        return torch.tensor([[0, -z, y],
                             [z, 0, -x],
                             [-y, x, 0]], dtype=vector.dtype, device=vector.device)

    def compute_bimanual_relative_jacobian(self, q_left, q_right, J_left, J_right, task_dim=3, scaling=False):
        """
        Compute the relative Jacobian matrix JR using PyTorch.

        Parameters:
            q_left (torch.Tensor): Joint angles for the left arm.
            q_right (torch.Tensor): Joint angles for the right arm.
            J_left (torch.Tensor): Jacobian matrix for the left arm (6xN).
            J_right (torch.Tensor): Jacobian matrix for the right arm (6xN).
            task_dim (int): Desired task dimension of the Jacobian.
            scaling (bool): Whether to scale the Jacobian matrix.

        Returns:
            torch.Tensor: The relative Jacobian matrix JR.
        """
        # Compute the left arm end-effector transformation
        T_left = self.left_arm_chain.forward_kinematics(q_left, end_only=False)
        left_tg = T_left['Link6']  # Left arm end-effector transformation
        self.p_21 = left_tg.get_matrix()[0, :3, 3] # Update left arm end-effector position
        self.R_21 = left_tg.get_matrix()[0, :3, :3].to(self.device)# Update left arm end-effector rotation
        self.R_24 = self.R_21 @ self.R_14  # Update relative rotation matrix
        self.R_24 = self.R_24.to(self.device)

        T_right = self.right_arm_chain.forward_kinematics(q_right, end_only=False)
        right_tg = T_right['Link6']  # Right arm end-effector transformation
        # get the inverse of the right arm end-effector transformation and apply transpose to get the rotation matrix
        self.p_43 = torch.linalg.inv(right_tg.get_matrix())[0, :3, 3]
        self.p_13 = self.p_14 -self.R_14 @ self.p_43
        self.p_23 = self.p_21 + self.R_21 @ self.p_13

        # Calculate the wrench transformation matrix
        S = self.skew_symmetric_matrix(self.p_23)
        I = torch.eye(3, dtype=J_left.dtype, device=self.device)
        psi =  torch.cat([torch.cat([I, -S], dim=1), torch.cat([torch.zeros((3, 3)), I], dim=1)], dim=0)  # 6x6 block diagonal matrix
        # Calculate rotation matrices in 6x6 form
        omega_21 = torch.cat([
    torch.cat([self.R_21, torch.zeros((3, 3), dtype=self.R_21.dtype, device=self.device)], dim=1),
    torch.cat([torch.zeros((3, 3), dtype=self.R_21.dtype, device=self.device), self.R_21], dim=1)
], dim=0)
        omega_24 = torch.cat([
    torch.cat([self.R_24, torch.zeros((3, 3), dtype=self.R_24.dtype, device=self.device)], dim=1),
    torch.cat([torch.zeros((3, 3), dtype=self.R_24.dtype, device=self.device), self.R_24], dim=1)
], dim=0)

        # Calculate the relative Jacobian matrix
        JR = torch.cat([-psi @ omega_21 @ J_left, omega_24 @ J_right], dim=2)[0][:task_dim, :]

        if scaling:
            # Scale the Jacobian matrix
            auxJ = torch.tensor([[6.6028, 2.1570, 1.2760],
                                 [7.4681, 9.7562, 4.8345],
                                 [0, 0, 0]], dtype=JR.dtype, device=JR.device)

            # Compute the scaling factor using the Frobenius norm
            scale = torch.norm(auxJ, p='fro') / torch.norm(JR, p='fro')

            # Apply scaling to JR
            JR = scale * JR

        return JR

    def compute_bimanual_relative_manipulability(self, JR):
        """
        Compute the relative manipulability ellipsoid of a dual-arm robot system given its relative Jacobian and weight matrix.

        Parameters:
            JR (torch.Tensor): The relative Jacobian matrix.
            W_rel (torch.Tensor): The weight matrix (usually 6x6).

        Returns:
            torch.Tensor: The relative manipulability ellipsoid matrix M.
        """
        # Compute the manipulability ellipsoid using the direct formulation
        M = JR @ self.W_rel @ JR.T

        return M


if __name__ == "__main__":
    # BimanualRobot Testing
    # urdf_path = "/home/zhuoli/dobot_xtrainer/ModelTrain/manipulability/data/urdf/nova2_robot.urdf"
    # robot = BimanualRobot(urdf_path)
    # # 2. 测试 compute_bimanual_jacobian
    # # 生左右臂固定值关节角
    # q_left = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    # q_right = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    #
    # print("\n--- Testing compute_bimanual_jacobian ---")
    # J_left, J_right = robot.compute_bimanual_jacobian(q_left, q_right)
    # print("J_left:\n", J_left)
    # print("J_right:\n", J_right)
    #
    # print("\n--- Testing compute_bimanual_relative_jacobian ---")
    # JR = robot.compute_bimanual_relative_jacobian(q_left, q_right, J_left, J_right, task_dim=3, scaling=False)
    # print("Relative Jacobian JR:\n", JR)
    #
    #
    # print("\n--- Testing compute_bimanual_relative_manipulability ---")
    # M = robot.compute_bimanual_relative_manipulability(JR)
    # print("Relative Manipulability Matrix M:\n", M)

    # # Example: Compute a gradient (e.g., minimize end-effector position error)
    # loss = torch.sum(J_left ** 2) + torch.sum(J_right ** 2)  # Example loss
    # loss.backward()  # Compute gradients with respect to q_left and q_right
    # print("\nGradients (q_left):", q_left.grad)
    # print("\nGradients (q_right):", q_right.grad)

    # GMRModel Testing
    # GMR initialization
    model_file = "/home/zhuoli/dobot_xtrainer/ModelTrain/manipulability/ckpt/gmm_ckpt.pth"
    gmr_model = GMRModel(model_file)

    # GMR regression example
    xIn = torch.tensor([0.1], requires_grad=True)
    xhat_pred, sigma_pred = gmr_model.gmr_regression(xIn)
    print("Predicted Means:")
    print(xhat_pred)
    print("\nConditional Covariance:")
    print(sigma_pred)



