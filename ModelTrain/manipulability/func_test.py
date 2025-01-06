import torch
import numpy as np
import random
from ModelTrain.manipulability.bimanual_manip_learning import GMRModel
from ModelTrain.manipulability.manip_utils import is_spd


if __name__ == '__main__':
    # GMR initialization
    model_file = "/home/zhuoli/dobot_xtrainer/ModelTrain/manipulability/ckpt/gmm_ckpt.pth"
    gmr_model = GMRModel(model_file)

    # # SPD parallel transport example
    # # Define fixed SPD matrices S1 and S2
    # S1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    # S2 = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    #
    # print("S1 (SPD matrix):")
    # print(S1)
    # print("\nS2 (SPD matrix):")
    # print(S2)
    #
    # # Compute the parallel transport operator
    # Ac = gmr_model.transp_operator(S1, S2)
    #
    # print("\nParallel transport operator (Ac):")
    # print(Ac)
    #
    # # Validate the result:
    # # Ac * S1 * Ac' should equal S2 (approximately)
    # transported_S1 = Ac @ S1 @ Ac.T
    # print("\nTransported S1 (should approximate S2):")
    # print(transported_S1)
    #
    # # Compute the difference between S2 and transported_S1
    # difference = torch.norm(transported_S1 - S2)
    # print("\nDifference (||Transported S1 - S2||):")
    # print(difference)
    #
    # # Assert that the difference is small
    # assert difference < 1e-5, "Transported S1 does not approximate S2 well!"


    # # tensor-based logmap example
    # # Define a 2x2 SPD matrix X
    # X = torch.tensor([[4.0, 1.0], [1.0, 3.0]])  # Single SPD matrix
    #
    # # Define a base SPD matrix S (identity matrix)
    # S = torch.eye(2)  # Identity matrix as the base point
    #
    # # Test logmap with a single SPD matrix
    # U_single = gmr_model.logmap(X, S)
    # print("Logarithmic map for a single SPD matrix:")
    # print(U_single)
    #
    # # Define multiple (2) SPD matrices X
    # X_multi = torch.stack([X, 2 * X], dim=-1)  # Stack two SPD matrices along the third dimension
    # print("\nMultiple SPD matrices X:\n", X_multi[:, :, 1])
    #
    # # Test logmap with multiple SPD matrices
    # U_multi = gmr_model.logmap(X_multi, S)
    # print("\nLogarithmic map for multiple SPD matrices:")
    # print(U_multi[:, :, 1])  # Print the logarithmic map for the second matrix in the batch

    # tensor-based symmat2vec & vec2symmat example
    # Define a 2x2 SPD matrix M
    M_original = torch.tensor([[4.0, 1.0], [1.0, 3.0]])  # A simple 2x2 SPD matrix
    print("Original matrix M:\n", M_original)

    # Test symmat2vec
    vec_M = gmr_model.symmat2vec(M_original)
    print("\nVectorized M:\n", vec_M)

    # Test vec2symmat
    M_reconstructed = gmr_model.vec2symmat(vec_M)
    print("\nReconstructed matrix M:\n", M_reconstructed)

    # Check if reconstructed matrix is SPD
    if is_spd(M_reconstructed):
        print("Reconstructed matrix is symmetric positive definite (SPD).")
    else:
        print("Reconstructed matrix is not symmetric positive definite (SPD).")

    # Test multiple symmetric matrices
    v1 = torch.tensor([4.0, 3.0, 2.0, torch.sqrt(torch.tensor(2.0)) * 1, torch.sqrt(torch.tensor(2.0)) * 1, torch.sqrt(torch.tensor(2.0)) * 1])
    v2 = torch.tensor([5.0, 6.0, 7.0, torch.sqrt(torch.tensor(2.0)) * 2, torch.sqrt(torch.tensor(2.0)) * 2, torch.sqrt(torch.tensor(2.0)) * 2])
    V = torch.column_stack((v1, v2))  # Combine two vectorized matrices into a matrix
    print("\nVectorized matrices V:\n", V)

    # Reconstruct multiple symmetric matrices
    M_reconstructed_multi = gmr_model.vec2symmat(V.T)  # Transpose to match (N, d*(d+1)//2)
    print("\nReconstructed Matrices (for multiple inputs):")
    print(M_reconstructed_multi)
