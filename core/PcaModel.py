# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:27:13 2018

@author: For_Gondor
"""
import numpy as np


def rescale_pca_basis(orthonormal_basis, eigenvalues):
    """
    Takes an orthonormal PCA basis matrix (a matrix consisting
    of the eigenvectors) and rescales it, i.e. multiplies each
    eigenvector by the square root of its corresponding
    eigenvalue.
    
    Args:
        orthonormal_basis: An orthonormal_basis PCA basis matrix.
        eigenvalues: A row or column vector of eigenvalues.
        
    Returns:
        The rescaled PCA basis matrix.
    """
    rescaled_basis = np.empty(np.shape(orthonormal_basis))
    for i in range(np.shape(orthonormal_basis)[1]):
        rescaled_basis[:, i] = orthonormal_basis[:, i] * np.sqrt(eigenvalues[i])
    return rescaled_basis


def normalise_pca_basis(rescaled_basis, eigenvalues):
    """
    Takes a rescaled PCA basis matrix and scales it back to
    an orthonormal basis matrix by multiplying each eigenvector
    by 1 over the square root of its corresponding eigenvalue.
    
    Args:
        rescaled_basis: A rescaled PCA basis matrix.
        eigenvalues: A row or column vector of eigenvalues.
        
    Returns:
        The orthonormal PCA basis matrix.
    """
    orthonormal_basis = np.empty(np.shape(rescaled_basis))
    for i in range(np.shape(orthonormal_basis)[1]):
        orthonormal_basis[:, i] = rescaled_basis[:, i] / np.sqrt(eigenvalues[i])
    return orthonormal_basis


class PcaModel:
    """
    Construct a PCA model from given mean, normalised PCA basis, eigenvalues
    and triangle list.
    
    See the documentation of the member variables for how the data should
    be arranged.
    
    Attributes:
        mean:
            The mean used to build the PCA model.
            A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices.
        orthonormal_pca_basis:
            An orthonormal PCA basis (eigenvectors).
            m x n (rows x cols) = numShapeDims x numShapePcaCoeffs. Each column is an eigenvector.
        eigenvalues:
            The eigenvalues used to build the PCA model.
            A col-vector of the eigenvalues (variances in the PCA space).
        triangle_list:
            An index list of how to assemble the mesh.
            List of triangles that make up the mesh of the model.
        rescaled_pca_basis:
            m x n (rows x cols) = numShapeDims x numShapePcaCoeffs. Each column is an eigenvector.
    '"""
    def __init__(self, mean, orthonormal_pca_basis, eigenvalues, triangle_list):
        """
        Initialize PcaModel
        """
        self.mean = mean
        self.orthonormal_pca_basis = orthonormal_pca_basis
        self.eigenvalues = eigenvalues
        self.triangle_list = triangle_list
        self.rescaled_pca_basis = rescale_pca_basis(orthonormal_pca_basis, eigenvalues)
    
    def get_num_principal_components(self):
        """
        Returns the number of principal components in the model.
            
        Returns:
            The number of principal components in the model.
        """
        return np.shape(self.rescaled_pca_basis)[1]
    
    def get_rescaled_pca_basis_at_point(self, vertex_id):
        """
        Returns the PCA basis for a particular vertex, from the rescaled basis.
        
        Note: The function does not return the i-th basis vector - it returns all basis
        vectors, but only the block that is relevant for the vertex \p vertex_id.
        
        Args:
            vertex_id: A vertex index. Make sure it is valid.
        
        Returns:
            A 3 x num_principal_components matrix of the relevant rows of the original basis.
        """
        vertex_id *= 3
        assert vertex_id < np.shape(self.rescaled_pca_basis)[0]
        return self.rescaled_pca_basis[vertex_id:vertex_id + 3, :]
    
    def get_mean_at_point(self, vertex_index):
        """
        Return the value of the mean at a given vertex index.
        
        Args:
            vertex_index: A vertex index.
            
        Returns:
            A 3-dimensional vector containing the values at the given vertex index.
        """
        vertex_index *= 3
        return self.mean[vertex_index: vertex_index + 3]
    
    def draw_sample(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Returns a sample from the model with the given PCA coefficients.
        The given coefficients should follow a standard normal distribution, i.e.
        not be "normalised" with their eigenvalues/variances.
        
        Args:
            coefficients:
                The PCA coefficients used to generate the sample, ndarray type.
        
        Returns:
            A model instance with given coefficients.
        """
        # Fill the rest with zeros if not all coefficients are given:
        if len(coefficients) < self.get_num_principal_components():
            alphas = np.hstack([coefficients, np.zeros([self.get_num_principal_components() - len(coefficients)])])
        else:
            alphas = coefficients
        
        model_sample = self.mean + self.rescaled_pca_basis.dot(alphas)
        
        return model_sample
