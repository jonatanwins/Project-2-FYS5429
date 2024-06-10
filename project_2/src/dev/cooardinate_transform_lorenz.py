"""
func for doing cooardinate transform as Kathleen does in her paper. Could be useless for our results, keeping it here for now.
"""
import numpy as np

def coordinate_transformation(xi, test_data):
    a1 = 1
    a2 = test_data['sindy_coefficients'][2, 0] / xi[2, 0]
    a3 = np.sqrt(-xi[5, 2] / xi[6, 1] * a2 ** 2)
    b3 = -xi[0, 2] / xi[3, 2]

    sindy_coefficients_transformed = np.zeros(xi.shape)
    sindy_coefficients_transformed[1, 0] = xi[1, 0]
    sindy_coefficients_transformed[2, 0] = xi[2, 0] * a2 / a1
    sindy_coefficients_transformed[1, 1] = xi[6, 1] * a1 / a2 * b3
    sindy_coefficients_transformed[2, 1] = xi[2, 1]
    sindy_coefficients_transformed[6, 1] = xi[6, 1] * a1 * a3 / a2
    sindy_coefficients_transformed[3, 2] = xi[3, 2]
    sindy_coefficients_transformed[5, 2] = xi[5, 2] * a1 * a2 / a3

    z0_transformed = np.array([test_data['z'][0, 0] / a1, test_data['z'][0, 1] / a2, (test_data['z'][0, 2] - b3) / a3])

    return z0_transformed, sindy_coefficients_transformed