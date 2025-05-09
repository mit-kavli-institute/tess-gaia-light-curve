import numpy as np

from tglc.epsf import make_tglc_design_matrix


def test_make_tglc_design_matrix():
    # Actual expected values
    image_shape = (150, 150)
    psf_shape_pixels = (11, 11)
    oversample_factor = 2
    star_positions = np.array([[10, 10]])
    background_strap_mask = np.zeros(image_shape)
    edge_compression_scale_factor = 1e-4

    design_matrix, n_normalization_zeros = make_tglc_design_matrix(
        image_shape,
        psf_shape_pixels,
        oversample_factor,
        star_positions,
        background_strap_mask,
        edge_compression_scale_factor,
    )
    assert design_matrix.shape[0] == 150 * 150 + 23 * 23
    assert design_matrix.shape[1] == 23 * 23 + 6
    assert n_normalization_zeros == 23 * 23

    simple_psf = np.ones((23, 23))
    simple_model_vector = np.hstack([simple_psf.reshape(23 * 23), np.zeros(6)])
    ones_around_star = np.zeros((150, 150))
    for i in range(-5, 6):
        for j in range(-5, 6):
            ones_around_star[10 + i, 10 + j] = 1
    modelled_image = np.dot(design_matrix, simple_model_vector)[:22500].reshape(150, 150)
    assert np.all(modelled_image == ones_around_star)
