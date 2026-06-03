"""Code to perform a Bayesian fit of visibility amplitude data to extract diameter of a ring"""
# common imports
import numpy as np
import matplotlib.pyplot as plt

# importing related to ehtim 
import os
import sys
import json

import ehtim as eh
import ehtim.model as em

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# defining the parameters of the ring
params = {
    'flux': 1.0,
    'diam': 51 * eh.RADPERUAS,
    'width': 1 * eh.RADPERUAS,
    'beta_list': [0 + 0.33j],
    'stretch': 1.05,
    'stretch_PA': 0
}


def stretch(x, y, params):
    x_stretch = ((x) * (np.cos(params['stretch_PA'])**2 + np.sin(params['stretch_PA'])**2 / params['stretch'])
               + (y) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA']) * (1.0/params['stretch'] - 1.0))
    y_stretch = ((y) * (np.cos(params['stretch_PA'])**2 / params['stretch'] + np.sin(params['stretch_PA'])**2)
               + (x) * np.cos(params['stretch_PA']) * np.sin(params['stretch_PA']) * (1.0/params['stretch'] - 1.0))
    return x_stretch, y_stretch


def stretched_ring(params, phi_array):
    diameter = params['diam']
    x_circ = diameter/2 * np.cos(phi_array)
    y_circ = diameter/2 * np.sin(phi_array)
    x_stretched, y_stretched = stretch(x_circ, y_circ, params)
    return x_stretched, y_stretched


def diameter_stretched_ring(params, phi_array):
    x, y = stretched_ring(params, phi_array)
    diameter = 2*np.sqrt(x**2 + y**2)
    return diameter


def main(params):
    output_folder = 'simulated_mring/'

    # configuring the image
    npix = 1024
    fov = 160 * eh.RADPERUAS   # field of view in radians

    # create model instance, and add a mring
    mod = em.Model()
    mod = mod.add_stretched_thick_mring(F0=params['flux'], 
                                d=params['diam'], 
                                alpha=params['width'], 
                                beta_list=params['beta_list'],
                                stretch=params['stretch'],
                                stretch_PA=params['stretch_PA'])

    # make the image
    model_im = mod.make_image(fov, npix)

    model_im.display(cbar_unit=['Tb'], has_title=False)
    # plt.savefig(output_folder + "mring_image.pdf", bbox_inches='tight')
    plt.show(block=True)
    
    # # saving image and the true parameters used (important for validation)
    # model_im.save_fits(output_folder + 'mring_image.fits')
    # save_params = {k: str(v) for k, v in params.items()}
    # with open(output_folder + "model_parameters.txt", "w") as file:
    #     json.dump(save_params, file)

main(params)