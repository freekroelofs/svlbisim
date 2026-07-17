# Based on ngEHT Analysis Challenge imaging script by Antonio Fuentes

import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

prior_fwhm  = 500.*eh.RADPERUAS  # Gaussian prior FWHM (radians)
sys_noise   = 0.01               # fractional systematic noise
                                 # added to complex visibilities

reg_term  = {'simple' : 0,       # Maximum-Entropy
             'tv'     : 0.03,       # Total Variation
             'tv2'    : 0,       # Total Squared Variation
             'l1'     : 0.1,      # L1 sparsity prior
             'flux'   : 0.}      # compact flux constraint

data_term = {'vis'    : 1.}      # complex visibilities

ttype     = 'fast'               # Type of Fourier transform ('direct', 'nfft', or 'fast')
npix      = 128                  # Number of pixels across the reconstructed image
maxit     = 350                  # Maximum number of convergence iterations for imaging
niter     = 5                    # Number of imaging iterations
stop      = 1e-6                 # Imager stopping criterion

def converge(imgr, res, major=niter, blur_frac=0.3):
    for repeat in range(major):
        init = imgr.out_last().blur_circ(blur_frac*res)
        imgr.init_next = init
        imgr.make_image_I(show_updates=False)
        
def image(obs, out, tflux, fov):
    res = obs.res()
    gaussprior = eh.image.make_square(obs, npix, fov)
    gaussprior = gaussprior.add_gauss(tflux*0.99, (prior_fwhm, prior_fwhm, 0., 0., 0.))
    gaussprior = gaussprior.add_gauss(tflux*1e-2, (prior_fwhm, prior_fwhm, 0, 100*eh.RADPERUAS, -100*eh.RADPERUAS))
    
    imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior,
                        flux=tflux, data_term=data_term, maxit=maxit,
                        norm_reg=True, reg_term=reg_term, ttype=ttype,
                        stop=stop)
    imgr.make_image_I(show_updates=False)
    converge(imgr, res)

    im = imgr.out_last().copy()
    im.display(scale='gamma', label_type='scale', has_title=False, export_pdf=out + '_reconstruction.pdf')
    im.save_fits(out+'_reconstruction.fits')

    return im
    


def main(params):
    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)
    res = obs.res()
    out = params['outdir'] + '/' + params['outtag']
    model = eh.image.load_fits(params['image_path'])
    model_blur = model.blur_circ(res)
    model.display(scale='gamma', label_type='scale', has_title=False, export_pdf=out+'_groundtruth.pdf')
    model_blur.display(scale='gamma', label_type='scale', has_title=False, beamparams=[res,res,0], export_pdf=out + '_groundtruth_blur.pdf')
    tflux = model.total_flux()
    fov = model.xdim*model.psize

    im = image(obs, out, tflux, fov)

    return im




