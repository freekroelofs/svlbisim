import ehtim as eh
import numpy as np
import os
import numpy.lib.recfunctions as rec

def observe(modelfile, modeltype, uvfile, Tsys, D, eta, bw, out, kb=1.38064852e-23):

    # Prepare uv data and image or movie
    uvdata = eh.obsdata.load_uvfits(uvfile)
    
    # Calculate system noise
    A = eta * np.pi*(0.5*D)**2 # Antenna effective area
    SEFD = (2*kb*Tsys/A)*1e26
    sigma_const = SEFD/(0.88*np.sqrt(2*bw))

    # Calculate and add sigmas to observation object
    for i in range(len(uvdata.data)):
        tint = uvdata.data['tint'][i]
        sigma = sigma_const/np.sqrt(tint)
        uvdata.data['sigma'][i] = sigma

    # Observe image or movie
    if modeltype == 'image':
        im = eh.image.load_fits(modelfile)
        im.ra = uvdata.ra
        im.dec= uvdata.dec
        im.rf = uvdata.rf
        obs = im.observe_same(uvdata, ttype='fast')
        
    elif modeltype == 'movie':
        mov = eh.movie.load_hdf5(modelfile)
        mov.ra = uvdata.ra
        mov.dec= uvdata.dec
        mov.rf = uvdata.rf        
        # Need to make sure we have at least 1 timestamp with 2 datapoints to avoid dtype issues with obs.tlist()
        extrapoint = [np.array(uvdata.data[-1], dtype=eh.DTPOL_STOKES)]
        uvdata.data=rec.stack_arrays((uvdata.data, extrapoint), asrecarray=True, usemask=False)
        obs = mov.observe_same(uvdata, ttype='fast', repeat=True)
        obs.data = np.delete(obs.data, [-1])
        
    obs.save_uvfits(out+'.uvfits')
    os.system('rm %s'%uvfile)
    
    return obs

def main(params):

    out = params['outdir'] + '/' + params['outtag']
    uvfile = out + '_uvcoords.uvfits'
    
    modelfile = params['image_path']
    if modelfile.split('.')[-1] == 'fits':
        modeltype = 'image'
    elif modelfile.split('.')[-1] == 'h5' or modelfile.split('.')[-1] == 'hdf5':
        modeltype = 'movie'
    else:
        raise ValueError('Use a fits or hdf5 file for the input model.')
    
    bw = params['bw']*1e9
    eta = params['eta']
    Tsys = params['tsys']
    D = params['diameter']

    obs = observe(modelfile, modeltype, uvfile, Tsys, D, eta, bw, out)

    return obs




