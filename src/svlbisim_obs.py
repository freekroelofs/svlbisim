import ehtim as eh
import numpy as np
import os

def observe(modelfile, uvfile, Tsys, D, eta, bw, out, kb=1.38064852e-23):

    # Prepare uv data and image
    uvdata = eh.obsdata.load_uvfits(uvfile)
    im = eh.image.load_fits(modelfile)
    im.ra = uvdata.ra
    im.dec= uvdata.dec
    im.rf = uvdata.rf

    # Calculate system noise
    A = eta * np.pi*(0.5*D)**2 # Antenna effective area
    SEFD = (2*kb*Tsys/A)*1e26
    print('bw', bw)
    sigma_const = SEFD/(0.88*np.sqrt(2*bw))

    # Calculate and add sigmas to observation object
    for i in range(len(uvdata.data)):
        tint = uvdata.data['tint'][i]
        sigma = sigma_const/np.sqrt(tint)
        uvdata.data['sigma'][i] = sigma
        print(SEFD, bw, sigma_const, tint, sigma)

    obs = im.observe_same(uvdata, sgrscat=False, ttype='fast')
    obs.save_uvfits(out+'.uvfits')
    os.system('rm %s'%uvfile)
    
    return obs

def main(params):

    out = params['outdir'] + '/' + params['outtag']
    uvfile = out + '_uvcoords.uvfits'
    modelfile = params['image_path']
    bw = params['bw']*1e9
    eta = params['eta']
    Tsys = params['tsys']
    D = params['diameter']

    obs = observe(modelfile, uvfile, Tsys, D, eta, bw, out)

    return obs




