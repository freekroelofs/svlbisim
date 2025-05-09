import ehtim as eh
import ehtim.scattering.stochastic_optics as so
import numpy as np
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rec

def gen_visgrid(obs, ncells, maxbl):
    visgrid = np.zeros((ncells, ncells), dtype='complex')
    sigmagrid = np.zeros((ncells, ncells))
    nums = np.zeros((ncells, ncells))

    for i in range(len(obs.data)):
        ugrid = int(np.round(obs.data['u'][i] / maxbl * ncells/2)) + int(ncells/2)
        vgrid = int(np.round(obs.data['v'][i] / maxbl * ncells/2)) + int(ncells/2)
        if (ugrid > 0):
           ugrid -= 1
        if (vgrid > 0):
           vgrid -= 1
        visgrid[ugrid,vgrid] += obs.data['vis'][i]
        sigmagrid[ugrid,vgrid] += obs.data['sigma'][i]**2
        nums[ugrid,vgrid]+=1.
        
    for ugrid in range(len(nums)):
        for vgrid in range(len(nums[0])):
            if (nums[ugrid, vgrid] == 0):
                nums[ugrid, vgrid] = 1
            sigmagrid[ugrid,vgrid] = np.sqrt(sigmagrid[ugrid,vgrid])/nums[ugrid,vgrid]

    visgrid = visgrid/nums

    return visgrid, sigmagrid, nums


def griduv(obs, out, ncells, fov):
    maxbl = np.max(np.sqrt(obs.data['u']**2+obs.data['v']**2))

    # Check cell size
    if ncells%2 != 0:
        ncells += 1
    cellsize_smearlimit = 1/fov
    ncells_smearlimit = int(round(maxbl*2/cellsize_smearlimit, 0))
    if ncells < ncells_smearlimit:
        print('Grid cell size too large, setting to uv-smearing limit')
        ncells = ncells_smearlimit
    print('Gridding visibilities in %s x %s grid'%(ncells, ncells))

    # Pre-process observation
    # Stokes I only: move everything to v > 0
    obs_preproc = obs.copy()
    for i in range(len(obs_preproc.data)):
        if obs_preproc.data['v'][i] < 0:
            obs_preproc.data['v'][i] *= -1
            obs_preproc.data['u'][i] *= -1
            obs_preproc.data['vis'][i] = np.conj(obs_preproc.data['vis'][i])
    
    # Grid visibilities        
    visgrid, sigmagrid, nums = gen_visgrid(obs_preproc, ncells, maxbl)

    # Make observation object
    obs_grid = obs_preproc.copy()
    obs_grid.data = obs_grid.data[0]

    griddata = []
    ctr = 0
    for ugrid in range(len(visgrid)):
        for vgrid in range(len(visgrid[0])):
            griddata.append(np.array((ctr, nums[ugrid, vgrid], 'SAT1', 'SAT2', 0.0, 0.0,
                            ugrid*(maxbl*2/ncells) - maxbl, vgrid*(maxbl*2/ncells) - maxbl,
                            visgrid[ugrid,vgrid], 0.0, 0.0, 0.0, sigmagrid[ugrid,vgrid], 0.0, 0.0, sigmagrid[ugrid,vgrid]),
                            dtype=eh.DTPOL_STOKES))
            ctr +=1
    obs_grid.data=rec.stack_arrays((obs_grid.data, griddata), asrecarray=True, usemask=False)
    obs_grid.data = np.delete(obs_grid.data, [0])

    # Get rid of zeros and save
    todel = []
    for i in range(len(obs_grid.data)):
        if np.abs(obs_grid.data['vis'][i]) == 0.0:
            todel.append(i)
    obs_grid.data = np.delete(obs_grid.data, todel)
    obs_grid.save_uvfits(out + '_gridded.uvfits')

    return obs_grid

def calc_fft(obs, out, ncells):
    maxbl = np.max(np.sqrt(obs.data['u']**2+obs.data['v']**2))
    
    # FFT image: need to add visibility conjugates before gridding
    obs_preproc = obs.copy()
    obs_conj = []
    for i in range(len(obs.data)):
        obs_conj.append(np.array((obs.data['time'][i], obs.data['tint'][i], 'SAT1', 'SAT2', 0.0, 0.0,
                -obs.data['u'][i], -obs.data['v'][i],
                np.conj(obs.data['vis'][i]), 0.0, 0.0, 0.0, obs.data['sigma'][i], 0.0, 0.0, -obs.data['sigma'][i]),
                dtype=eh.DTPOL_STOKES))
    obs_preproc.data=rec.stack_arrays((obs.data, obs_conj), asrecarray=True, usemask=False)
    
    visgrid, sigmagrid, nums = gen_visgrid(obs_preproc, ncells, maxbl)

    # Make image and rotate/flip
    fft=np.abs(np.fft.fftshift(np.fft.ifft2(visgrid)))
    fov = 1 / (maxbl*2/ncells)
    im = eh.image.make_square(obs_preproc, ncells, fov)
    im.imvec = fft.flatten()
    im = im.rotate(np.pi/2)
    img = im.imvec.reshape((im.ydim, im.xdim))
    img = np.flip(img,0)
    img = img.reshape(im.ydim**2)
    im.imvec = img
    im.save_fits(out + '_fft.fits')

    return im


def main(params):
    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)
    out = params['outdir'] + '/' + params['outtag']
    fov = float(params['fov'])*eh.RADPERUAS
    ncells = int(params['ncells'])
    obs_grid = griduv(obs, out, ncells, fov)
    im_fft = calc_fft(obs, out, ncells)

    # Deblur FFT for SGRA
    if params['source'] == 'SGRA':
        sm = so.ScatteringModel()
        obs_deblur = sm.Deblur_obs(obs)
        out_deblur = params['outdir'] + '/' + params['outtag'] + '_deblur'
        im_fft_deblur = calc_fft(obs_deblur, out_deblur, ncells)

    return 0




