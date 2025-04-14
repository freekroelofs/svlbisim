import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rec

def griduv(obs, out, ncells, fov):
    maxbl = np.max(np.sqrt(obs.data['u']**2+obs.data['v']**2))
    
    if ncells%2 != 0:
        ncells += 1
    cellsize_smearlimit = 1/fov
    ncells_smearlimit = int(round(maxbl*2/cellsize_smearlimit, 0))
    if ncells < ncells_smearlimit:
        print('Grid cell size too large, setting to uv-smearing limit')
        ncells = ncells_smearlimit
    print('Gridding visibilities in %s x %s grid'%(ncells, ncells))

    # Stokes I only: move everything to v > 0
    for i in range(len(obs.data)):
        if obs.data['v'][i] < 0:
            obs.data['v'][i] *= -1
            obs.data['u'][i] *= -1
            obs.data['vis'][i] = np.conj(obs.data['vis'][i])
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

    obs_grid = obs.copy()
    obs_grid.data = obs_grid.data[0]

    griddata = []
    ctr = 0
    for ugrid in range(len(nums)):
        for vgrid in range(len(nums[0])):
            griddata.append(np.array((ctr, nums[ugrid, vgrid], 'SAT1', 'SAT2', 0.0, 0.0,
                            ugrid*(maxbl*2/ncells) - maxbl, vgrid*(maxbl*2/ncells) - maxbl,
                            visgrid[ugrid,vgrid], 0.0, 0.0, 0.0, sigmagrid[ugrid,vgrid], 0.0, 0.0, sigmagrid[ugrid,vgrid]),
                            dtype=eh.DTPOL_STOKES))
            ctr +=1
    obs_grid.data=rec.stack_arrays((obs_grid.data, griddata), asrecarray=True, usemask=False)
    obs_grid.data = np.delete(obs_grid.data, [0])
    todel = []
    for i in range(len(obs_grid.data)):
        if np.abs(obs_grid.data['vis'][i]) == 0.0:
            todel.append(i)
    obs_grid.data = np.delete(obs_grid.data, todel)
    obs_grid.save_uvfits(out + '_gridded.uvfits')

    return obs_grid


def main(params):
    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)
    out = params['outdir'] + '/' + params['outtag']
    fov = float(params['fov'])*eh.RADPERUAS
    ncells = int(params['ncells'])
    griduv(obs, out, ncells, fov)

    return 0




