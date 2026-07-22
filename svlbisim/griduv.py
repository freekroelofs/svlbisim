import ehtim as eh
import ehtim.scattering.stochastic_optics as so
import numpy as np
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rec

def grid_index(coord, maxbl, ncells):
    idx = np.round(coord / maxbl * ncells/2).astype(int) + int(ncells/2)
    idx = np.where(idx > 0, idx - 1, idx)
    return idx

def gen_visgrid(obs, ncells, maxbl):
    visgrid = np.zeros((ncells, ncells), dtype='complex')
    qvisgrid = np.zeros((ncells, ncells), dtype='complex')
    uvisgrid = np.zeros((ncells, ncells), dtype='complex')
    vvisgrid = np.zeros((ncells, ncells), dtype='complex')
    sigmagrid = np.zeros((ncells, ncells))

    nums = np.zeros((ncells, ncells))

    ugrid = grid_index(obs.data['u'], maxbl, ncells)
    vgrid = grid_index(obs.data['v'], maxbl, ncells)

    np.add.at(visgrid, (ugrid, vgrid), obs.data['vis'])
    np.add.at(qvisgrid, (ugrid, vgrid), obs.data['qvis'])
    np.add.at(uvisgrid, (ugrid, vgrid), obs.data['uvis'])
    np.add.at(vvisgrid, (ugrid, vgrid), obs.data['vvis'])
    np.add.at(sigmagrid, (ugrid, vgrid), obs.data['sigma']**2)
    np.add.at(nums, (ugrid, vgrid), 1.)

    nums[nums == 0] = 1
    sigmagrid = np.sqrt(sigmagrid) / nums

    visgrid = visgrid/nums
    qvisgrid = qvisgrid/nums
    uvisgrid = uvisgrid/nums
    vvisgrid = vvisgrid/nums

    return visgrid, qvisgrid, uvisgrid, vvisgrid, sigmagrid, nums


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

    # Grid visibilities
    visgrid, qvisgrid, uvisgrid, vvisgrid, sigmagrid, nums = gen_visgrid(obs, ncells, maxbl)

    # Make observation object
    obs_grid = obs.copy()
    obs_grid.data = obs_grid.data[0]

    griddata = []
    ctr = 0
    for ugrid in range(len(visgrid)):
        for vgrid in range(len(visgrid[0])):
            griddata.append(np.array((ctr, nums[ugrid, vgrid], 'SAT1', 'SAT2', 0.0, 0.0,
                            ugrid*(maxbl*2/ncells) - maxbl, vgrid*(maxbl*2/ncells) - maxbl,
                            visgrid[ugrid,vgrid], qvisgrid[ugrid,vgrid], uvisgrid[ugrid,vgrid], vvisgrid[ugrid,vgrid], sigmagrid[ugrid,vgrid], sigmagrid[ugrid,vgrid], sigmagrid[ugrid,vgrid], sigmagrid[ugrid,vgrid]),
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

    return obs_grid, visgrid, qvisgrid, uvisgrid, vvisgrid, nums, maxbl, ncells

def calc_fft(obs_grid, out, ncells, fov):
    im = obs_grid.dirtyimage(ncells,fov)
    im.save_fits(out + '_fft.fits')

    return im

def calc_fft_from_grid(obs_grid, visgrid, qvisgrid, uvisgrid, vvisgrid, nums, maxbl, ncells, out):
    # Direct FFT of the already-gridded visibilities
    centers = np.arange(ncells) * (maxbl*2/ncells) - maxbl
    mirror_idx = grid_index(-centers, maxbl, ncells)

    def hermitian_symmetrize(grid):
        mirror_grid = np.conj(grid[np.ix_(mirror_idx, mirror_idx)])
        mirror_nums = nums[np.ix_(mirror_idx, mirror_idx)]
        total_nums = nums + mirror_nums
        total_nums = np.where(total_nums == 0, 1, total_nums)
        return (nums*grid + mirror_nums*mirror_grid) / total_nums

    visgrid_herm = hermitian_symmetrize(visgrid)
    qvisgrid_herm = hermitian_symmetrize(qvisgrid)
    uvisgrid_herm = hermitian_symmetrize(uvisgrid)
    vvisgrid_herm = hermitian_symmetrize(vvisgrid)

    dc_idx = int(grid_index(np.array([0.]), maxbl, ncells)[0])

    def dc_to_origin(grid):
        return np.roll(grid, -dc_idx, axis=(0, 1))

    fft=np.fft.fftshift(np.fft.ifft2(dc_to_origin(visgrid_herm))).real
    qfft=np.fft.fftshift(np.fft.ifft2(dc_to_origin(qvisgrid_herm))).real
    ufft=np.fft.fftshift(np.fft.ifft2(dc_to_origin(uvisgrid_herm))).real
    vfft=np.fft.fftshift(np.fft.ifft2(dc_to_origin(vvisgrid_herm))).real
    fov = 1 / (maxbl*2/ncells)
    im = eh.image.make_square(obs_grid, ncells, fov)
    im.imvec = fft.flatten()
    im.qvec = qfft.flatten()
    im.uvec = ufft.flatten()
    im.vvec = vfft.flatten()
    im.save_fits(out + '_fft-from_grid.fits')

    return im


def main(params):
    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)
    out = params['outdir'] + '/' + params['outtag']
    fov = float(params['fov'])*eh.RADPERUAS
    ncells = int(params['ncells'])
    obs_grid, visgrid, qvisgrid, uvisgrid, vvisgrid, nums, maxbl, ncells = griduv(obs, out, ncells, fov)

    # Match calc_fft()'s output image to the input model's field of view/pixel size
    modelfile = params['image_path']
    if modelfile.split('.')[-1] == 'fits':
        model = eh.image.load_fits(modelfile)
    elif modelfile.split('.')[-1] == 'h5' or modelfile.split('.')[-1] == 'hdf5':
        model = eh.movie.load_hdf5(modelfile)
    else:
        raise ValueError('Use a fits or hdf5 file for the input model.')
    ncells_out = model.xdim
    fov_out = model.xdim * model.psize

    im_fft = calc_fft(obs_grid, out, ncells_out, fov_out)
    im_fft_from_grid = calc_fft_from_grid(obs_grid, visgrid, qvisgrid, uvisgrid, vvisgrid, nums, maxbl, ncells, out)

    # Deblur FFT
    if params['deblur'] == 'True':
        sm = so.ScatteringModel()
        obs_deblur = sm.Deblur_obs(obs_grid)
        out_deblur = params['outdir'] + '/' + params['outtag'] + '_deblur'

        im_fft_deblur = calc_fft(obs_deblur, out_deblur, ncells_out, fov_out)

    return 0




