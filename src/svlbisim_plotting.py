import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

def plot_uvdist_amp(obs, out, fs=15):
    fig, ax = plt.subplots()
    uvdist = np.sqrt(obs.data['u']**2 + obs.data['v']**2)/1e9
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    
    ax.errorbar(uvdist, amp, yerr=sigma, marker='.', linestyle='None')

    ax.set_xlabel('$uv$-distance (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('Visibility Amplitude (Jy)', fontsize=fs)
    
    fig.savefig(out + '_uvdist_amp.pdf', bbox_inches='tight')
        
    return 0

def plot_uv_amp(obs, out, fs=15, s=3):
    fig, ax = plt.subplots()
    u = obs.data['u']/1e9
    v = obs.data['v']/1e9
    amp = np.abs(obs.data['vis'])
    
    scat = ax.scatter(u, v, c=amp, s=s)
    scat2 = ax.scatter(-u, -v, c=amp, s=s)

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    cbar = fig.colorbar(scat)
    cbar.set_label('Visibility Amplitude (Jy)')

    fig.savefig(out + '_uv_amp.pdf', bbox_inches='tight')
        
    return 0

def plot_uv_snr(obs, out, fs=15, s=3, vmin=0.1, vmax=1e3):
    fig, ax = plt.subplots()
    u = obs.data['u']/1e9
    v = obs.data['v']/1e9
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    amp_debias = eh.observing.obs_helpers.amp_debias(amp, sigma)
    snr = amp_debias/sigma
    
    scat = ax.scatter(u, v, c=snr, s=s, norm='log', vmin=vmin, vmax=vmax, edgecolor='none')
    scat2 = ax.scatter(-u, -v, c=snr, s=s, norm='log', vmin=vmin, vmax=vmax, edgecolor='none')

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    cbar = fig.colorbar(scat, fraction=0.040, pad=0.14, orientation='horizontal')
    cbar.set_label('S/N')
    
    fig.savefig(out + '_uv_snr.pdf', bbox_inches='tight')
        
    return 0

def plot_uv_snrthreshold(obs, out, fs=15, s=3, threshold=1):
    fig, ax = plt.subplots()
    u = obs.data['u']/1e9
    v = obs.data['v']/1e9
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    amp_debias = eh.observing.obs_helpers.amp_debias(amp, sigma)
    snr = amp_debias/sigma
    mask = snr > threshold

    
    scat = ax.scatter(u[mask], v[mask], c=snr[mask], s=s, norm='log', edgecolor='none')
    scat = ax.scatter(-u[mask], -v[mask], c=snr[mask], s=s, norm='log', edgecolor='none')
    nondet = ax.scatter(u[~mask], v[~mask], s=s, color='gray', label='S/N < %s'%threshold, edgecolor='none')
    #nondet = ax.scatter(-u[~mask], -v[~mask], s=s, color='gray')
    
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    ax.legend()

    cbar = fig.colorbar(scat, fraction=0.040, pad=0.14, orientation='horizontal')
    cbar.set_label('S/N')
    
    fig.savefig(out + '_uv_snrthreshold.pdf', bbox_inches='tight')
        
    return 0

def plot_uv_snr_grid(obs, out, ncells, fs=15, s=5, vmin=0.1, vmax=1e3):
    fig, ax = plt.subplots()
    u = obs.data['u']/1e9
    v = obs.data['v']/1e9
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    amp_debias = eh.observing.obs_helpers.amp_debias(amp, sigma)
    snr = amp/sigma
    scat = ax.scatter(u, v, c=snr, s=s, norm='log', vmin=vmin, vmax=vmax, edgecolor='none', marker='s')
    #scat2 = ax.scatter(-u, -v, c=snr, s=s, norm='log', vmin=vmin, vmax=vmax, edgecolor='none')

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    cbar = fig.colorbar(scat, fraction=0.040, pad=0.14, orientation='horizontal')
    cbar.set_label('S/N')
    plt.tight_layout()
    fig.savefig(out + '_uv_snr_grid.pdf', bbox_inches='tight')
        
    return 0

def plot_fft(fftim, out):
    fftim.display(scale='gamma', label_type='scale', cbar_unit=['Tb'], cbar_orientation='horizontal', has_title=False, export_pdf=out + '_fft.pdf')
        
    return 0

def plot_groundtruth(model, out):
    model.display(scale='gamma', label_type='scale', cbar_unit=['Tb'], cbar_orientation='horizontal', has_title=False, export_pdf=out + '_groundtruth.pdf')
        
    return 0

def plot_im_pol(im, out, Pmin=0., Pmax=1., cfun='afmhot', axis=True, dobar=True, scale=False, fs=12, pcut=0.05, nvec=50):
    # Adapted from EHT pol plotting script

    Imax=max(im.imvec)

    fig, ax = plt.subplots()
    pixel=im.psize/eh.RADPERUAS #uas
    FOV=pixel*im.xdim

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                    slice(-FOV/2, FOV/2, pixel)]

    # reverse x axis
    plt.axis([FOV/2,-FOV/2,-FOV/2,FOV/2])

    imarr = im.imvec.reshape(im.ydim, im.xdim)
    
    #to brightness temperature
    TBfactor = 3.254e13/(im.rf**2 * im.psize**2)/1e9

    vmax = np.max(imarr*TBfactor)

    implot = plt.imshow(imarr*TBfactor,origin='upper',cmap=cfun,
               extent=[np.max(x),np.min(x),np.min(y),np.max(y)],interpolation='Gaussian')
    
    qarr = im.qvec.reshape(im.ydim, im.xdim)
    uarr = im.uvec.reshape(im.ydim, im.xdim)
    
    # length of the tick proportional to sqrt(Q^2+U^2)
    amp = np.sqrt(im.qvec*im.qvec + im.uvec*im.uvec)
    scal=np.max(amp)
    vx = (-np.sin(np.angle(im.qvec+1j*im.uvec)/2)*amp/scal).reshape(im.ydim, im.xdim)
    vy = ( np.cos(np.angle(im.qvec+1j*im.uvec)/2)*amp/scal).reshape(im.ydim, im.xdim)

    # tick color proportional to mfrac
    mfrac=(amp/im.imvec).reshape(im.xdim, im.ydim)
    mfrac_map=(np.sqrt(im.qvec**2+im.uvec**2)).reshape(im.xdim, im.ydim)
    QUmax=max(np.sqrt(im.qvec**2+im.uvec**2))
    mfrac_m = np.ma.masked_where(mfrac_map < pcut * QUmax , mfrac)
    mfrac_m = np.ma.masked_where(imarr < pcut * Imax, mfrac_m)

    x = np.ma.masked_where(imarr < pcut * Imax, x) 
    y = np.ma.masked_where(imarr < pcut * Imax, y) 
    vx = np.ma.masked_where(imarr < pcut * Imax, vx) 
    vy = np.ma.masked_where(imarr < pcut * Imax, vy)

    skip = int(round(im.xdim/nvec,0))
    
    tickplot = plt.quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
               mfrac_m[::skip,::skip],
               headlength=0,
               headwidth = 1,
               pivot='mid',
               width=0.005,
               cmap='rainbow',
               scale=20)
    
    plt.clim(Pmin,Pmax)
    if dobar == True:
        cbar = fig.colorbar(ScalarMappable(norm=tickplot.norm, cmap=tickplot.cmap), pad=0.14,fraction=0.038, orientation="vertical", ax=ax) 
        cbar.set_label(r'Fractional Polarization $|m|$', fontsize=fs)            
        
        cbar_stokesi = fig.colorbar(ScalarMappable(norm=implot.norm, cmap=implot.cmap), pad=0.04, fraction=0.046,label='$T_B$ ($10^9$ K)', ax=ax)
        cbar_stokesi.set_label(r'$T_B$ ($10^9$ K)', fontsize=fs) 
        
    if scale == True:
        fov_uas = im.xdim * im.psize / eh.RADPERUAS # get the fov in uas
        roughfactor = 1./3. # make the bar about 1/3 the fov
        fov_scale = int( math.ceil(fov_uas * roughfactor / 10.0 ) ) * 10 
        start = im.xdim * roughfactor / 3.0 # select the start location
        end = start + fov_scale/fov_uas * im.xdim # determine the end location 
        plt.plot([start*im.psize/eh.RADPERUAS-2, end*im.psize/eh.RADPERUAS-2],
                 [-(im.ydim-start-3)*im.psize/eh.RADPERUAS/2, -(im.ydim-start-3)*im.psize/eh.RADPERUAS/2],
                 color="black", lw=3) # plot a line
        plt.text(x=(start+end)/2.0*im.psize/eh.RADPERUAS-2, y=-(im.ydim-start+im.ydim/30)*im.psize/eh.RADPERUAS/2, 
                 s= str(fov_scale) + " $\mu$as", color="black", 
                 ha="center", va="center",fontsize=50.)

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout(pad=0)
    plt.savefig(out,facecolor='none', bbox_inches='tight', pad_inches = 0.0,edgecolor='none', bbox_extra_artists=[])
    plt.cla()

    return 0

def main(params):
    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)
    out = params['outdir'] + '/' + params['outtag']
    modelfile = params['image_path']
    if modelfile.split('.')[-1] == 'fits':
        modeltype = 'image'
    elif modelfile.split('.')[-1] == 'h5' or modelfile.split('.')[-1] == 'hdf5':
        modeltype = 'movie'
    
    plot_uvdist_amp(obs, out)
    plot_uv_amp(obs, out)
    plot_uv_snr(obs, out)
    plot_uv_snrthreshold(obs, out)
    if modeltype == 'image':
        model = eh.image.load_fits(params['image_path'])
        plot_groundtruth(model, out)
        if np.any(model.qvec):
            plot_im_pol(model, out+ '_groundtruth_pol.pdf')
    elif modeltype == 'movie':
        mov = eh.movie.load_hdf5(params['image_path'])
        avgim = mov.avg_frame()
        plot_groundtruth(avgim, out)
        if np.any(model.qvec):
            plot_im_pol(avgim, out+ '_groundtruth_pol.pdf')
        
    if params['grid_uv'] == 'True':
        obsfile_grid = params['outdir'] + '/' + params['outtag'] + '_gridded.uvfits'
        obs_grid = eh.obsdata.load_uvfits(obsfile_grid)
        plot_uv_snr_grid(obs_grid, out, int(params['ncells']))

        fftimfile = params['outdir'] + '/' + params['outtag'] + '_fft.fits'
        fftim = eh.image.load_fits(fftimfile)
        plot_fft(fftim, out)

        if np.any(model.qvec):
            plot_im_pol(fftim, out+ '_fft_pol.pdf')

        if params['source'] == 'SGRA':
            fftimfile = params['outdir'] + '/' + params['outtag'] + '_deblur_fft.fits'
            fftim = eh.image.load_fits(fftimfile)
            out = params['outdir'] + '/' + params['outtag'] + '_deblur'
            plot_fft(fftim, out)


    return 0




