import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt

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

def plot_uv_snr(obs, out, fs=15, s=3):
    fig, ax = plt.subplots()
    u = obs.data['u']/1e9
    v = obs.data['v']/1e9
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    amp_debias = eh.observing.obs_helpers.amp_debias(amp, sigma)
    snr = amp_debias/sigma
    
    scat = ax.scatter(u, v, c=snr, s=s, norm='log')
    scat2 = ax.scatter(-u, -v, c=snr, s=s, norm='log')

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    cbar = fig.colorbar(scat)
    cbar.set_label('S/N')
    
    fig.savefig(out + '_uv_snr.pdf', bbox_inches='tight')
        
    return 0

def plot_uv_snrthreshold(obs, out, fs=15, s=3, threshold=3):
    fig, ax = plt.subplots()
    u = obs.data['u']/1e9
    v = obs.data['v']/1e9
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    amp_debias = eh.observing.obs_helpers.amp_debias(amp, sigma)
    snr = amp_debias/sigma
    mask = snr > threshold

    nondet = ax.scatter(u[~mask], v[~mask], s=s, color='gray', label='S/N < %s'%threshold)
    scat = ax.scatter(u[mask], v[mask], c=snr[mask], s=s, norm='log')
    scat = ax.scatter(-u[mask], -v[mask], c=snr[mask], s=s, norm='log')
    
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    ax.legend()

    cbar = fig.colorbar(scat)
    cbar.set_label('S/N')
    
    fig.savefig(out + '_uv_snrthreshold.pdf', bbox_inches='tight')
        
    return 0

def plot_uv_snr_grid(obs, out, ncells, fs=15, s=3):
    fig, ax = plt.subplots()
    u = obs.data['u']/1e9
    v = obs.data['v']/1e9
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    amp_debias = eh.observing.obs_helpers.amp_debias(amp, sigma)
    snr = amp_debias/sigma
    
    scat = ax.scatter(u, v, c=snr, s=s, norm='log')
    scat2 = ax.scatter(-u, -v, c=snr, s=s, norm='log')

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    cbar = fig.colorbar(scat, fraction=0.040, pad=0.13, orientation='horizontal')
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
    elif modeltype == 'movie':
        mov = eh.movie.load_hdf5(params['image_path'])
        avgim = mov.avg_frame()
        plot_groundtruth(avgim, out)
        
    if params['grid_uv'] == 'True':
        obsfile_grid = params['outdir'] + '/' + params['outtag'] + '_gridded.uvfits'
        obs_grid = eh.obsdata.load_uvfits(obsfile_grid)
        plot_uv_snr_grid(obs_grid, out, int(params['ncells']))

        fftimfile = params['outdir'] + '/' + params['outtag'] + '_fft.fits'
        fftim = eh.image.load_fits(fftimfile)
        plot_fft(fftim, out)

        if params['source'] == 'SGRA':
            fftimfile = params['outdir'] + '/' + params['outtag'] + '_deblur_fft.fits'
            fftim = eh.image.load_fits(fftimfile)
            out = params['outdir'] + '/' + params['outtag'] + '_deblur'
            plot_fft(fftim, out)


    return 0




