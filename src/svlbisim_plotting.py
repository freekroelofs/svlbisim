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
    
    scat = ax.scatter(u, v, c=snr, s=s)
    scat2 = ax.scatter(-u, -v, c=snr, s=s)

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
    scat = ax.scatter(u[mask], v[mask], c=snr[mask], s=s)
    scat = ax.scatter(-u[mask], -v[mask], c=snr[mask], s=s)
    
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$u$ (G$\lambda$)', fontsize=fs)
    ax.set_ylabel('$v$ (G$\lambda$)', fontsize=fs)

    ax.legend()

    cbar = fig.colorbar(scat)
    cbar.set_label('S/N')
    
    fig.savefig(out + '_uv_snrthreshold.pdf', bbox_inches='tight')
        
    return 0



def main(params):
    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)
    out = params['outdir'] + '/' + params['outtag']
    
    plot_uvdist_amp(obs, out)
    plot_uv_amp(obs, out)
    plot_uv_snr(obs, out)
    plot_uv_snrthreshold(obs, out)

    return 0




