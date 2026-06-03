
"""Code to perform a Bayesian fit of visibility amplitude data to extract diameter of a ring"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os
import sys
import ehtim as eh
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import svlbisim_input_reader
infile = sys.argv[1]
params = svlbisim_input_reader.load_yaml(infile)

save_figure = True
plt.rcParams['text.usetex'] = True # TeX rendering

# Doing this method for multiple fourier slices
def bin_amp(n_uv_steps, uvdist, amp, yerr):
    uv_min, uv_max = np.min(uvdist), np.max(uvdist) 
    bins = np.linspace(uv_min, uv_max, n_uv_steps + 1)
    uvdist_amp_yerr = np.stack((uvdist, amp, yerr), axis=1)
    bin_indices = np.digitize(uvdist_amp_yerr[:, 0], bins)

    binned_data = []

    for i in range(1, n_uv_steps + 1):
        mask = bin_indices == i

        x_bin = uvdist_amp_yerr[:, 0][mask]
        y_bin = uvdist_amp_yerr[:, 1][mask]
        yerr_bin = uvdist_amp_yerr[:, 2][mask]
        print(yerr_bin)
        weights = 1/yerr_bin**2

        x_mean = np.sum(weights*x_bin)/np.sum(weights)
        y_mean = np.sum(weights*y_bin)/np.sum(weights)
        #standard_deviation = np.std(y_bin)
        #standard_deviation_mean = standard_deviation/np.sqrt(len(y_bin))

        standard_deviation_mean = np.sqrt(1/np.sum(weights))

        binned_data.append([x_mean, y_mean, standard_deviation_mean])

    binned_data = np.array(binned_data)
    return binned_data


def directional_sort_uvdata(obs, angle, dangle, plot_uv_plane=True):
    u = obs.data['u']
    v = obs.data['v']
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']

    def angle_diff(phi, phi_center):
        return (phi - phi_center + np.pi) % (2*np.pi) - np.pi

    phi_uv = np.arctan2(v, u)
    dphi_1 = np.abs(angle_diff(phi_uv, angle))
    dphi_2 = np.abs(angle_diff(phi_uv, angle + np.pi))
    mask = (dphi_1 < dangle/2) | (dphi_2 < dangle/2)

    u_sorted = u[mask]
    v_sorted = v[mask]

    uvdist = np.sqrt(u_sorted**2 + v_sorted**2)/1e9
    amp = amp[mask]
    sigma = sigma[mask]

    if plot_uv_plane:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.scatter(obs.data['u'], obs.data['v'], s=0.1, color='grey')
        ax.scatter(u_sorted, v_sorted, s=0.5, color='green')
        plt.show()
    
    uvdist_amp_sigma_sorted = np.stack((uvdist, amp, sigma), axis=1)
    return uvdist_amp_sigma_sorted 


def angled_uvdist_amp(obs, angle, dangle=np.pi/16):
    sorted_data = directional_sort_uvdata(obs, angle, dangle)
    binned_data = bin_amp(50, uvdist=sorted_data[:,0], amp=sorted_data[:,1], yerr=sorted_data[:,2])
    return binned_data, sorted_data


# --------------------------- Running Main -----------------------------------------------
def main():
    print(f"Comparing datasets...")

    obsfile1 = '/net/vdesk/data2/tmayer/svlbisim/SgrA_kgeo_runs/a0.99i30/a0.99i30.uvfits'
    obs1 = eh.obsdata.load_uvfits(obsfile1)

    obsfile2 = '/net/vdesk/data2/tmayer/svlbisim/SgrA_kgeo_runs/a0.99i30_150K_test/a0.99i30_150K.uvfits'
    obs2 = eh.obsdata.load_uvfits(obsfile2)

    # obsfile3 = '/net/vdesk/data2/tmayer/svlbisim/svlbisim_runs/simulated_mring/output_mring.uvfits'
    # obs3 = eh.obsdata.load_uvfits(obsfile3)

    def sort_data(obs, angle):
        print(f"Running for angle {angle}...")
        bin_data_phi, data_phi = angled_uvdist_amp(obs, angle)
        return bin_data_phi, data_phi

    bin_data1, data1 = sort_data(obs1, angle=1.5)
    bin_data2, data2 = sort_data(obs2, angle=1.5)
    # bin_data3, data3 = sort_data(obs3, angle)

    fig, ax = plt.subplots()
    ax.scatter(data1[:,0], data1[:,1], s=0.1, color='orange', alpha=0.3)
    ax.scatter(data2[:,0], data2[:,1], s=0.1, color='blue', alpha=0.3)
    # ax.scatter(data3[:,0], data3[:,1], s=0.1, color='green', alpha=0.2)
    ax.errorbar(bin_data1[:,0], bin_data1[:,1], yerr=bin_data1[:,2], marker='.', linestyle='None', label='Set 1 (10K)', color='orange')
    ax.errorbar(bin_data2[:,0], bin_data2[:,1], yerr=bin_data2[:,2], marker='.', linestyle='None', label='Set 2 (150K)', color='blue')
    # ax.errorbar(bin_data3[:,0], bin_data3[:,1], yerr=bin_data3[:,2], marker='.', linestyle='None', label='150K', color='green')

    ax.set_xlabel('$uv$-distance (G$\lambda$)', fontsize=10)
    ax.set_ylabel('Visibility Amplitude (Jy)', fontsize=10)
    ax.set_yscale('log')

    ax.axvline(20, color='grey', linestyle='--', linewidth=1)
    ax.legend()

    ax.set_title(f'Visibility amplitude spectrum for different system parameters')

    plt.tight_layout()
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    if save_figure:
        plt.savefig('vis_comparison_001.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()

    return 0



main()