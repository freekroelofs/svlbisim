
"""Code to compile results from bayesian fitting in projected diameter plot"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import json
import arviz as az

import os
import sys
import ehtim as eh
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import svlbisim_input_reader
infile = sys.argv[1]
params = svlbisim_input_reader.load_yaml(infile)

plt.rcParams['text.usetex'] = True # TeX rendering

save_figure = True
directory = "/net/vdesk/data2/tmayer/svlbisim/SgrA_kgeo_runs/slice_method_results/"


def load_circlipse_dictionary(directory="mcmc_cache/", filename="ellipse_dict.json"):
    with open(directory + filename, "r") as f:
        dictionary = json.load(f)
    return dictionary

def diameter_ellipse(params, phi):
    R0 = params['R0'] / eh.RADPERUAS /1e9
    Rf = params['Rf']
    psi = params['psi']

    diameter = 2*np.sqrt((R0*np.cos(phi - psi))**2 + (R0*Rf*np.sin(phi - psi))**2)
    return diameter


def diameter_circlipse(params, phi):
    R0 = params['R0'] / eh.RADPERUAS /1e9
    R1 = params['R1'] / eh.RADPERUAS /1e9
    R2 = params['R2'] / eh.RADPERUAS /1e9
    phi0 = params['phi0']

    diameter_phi = 2*(R0 + np.sqrt((R1*np.cos(phi - phi0))**2 + (R2*np.sin(phi - phi0))**2))
    return diameter_phi


def polar_to_cartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def re_center(x, y):
    dx = np.diff(x)
    dy = np.diff(y)

    ds = np.sqrt(dx**2 + dy**2)

    # Midpoints (so ds aligns)
    x_mid = 0.5*(x[1:] + x[:-1])
    y_mid = 0.5*(y[1:] + y[:-1])

    x0 = np.sum(x_mid * ds) / np.sum(ds)
    y0 = np.sum(y_mid * ds) / np.sum(ds)

    x = x - x0
    y = y - y0
    return x, y


def diameter_vs_phi(x, y, phi_grid):
    diameter = np.zeros_like(phi_grid)

    for k, phi in enumerate(phi_grid):
        nx = np.cos(phi)
        ny = np.sin(phi)

        # Project curve onto ray direction
        s = x*nx + y*ny

        diameter[k] = s.max() - s.min()

    return diameter


def kerr_shape(D, i, a, M, nr=10000, plot=True, plot2=True):
    """Kerr critical shape from Johnson et. al. 2020
    D   : Distance to the spin axis
    i   : Inclination w.r.t spin axis
    a   : spin parameter, where J=aM is angular momentum. Range [0, 1]
    M   : Mass of Kerr blackhole
    """
    # In the equations below a is 0<a<M 
    a = a*M 

    r_plus = 2*M*(1 + np.cos(2/3*np.arccos(a/M))) 
    r_min = 2*M*(1 + np.cos(2/3*np.arccos(-a/M))) 
    
    r_array = np.linspace(r_min, r_plus, nr) 

    eps = 1e-6 * M
    r = r_array[np.abs(r_array - M) > eps]

    delta = r**2 -2*M*r + a**2
    l = (M*(r**2 - a**2) - r*delta)/(a*(r-M))

    u_plus = r/(a**2*(r - M)**2)*(-r**3 + 3*M**2*r - 2*a**2*M + 2*np.sqrt(M*delta*(2*r**3 - 3*M*r**2 + a**2*M)))
    u_min  = r/(a**2*(r - M)**2)*(-r**3 + 3*M**2*r - 2*a**2*M - 2*np.sqrt(M*delta*(2*r**3 - 3*M*r**2 + a**2*M)))

    rho = (1/D)*np.sqrt(a**2*(np.cos(i)**2 - u_plus*u_min) + l**2)
    phi_rho = np.arccos(-(l)/(rho*D*np.sin(i)))
    print(phi_rho)
    
    # Remove Nans
    mask = ~np.isnan(phi_rho)
    phi_rho = phi_rho[mask]
    rho = rho[mask]

    phi1 = phi_rho
    phi2 = 2*np.pi - phi_rho

    rho_full = np.concatenate([np.flip(rho), rho])
    phi_full = np.concatenate([np.flip(phi1), phi2])

    # Interpolating the shape to extract diameter after shifting shape to the center
    phi_grid = np.linspace(0, 2*np.pi, 100)
    rho_grid = np.interp(phi_grid, phi_full, rho_full)

    x, y = polar_to_cartesian(rho_full, phi_full)
    x_grid, y_grid = polar_to_cartesian(rho_grid, phi_grid)

    #x, y = re_center(x, y)
    #x_grid, y_grid = re_center(x_grid, y_grid)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1)
        ax.scatter(x_grid, y_grid, s=2)
        ax.set_aspect('equal')
        plt.show()

    diam_phi = diameter_vs_phi(x_grid, y_grid, phi_grid)/eh.RADPERUAS

    if plot2:
        fig, ax = plt.subplots()
        ax.plot(phi_grid, diam_phi)
        plt.show()
    
    return phi_grid[:int(len(phi_grid)/2)], diam_phi[:int(len(diam_phi)/2)]


def sort_data(array):
    phi_array = np.linspace(0, np.pi, 30)
    diameter_array = array[:,0]/10**9 * 1/eh.RADPERUAS
    hpdi_lower = array[:,1]/10**9 * 1/eh.RADPERUAS
    hpdi_upper = array[:,2]/10**9 * 1/eh.RADPERUAS
    return phi_array, diameter_array, hpdi_lower, hpdi_upper


# --------------------------- Plotting -----------------------------------------------

def plot_error_bar(ax, data, label='a=0.01 i=15', color='black'):
    ax.errorbar(
    data[0], data[1],
    yerr=[data[1] - data[2], data[3] - data[1]],
    fmt='s',
    markersize=3,
    markerfacecolor=color,
    markeredgecolor=color,
    linestyle='none',
    ecolor=color,
    elinewidth=1,
    capsize=3,
    label=f'{label}'
    )


def plot_kerr_boundary(ax, angle, spin, l_color):
    MoverD = 5.29*eh.RADPERUAS
    phi, diameter = kerr_shape(
        D=1.0, # dimensionless
        i=np.deg2rad(angle),
        a=spin,
        M=MoverD, 
        plot=False, plot2=False)
    ax.plot(phi, diameter, label=f'C$_\\gamma$: $a={spin}, i={angle}$ [deg]', 
            color=l_color, linestyle='--')
    

def plot_ring(ax, ring_params, ring_function, label, color, alpha=1):
    phi_array = np.linspace(0, np.pi, 1000)
    diameter_array = ring_function(ring_params, phi_array) 
    ax.plot(phi_array, diameter_array,
            linewidth=1.5, 
            label=f'{label}', 
            color=f'{color}',
            alpha=alpha)


def plot_posterior_band(ax, directory, filename, ring_function,
                        N=10000, burnin=4500,
                        color='green', label='Posterior band'):

    data = np.load(directory + filename)
    samples = {k: data[k] for k in data.files}
    keys = list(samples.keys())

    # total samples
    n_total = len(samples[keys[0]])

    # --- remove burn-in ---
    samples = {k: v[burnin:] for k, v in samples.items()}
    n_post = len(samples[keys[0]])

    # --- subsample after burn-in ---
    idx = np.random.choice(n_post, N, replace=False)

    phi_array = np.linspace(0, np.pi, 1000)

    curves = []

    for i in idx:
        ring_params = {k: samples[k][i] for k in keys}
        diameter = ring_function(ring_params, phi_array)
        curves.append(diameter)

    curves = np.array(curves)

    # --- compute band ---
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)
    median = np.percentile(curves, 50, axis=0)

    # --- plot ---
    ax.fill_between(phi_array, lower, upper,
                    color=color, alpha=0.3, label=label)

    ax.plot(phi_array, median, color=color, lw=2.5)
    

def plot_sampled_posterior(ax, directory, filename):
    posterior_sampled = az.from_netcdf(directory + filename)
    first = True
    for i in range(len(posterior_sampled["posterior"]["R0"].values)):
        ring_params = {k: posterior_sampled.posterior[k].values[i] for k in posterior_sampled.posterior.data_vars}
        print(ring_params, 'one')
        if first:
            labelling = "100 draws from posterior"
            first = False
        else:
            labelling = "_nolegend_"
        plot_ring(ax, ring_params, diameter_circlipse, label=labelling, color='green', alpha=0.1)


def plot_sampled_posterior_np(ax, directory, filename, N=100, color='blue', label='label'):
    data = np.load(directory + filename)
    samples = {k: data[k] for k in data.files}
    keys = list(samples.keys())
    n_total = len(samples[keys[0]])
    idx = np.random.choice(n_total, N, replace=False)
    first = True
    for i in idx:
        ring_params = {k: samples[k][i] for k in keys}
        print(ring_params, 'two')
        if first:
            labelling = label
            first = False
        else:
            labelling = "_nolegend_"
        plot_ring(ax, ring_params, diameter_circlipse, label=labelling, color=color, alpha=0.1)

# --------------------------- Running Main -----------------------------------------------

def main(directory):
    print(f"Comparing datasets...")
    
    file1 = 'slice_a0.99i15.npy'
    #file2 = 'slice_a0.99i15.npy'
    #file3 = 'slice_a0.99i30.npy'
    
    obs1 = np.load(directory + file1)
    #obs2 = np.load(directory + file2)
    #obs3 = np.load(directory + file3)

    data1 = sort_data(obs1)
    #data2 = sort_data(obs2)
    #data3 = sort_data(obs3)


    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    """Plotting different SgrA* type blackholes"""
    # plot_error_bar(ax, data1, label='Slice method set 1 (10K)', color='Black')
    # plot_error_bar(ax, data2, label='kgeo: $a=0.99, i=15$ [deg]', color='blue')
    # plot_error_bar(ax, data3, label='kgeo: $a=0.99, i=30$ [deg]', color='red')
    # plot_kerr_boundary(ax, 15, 0.01, 'black')
    plot_kerr_boundary(ax, 15, 0.01, 'green')
    plot_kerr_boundary(ax, 15, 0.5, 'orange')
    plot_kerr_boundary(ax, 15, 0.99, 'blue')
    

    ellipse_params = {
    'beta0': 1.0,
    'R0': 51 * eh.RADPERUAS / 2 * 1e9,
    'w': 1 * eh.RADPERUAS * 1e9,
    'a1': 0,
    'b1': 0.33j,
    'Rf': 1.05,
    'psi': 0
    }

    circlipse_params = load_circlipse_dictionary(directory="mcmc_cache/", 
                                                 filename="circlipse_dict.json")
    
    #plot_band(ax, circlipse_params, diameter_ellipse, label=f'90$\%$ HPDI band', color='green')
    #plot_ring(ax, ellipse_params, diameter_ellipse, label='Ground truth', color='black')
    #plot_ring(ax, circlipse_params["mean"], diameter_circlipse, label='Median extended method', color='black')

    # plot_sampled_posterior(ax, 
    #                        directory='/net/vdesk/data2/tmayer/svlbisim/mcmc_cache/', 
    #                        filename='mcmc_run001_posterior_sampled.nc'
    #                        )
    
    # plot_sampled_posterior_np(ax, 
    #                           directory='/net/vdesk/data2/tmayer/svlbisim/mcmc_cache/', 
    #                           filename='mcmc_run_a0.99i15_50K_samples.npz',
    #                           color='green',
    #                           label='100 draws from posterior 50K')

    # plot_sampled_posterior_np(ax, 
    #                         directory='/net/vdesk/data2/tmayer/svlbisim/mcmc_cache/', 
    #                         filename='mcmc_run_a0.99i15_100K_samples.npz',
    #                         label='100 draws from posterior 100K')

    plot_posterior_band(ax, 
                directory='/net/vdesk/data2/tmayer/svlbisim/mcmc_cache/', 
                filename='mcmc_run_a0.01i15_100K_samples.npz',
                ring_function=diameter_circlipse,
                label='95$\%$ credible interval set 2 (100K)',
                color='green')
    
    plot_posterior_band(ax, 
                directory='/net/vdesk/data2/tmayer/svlbisim/mcmc_cache/', 
                filename='mcmc_run_a0.5i15_100K_samples.npz',
                ring_function=diameter_circlipse,
                label='95$\%$ credible interval set 2 (100K)',
                color='orange')

    plot_posterior_band(ax, 
                directory='/net/vdesk/data2/tmayer/svlbisim/mcmc_cache/', 
                filename='mcmc_run_a0.99i15_100K_samples.npz',
                ring_function=diameter_circlipse,
                label='95$\%$ credible interval set 2 (100K)',
                color='blue')


    # General plotting configuration
    ax.set_xlabel('Projection angle $\phi$ in [rad]', fontsize=10)
    ax.set_ylabel(f'Projected diameter $d_\phi$ in $\mu$as', fontsize=10)
    # ax.set_title('Diameter of $n=1$ ring as function $\phi$, and critical curve C$_\gamma$', fontsize=14)
    ax.set_title('Projected diameter as function $\phi$', fontsize=14)
    #ax.tick_params(direction='in', top=True, right=True)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,              # 3 columns → 2 rows (3 + 3)
        frameon=False
    )
    plt.tight_layout()
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    if save_figure:
        plt.savefig(directory + 'results_full_method_layers.pdf', dpi=300, bbox_inches='tight')

    plt.show()
    return 0

main(directory)