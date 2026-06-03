"""Code to perform a Bayesian fit of visibility amplitude data to extract diameter of a ring"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import os
import sys
import ehtim as eh
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import svlbisim_input_reader
infile = sys.argv[1]
params = svlbisim_input_reader.load_yaml(infile)

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, ESS
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import arviz as az
rng_key = random.PRNGKey(0)
from jax.scipy.special import i0


# Defining some parameters for pyro to run on
N_chains = 12
N_warmup = 3000
N_samples = 14000
eps = 1e-6

os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count=12'


class Rice(numpyro.distributions.Distribution):     #Ref: AI generated
    support = numpyro.distributions.constraints.positive

    def __init__(self, nu, sigma):
        self.nu = nu
        self.sigma = sigma
        batch_shape = jnp.broadcast_shapes(jnp.shape(nu), jnp.shape(sigma))
        super().__init__(batch_shape=batch_shape, event_shape=())

    def log_prob(self, value):
        nu = self.nu
        sigma = self.sigma

        return (
            jnp.log(value)
            - 2 * jnp.log(sigma)
            - (value**2 + nu**2) / (2 * sigma**2)
            + jnp.log(i0(value * nu / sigma**2))
        )


# Defining functions 
def simple_ring_model(u, d, a_min, a_plus, w):
    numerator = a_plus**2 + a_min**2 + 2*a_plus*a_min*jnp.sin(2*jnp.pi*d*u)
    numerator = jnp.clip(numerator, a_min=eps)  # Not the same a_min as the parameter!! see np documentation!
    model = jnp.sqrt(numerator / (u + eps)) * jnp.exp(-u * w)
    return model


def model(u, vis=None, vis_sigma=None):
    d = numpyro.sample('d', dist.Uniform(0.23, 0.26))
    a_min = numpyro.sample("lower", dist.Uniform(0.0001, 1))
    a_plus = numpyro.sample("upper", dist.Uniform(0.0001, 1))
    w = numpyro.sample("w", dist.Exponential(10))

    model_vis = simple_ring_model(u, d, a_min, a_plus, w)

    # # Important! We need this extra noise term for the mcmc not to get stuck!
    sigma_extra = numpyro.sample("sigma_extra", dist.HalfNormal(0.001))
    sigma_total = jnp.sqrt(vis_sigma**2 + sigma_extra**2)
    
    numpyro.sample('obs', Rice(model_vis, sigma_total), obs=vis)
    return 0


def plot_data_and_model(data, best_fit_params):
    fig, ax = plt.subplots()

    d, a_min, a_plus, w = best_fit_params
    u_min = min(data[:, 0])
    u_max = max(data[:, 0])
    u_range = np.linspace(u_min, u_max, 1000)

    print(best_fit_params, 'THESE INPUTS!')

    model_vis_amp = simple_ring_model(u_range, d, a_min, a_plus, w)
    
    ax.scatter(data[:, 0], data[:, 1], marker='.', s=3, c='orange', label='Data')
    ax.plot(u_range, model_vis_amp, label='MCMC fit from median parameters')

    ax.set_xlabel('$uv$-distance (G$\lambda$)', fontsize=10)
    ax.set_ylabel('Visibility Amplitude (Jy)', fontsize=10)
    ax.set_yscale('log')

    ax.axvline(20, color='grey', linestyle='--', linewidth=1)
    ax.legend()

    ax.set_title(f'Visibility amplitude with separated data and model with diameter {round(d/10**9 * 1/eh.RADPERUAS)} $\mu m$')
    plt.show()

    return 0


def pyro_main(data, show_summary=True, show_fit=False, show_walker_corner=False, save_model_render=False):
    if save_model_render:
        # Render a model of the parameters
        numpyro.render_model(model, render_params=True, render_distributions=True, 
                             model_args=(data[:,0], ), 
                             model_kwargs={'vis': data[:,1], 'vis_sigma': data[:,2]}, 
                             filename="model_ESS.pdf")
    
    # Run NUTS
    kernel = ESS(model)
    mcmc = MCMC(kernel, num_warmup=N_warmup, num_samples=N_samples, num_chains=N_chains, chain_method='vectorized')
    mcmc.run(rng_key, u=data[:,0], vis=data[:,1], vis_sigma=data[:,2])
    samples = mcmc.get_samples()
    print(samples["d"])
    # Getting the median values
    d_median = jnp.median(samples["d"])
    lower_median = jnp.median(samples["lower"])
    upper_median = jnp.median(samples["upper"])
    w_median = jnp.median(samples["w"])
    best_fit_params = (d_median, lower_median, upper_median, w_median)

    # Getting the highest posterior density interval
    d_hpdi = hpdi(samples["d"], prob=0.95)
    d_minus = d_hpdi[0]
    d_plus  = d_hpdi[1]
    best_d = (d_minus, d_median, d_plus)
    
    if show_summary:
        mcmc.print_summary()

    if show_fit:
        plot_data_and_model(data, best_fit_params=best_fit_params)

    if show_walker_corner:
        idata = az.from_numpyro(mcmc)
        print(idata)
        az.plot_trace(idata)
        plt.show()
        az.plot_pair(idata, var_names=["d", "upper", "lower", "w"], kind="hexbin", marginals=True)
        plt.show()

    print(f"found best diameter {d_median/10**9 * 1/eh.RADPERUAS}!!")
    return d_median, d_hpdi


def directional_sort_uvdata(obs, angle, dangle, plot_uv_plane=False):
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
        ax.set_xlabel('$u$ coord. in (G$\lambda$)', fontsize=10)
        ax.set_ylabel('$v$ coord. in (G$\lambda$)', fontsize=10)
        ax.set_title(f'$uv$-plane', fontsize=14)
        ax.scatter(obs.data['u']/1e9, obs.data['v']/1e9, s=0.1, color='grey')
        ax.scatter(u_sorted/1e9, v_sorted/1e9, s=0.5, color='orange')
        plt.show()
    
    uvdist_amp_sigma_sorted = np.stack((uvdist, amp, sigma), axis=1)
    return uvdist_amp_sigma_sorted 


def angled_uvdist_amp(obs, angle, dangle=np.pi/16):
    sorted_data = directional_sort_uvdata(obs, angle, dangle)
    return sorted_data


def plot_diameter_angles(phi_array, d_array, d_hpdi_array):
    fig, ax = plt.subplots()

    # Convert to uas
    d_array = d_array/10**9 * 1/eh.RADPERUAS
    d_hpdi_array = d_hpdi_array/10**9 * 1/eh.RADPERUAS

    # Plotting
    ax.scatter(phi_array, d_array)
    ax.fill_between(phi_array, d_hpdi_array[:, 0], d_hpdi_array[:, 1], alpha=0.3)

    ax.set_xlabel('Projection angle $\phi$ in [rad]', fontsize=10)
    ax.set_ylabel(f'Projected Ddiameter $d_\phi$ in $\mu$as', fontsize=10)
    plt.show()


# --------------------------- Running Main -----------------------------------------------
def main(params, angles, u_domain, run_pyro=True):
    print(f"Starting mcmc code...")
    d_array = []
    d_hdpi_array = []

    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)

    print("Got data")
    print(f"Running mcmc code for angles {angles}...")

    u_min, u_max = u_domain
    for angle in angles:
        print(f"Running for angle {angle}...")
        
        data_phi = angled_uvdist_amp(obs, angle)
        mask = (data_phi[:, 0] >= u_min) & (data_phi[:, 0] <= u_max)
        data_phi_in_domain = data_phi[mask]
        mask_Nan = (data_phi_in_domain[:, 0] != u_min) & (data_phi_in_domain[:, 1] != u_max) & (data_phi_in_domain[:, 2] != u_max)
        data_phi_in_domain = data_phi_in_domain[mask_Nan]


        if run_pyro:
            print(f"Running Numpyro!")
            diameter, d_hdpi = pyro_main(jnp.array(data_phi_in_domain), show_fit=False, show_walker_corner=False, save_model_render=False)
            d_array.append(diameter)
            d_hdpi_array.append(d_hdpi)
    
    d_hdpi_array = np.array(d_hdpi_array)
    d_array = np.array(d_array)
    
    results = np.column_stack((d_array, d_hdpi_array))
    np.save('/net/vdesk/data2/tmayer/svlbisim/SgrA_kgeo_runs/slice_method_results/slice_a0.99i30_100K_test', results)

    print('fit_result', results)
    plot_diameter_angles(angles, d_array, d_hdpi_array)
    
    return 0


angles=np.linspace(0, np.pi, 30)
#angles=np.array([0])
u_domain = (27, 37.5)

main(params, angles, u_domain)