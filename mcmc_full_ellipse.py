"""Code to perform a Bayesian fit of visibility amplitude data to extract diameter of a ring"""
# Common imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rice
import warnings

# importing related to ehtim 
import os
import sys
import ehtim as eh
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import svlbisim_input_reader
infile = sys.argv[1]
fit_infile = sys.argv[2]
params = svlbisim_input_reader.load_yaml(infile)
fit_params = svlbisim_input_reader.load_yaml(fit_infile)

# specify number of cores
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={fit_params['N_cores']}"

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, ESS

# Jax
import jax
from jax import random
import jax.numpy as jnp
import arviz as az
rng_key = random.PRNGKey(0)
from jax.scipy.special import i0e

# importing priors
R0l, R0u = jnp.array(fit_params["R0_d"]) * 1e9 * eh.RADPERUAS
Rfl, Rfu = jnp.array(fit_params["Rf_d"]) 
psil, psiu = jnp.array(fit_params["psi_d"]) 
beta0l, beta0u = jnp.array(fit_params["beta0_d"]) 
a1l, a1u = jnp.array(fit_params["a1_d"]) 
b1l, b1u = jnp.array(fit_params["b1_d"]) 
wl, wu = jnp.array(fit_params["w_d"]) * 1e9 * eh.RADPERUAS


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

        x = value * nu / sigma**2

        return (
            jnp.log(value)
            - 2 * jnp.log(sigma)
            - (value**2 + nu**2) / (2 * sigma**2)
            + jnp.log(i0e(x)) + jnp.abs(x)
        )

#jax.debug.print("numerator {}", jnp.any(jnp.isnan(numerator)))
#jax.debug.print("model {}", model)

# Defining functions 
def simple_ring_model(u, d, a_min, a_plus, w):
    x = jnp.pi * d * u
    visibility = (a_min * jnp.cos(x) + a_plus * jnp.sin(x)) / jnp.sqrt(d * u)
    vis_amp = jnp.abs(visibility)
    exponent = jnp.exp(-(jnp.pi*w*u)**2) 
    model = vis_amp * exponent
    return model

def stretched_ring_shape(phi, R0, Rf, psi):
    diameter = 2*jnp.sqrt((R0*jnp.cos(phi - psi))**2 + (R0*Rf*jnp.sin(phi - psi))**2)
    return diameter

def c_exponent(phi, m, pm):
    exponent = jnp.exp(1j*m*(phi + jnp.pi/2*(m - 1 + pm)))
    return exponent

def alpha_pm_one_mode(phi, beta0, a1, b1):
    betap1 = a1 + 1j*b1
    betam1 = a1 - 1j*b1
    a_plus = 1/jnp.pi*(beta0 + betap1*c_exponent(phi, m=1, pm=1) + betam1*c_exponent(phi, m=-1, pm=1))
    a_minus = 1/jnp.pi*(beta0 + betap1*c_exponent(phi, m=1, pm=-1) + betam1*c_exponent(phi, m=-1, pm=-1))

    return a_plus, a_minus


def plot_diameter_angles(params, phi_array):
    fig, ax = plt.subplots()

    R0, Rf, psi = params
    d_array = stretched_ring_shape(phi_array, R0, Rf, psi)

    # Convert to uas
    d_array = d_array/10**9 * 1/eh.RADPERUAS

    # Plotting
    ax.scatter(phi_array, d_array)

    ax.set_xlabel('Projection angle $\phi$ in [rad]', fontsize=10)
    ax.set_ylabel(f'Projected Ddiameter $d_\phi$ in $\mu$as', fontsize=10)
    plt.show()


def model(data, vis=None, vis_sigma=None):
    phi_uv = data[:, 0]
    uv_dist = data[:, 1]

    R0 = numpyro.sample('R0', dist.Uniform(R0l, R0u))
    Rf = numpyro.sample('Rf', dist.Uniform(Rfl, Rfu))
    psi = numpyro.sample('psi', dist.Uniform(psil, psiu))

    d_phi = stretched_ring_shape(phi_uv, R0, Rf, psi)
    numpyro.deterministic("d_phi", d_phi)

    beta0 = numpyro.sample("beta0", dist.Uniform(beta0l, beta0u))
    a1 = numpyro.sample("a1", dist.Uniform(a1l, a1u))
    b1 = numpyro.sample("b1", dist.Uniform(b1l, b1u))

    a_plus, a_min = alpha_pm_one_mode(phi_uv, beta0, a1, b1)
    numpyro.deterministic("a_plus", a_plus)
    numpyro.deterministic("a_min", a_min)

    w = numpyro.sample("w", dist.Uniform(wl, wu))
    model_vis = simple_ring_model(uv_dist, d_phi, a_min, a_plus, w)

    sigma_extra = numpyro.sample("sigma_extra", dist.HalfNormal(0.001))
    sigma_total = jnp.sqrt(vis_sigma**2 + sigma_extra**2)

    numpyro.sample('obs', Rice(model_vis, sigma_total), obs=vis)
    return 0


def pyro_main(data, fit_params, save_render=True):
    data_jax = jnp.asarray(data[:, :2]) # ensure jax
    vis_jax = jnp.asarray(data[:, 2])   # ensure jax
    sigma_jax = jnp.asarray(data[:, 3]) # ensure jax

    if jnp.any(sigma_jax < 0.001):
        warnings.warn('The error on the data is smaller than 0.001; this might raise issues!!')

    # Render a graphic of the model
    if save_render:
        print('Saving a render of the model...')
        numpyro.render_model(model, render_params=True, render_distributions=True, 
                             model_args=(data[:, :2], ), 
                             model_kwargs={'vis': data[:, 2], 'vis_sigma': data[:, 3]}, 
                             filename="model.pdf")

    # Run NUTS
    mcmc = MCMC(sampler=ESS(model), 
                num_warmup=fit_params['N_warmup'], 
                num_samples=fit_params['N_samples'], 
                num_chains=fit_params['N_chains'], 
                chain_method=fit_params['chain_method'])
    
    mcmc.run(rng_key, data=data_jax, vis=vis_jax, vis_sigma=sigma_jax)
    return mcmc


def mcmc_analytics(mcmc, fit_params, show_summary=True, show_walker_corner=True):
    samples = mcmc.get_samples()

    # Getting the median values
    R0_median = jnp.median(samples["R0"])
    Rf_median = jnp.median(samples["Rf"])
    psi_median = jnp.median(samples["psi"])
    a1_median = jnp.median(samples["a1"])
    b1_median = jnp.median(samples["b1"])
    w_median = jnp.median(samples["w"])

    best_fit_params = (R0_median, Rf_median, psi_median, a1_median, b1_median, w_median)

    if show_summary:
        mcmc.print_summary()

    if show_walker_corner:
        idata = az.from_numpyro(mcmc)
        az.plot_trace(idata, 
                      var_names=["R0", "Rf", "psi", "beta0", "a1", "b1", "w"],
                      lines=[
                          ("R0", {}, fit_params["R0"]* 1e9 * eh.RADPERUAS),
                          ("Rf", {}, fit_params["Rf"]),
                          ("psi", {}, fit_params["psi"]),
                          ("beta0", {}, fit_params["beta0"]),
                          ("a1", {}, fit_params["a1"]),
                          ("b1", {}, fit_params["b1"]),
                          ("w", {}, fit_params["w"]* 1e9 * eh.RADPERUAS),
                      ]
                      )
        plt.show()
        az.plot_pair(idata, var_names=["R0", "Rf", "psi", "a1", "b1", "w"], kind="hexbin", marginals=True)
        plt.show()
    
    print(f"found best parameters (R0, Rf, psi) = ({R0_median, Rf_median, psi_median})!!")
    return best_fit_params


def directional_sort(obs, angles, dangle, u_domain, plot=True):
    # Selecting the data
    u = obs.data['u']
    v = obs.data['v']
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    phi_uv = np.arctan2(v, u)

    # Masking based on uv-distance
    uv_dist = np.sqrt(u**2 + v**2)/1e9
    uv_min, uv_max = u_domain
    mask = (uv_dist >= uv_min) & (uv_dist <= uv_max)
    
    u = u[mask]
    v = v[mask]
    amp = amp[mask]
    sigma = sigma[mask]
    phi_uv = phi_uv[mask]

    def angle_diff(phi, phi_center):
        return (phi - phi_center + np.pi) % (2*np.pi) - np.pi

    # convert angles to array
    angles = np.asarray(angles)
    dphi_1 = np.abs(angle_diff(phi_uv[None, :], angles[:, None]))
    dphi_2 = np.abs(angle_diff(phi_uv[None, :], (angles + np.pi)[:, None]))

    mask = (dphi_1 < dangle/2) | (dphi_2 < dangle/2)
    angle_idx, point_idx = np.where(mask)

    phi_selected = angles[angle_idx]
    u_sel = u[point_idx]
    v_sel = v[point_idx]
    amp_sel = amp[point_idx]
    sigma_sel = sigma[point_idx]

    uvdist_sel = np.sqrt(u_sel**2 + v_sel**2)/1e9

    # final table: (Nselected, 4)
    result = np.column_stack([
        phi_selected,
        uvdist_sel,
        amp_sel,
        sigma_sel
    ])


    if plot:
        # plotting (optional)
        fig, ax = plt.subplots()
        cmap = plt.cm.hsv
        colors = cmap((phi_selected - angles.min())/(angles.max()-angles.min()+1e-12))

        ax.scatter(u_sel/1e9, v_sel/1e9, s=0.5, color=colors)
        ax.set_aspect('equal')
        ax.set_xlabel('$u$ coord. in (G\\lambda$)')
        ax.set_ylabel('$v$ coord. in (G\\lambda$)')
        ax.set_title('$uv$-plane')

        plt.show()
        
    print(f"Final data shape: {result.shape}")
    return result


def domain_sort(obs, fit_params):
    # Selecting the data
    u = obs.data['u']
    v = obs.data['v']
    amp = np.abs(obs.data['vis'])
    sigma = obs.data['sigma']
    phi_uv = np.arctan2(v, u)

    # Masking based on uv-distance
    uv_dist = np.sqrt(u**2 + v**2)/1e9
    uv_min, uv_max = fit_params['u_domain']
    mask = (uv_dist >= uv_min) & (uv_dist <= uv_max)
    
    u = u[mask]
    v = v[mask]
    amp = amp[mask]
    sigma = sigma[mask]
    phi_uv = phi_uv[mask]
    uv_dist = np.sqrt(u**2 + v**2)/1e9

    # final table: (Nselected, 4)
    result = np.column_stack([phi_uv, uv_dist, amp, sigma])

    # N_keep = 1000
    # if result.shape[0] > N_keep:
    #     idx = np.random.choice(result.shape[0], size=N_keep, replace=False)
    #     result = result[idx]

    print(f'datashape:', result.shape)
    return result


def generate_fake_data(fit_params, plot=True):
    grid = make_uv_grid(fit_params)
    phi = grid[:,0]
    uvdist = grid[:,1]

    R0 = fit_params["R0"] * 1e9 * eh.RADPERUAS
    Rf = fit_params["Rf"]
    psi = fit_params["psi"]
    beta0 = fit_params["beta0"]
    a1 = fit_params["a1"]
    b1 = fit_params["b1"]
    w = fit_params["w"] * 1e9 * eh.RADPERUAS

    phi = phi.flatten()         # ensure 1D
    uvdist = uvdist.flatten()   # ensure 1D

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    d_phi = stretched_ring_shape(phi, R0, Rf, psi)
    a_plus, a_min = alpha_pm_one_mode(phi, beta0, a1, b1)
    true_vis_amp = simple_ring_model(uvdist, d_phi, a_min, a_plus, w)

    def rice_distribution(true_signal, sigma):
        b = true_signal/sigma
        signal = rice.rvs(b, scale=sigma, size=true_signal.shape)
        return signal

    vis_amp = rice_distribution(true_vis_amp, fit_params['noise_sigma'])
    sigma = np.full(len(vis_amp), fit_params['noise_sigma'])

    fake_data = np.column_stack([phi, uvdist, vis_amp, sigma])
    print(f"Final fake data shape: {fake_data.shape}")
    return fake_data


def make_uv_grid(fit_params, plot=True):
    ncells = fit_params['n_cells']
    minbl, maxbl = fit_params['u_domain']
    if ncells % 2 == 0:
        n = int(np.sqrt(ncells))
        u = np.linspace(-maxbl, maxbl, n)
        v = u
        u, v = np.meshgrid(u, v)
        uv_grid = np.sqrt(u**2 + v**2)
    else:
        raise Exception('make_uv_grid expects even number of cells')
    
    phi_grid = np.arctan2(v, u)
    mask = (uv_grid >= minbl) & (uv_grid <= maxbl)
    phi_grid = phi_grid[mask]
    uv_grid = uv_grid[mask]
    grid = np.stack((phi_grid, uv_grid), axis=1)

    if plot:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
        ax.scatter(grid[:,0], grid[:,1])
        plt.show()

    print(f'grid shape:{grid.shape}')
    return grid


def visualize_slice(data, angle, dangle):
    phi = data[:, 0]
    uvdist = data[:, 1]
    vis_amp = data[:, 2]
    sigma = data[:, 3]

    def angle_diff(phi, phi_center):
        return (phi - phi_center + np.pi) % (2*np.pi) - np.pi

    dphi_1 = np.abs(angle_diff(phi, angle))
    dphi_2 = np.abs(angle_diff(phi, angle + np.pi))
    mask = (dphi_1 < dangle/2) | (dphi_2 < dangle/2)

    phi_slice = phi[mask]
    uvdist_slice = uvdist[mask]
    vis_amp_slice = vis_amp[mask]
    sigma_slice = sigma[mask]

    fig, ax = plt.subplots()
    ax.scatter(uvdist_slice, vis_amp_slice)
    ax.set_yscale('log')
    plt.show()


def save_mcmc(mcmc, directory="mcmc_cache/", filename="mcmc_run_ellipse"):      #Ref: AI
    print('Saving samples and arviZ data')

    samples = mcmc.get_samples()
    samples_np = {k: np.array(v) for k, v in samples.items()}
    np.savez(directory + filename + "_samples.npz", **samples_np)

    idata = az.from_numpyro(mcmc)
    az.to_netcdf(idata, directory + filename + "_idata.nc")
    print('Saving completed')


def main(params, fit_params):
    print(f"Starting mcmc code...")
    obsfile = params['outdir'] + '/' + params['outtag'] + '.uvfits'
    obs = eh.obsdata.load_uvfits(obsfile)

    faked_data = generate_fake_data(fit_params, plot=False)

    #data_phi_in_domain = directional_sort(obs, phi_array_circlipse, dphi, u_domain)
    data_phi_in_domain = domain_sort(obs, fit_params)

    print(f"Running mcmc...")
    visualize_slice(data_phi_in_domain, angle=0, dangle=np.pi/16)
    mcmc = pyro_main(data_phi_in_domain, fit_params)
    save_mcmc(mcmc)
    best_fit_params = mcmc_analytics(mcmc, fit_params)

    return 0


main(params, fit_params)