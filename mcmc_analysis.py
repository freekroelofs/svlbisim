import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import ehtim as eh
import arviz as az
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import svlbisim_input_reader
mcmc_infile = sys.argv[1]
fit_params = svlbisim_input_reader.load_yaml(mcmc_infile)


make_sample_posterior = False
make_circlipse_dictionary = False
make_trace_plot = True
make_corner_plot = False
save = False

def make_circlipse_dictionary(summary, outdir="mcmc_cache/", filename='circlipse_dict'):
    dictionary = {"mean": {},
                  "upper": {},
                  "lower": {}
    }

    for variable in summary.index:
        dictionary["mean"][variable] = float(summary.loc[variable, "mean"])
        dictionary["upper"][variable] = float(summary.loc[variable, "hdi_82.5%"]) 
        dictionary["lower"][variable] = float(summary.loc[variable, "hdi_17.5%"])

    print(dictionary)

    os.makedirs(outdir, exist_ok=True)  # Ref AI
    filepath = os.path.join(outdir, filename + ".json")
    with open(filepath, "w") as f:
        json.dump(dictionary, f, indent=2)

    return dictionary 


def remove_burnin(idata, N_remove):
    idata_cut = idata.sel(draw=slice(N_remove, None))
    return idata_cut


def draw_from_posterior(idata, N_draws):
    posterior = idata.posterior
    posterior_flattened = posterior.stack(sample=("chain", "draw"))
    posterior_flattened = posterior_flattened.reset_index("sample")
    samples = np.random.choice(posterior_flattened.sample.size, N_draws, replace=False)
    posterior_sampled = posterior_flattened.isel(sample=samples)
    return posterior_sampled


def main(directory = "mcmc_cache/", filename="mcmc_run_a0.99i30_set2"):
    idata = az.from_netcdf(directory + filename + "_idata.nc")
    idata_cut = remove_burnin(idata, N_remove=2500)
    summary = az.summary(idata_cut, round_to=6, hdi_prob=0.65)
    print(summary)

    if make_sample_posterior:
        posterior_sampled = draw_from_posterior(idata_cut, N_draws=100)
        print(posterior_sampled)
        if save:
            az.to_netcdf(posterior_sampled, directory + filename + "_posterior_sampled.nc")

    if make_circlipse_dictionary:
        make_circlipse_dictionary(summary)
    
    if make_trace_plot:
        # vars_ = ["R0", "Rf", "psi", "beta0", "a1", "b1", "w"]
        az.plot_trace(
            idata_cut,
            # var_names=vars_,
            # lines=[
            #     ("R0", {}, fit_params["R0"] * 1e9 * eh.RADPERUAS),
            #     ("Rf", {}, fit_params["Rf"]),
            #     ("psi", {}, fit_params["psi"]),
            #     ("beta0", {}, fit_params["beta0"]),
            #     ("a1", {}, fit_params["a1"]),
            #     ("b1", {}, fit_params["b1"]),
            #     ("w", {}, fit_params["w"] * 1e9 * eh.RADPERUAS),
            # ],
            # figsize=(10, 12),
            # compact=True
        )
        plt.tight_layout()
        if save:
            plt.savefig(directory + filename + 'trace.pdf')
        plt.show()

    if make_corner_plot:
        reference_dict = {
            "R0": fit_params["R0"] * 1e9 * eh.RADPERUAS,
            "Rf": fit_params["Rf"],
            "psi": fit_params["psi"],
            "beta0" : fit_params["beta0"],
        }

        az.plot_pair(
            idata_cut,
            var_names=vars,
            kind="hexbin",
            marginals=True,
            reference_values=reference_dict,
            reference_values_kwargs={
                "color": "red",
                "linestyle": "--",
                "linewidth": 1.0,
            },
            figsize=(16, 12),   # BIGGER PLOT
        )

        plt.tight_layout()

        # ---- make fonts smaller ----
        plt.gcf().axes
        for ax in plt.gcf().axes:
            ax.tick_params(labelsize=10)
            ax.xaxis.label.set_size(10)
            ax.yaxis.label.set_size(10)
            ax.title.set_size(14)
        
        if save:   
            plt.savefig(directory + filename + 'corner.pdf')
        plt.show()

main()