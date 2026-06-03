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


# --------------------------- Running Main -----------------------------------------------
def extract_model_vis_amp(path, angle, u_domain):
    model = eh.image.load_fits(path)

    d_min, d_max = u_domain
    res = 10000
    d = np.linspace(d_min, d_max, res)

    u = d * np.cos(angle)
    v = d * np.sin(angle)
    uv = np.column_stack((u, v))

    vis = model.sample_uv(uv=uv)[0]
    vis_amp = np.abs(vis)

    return d, vis_amp


def intensity_unit_conversion(intensity, pixel_size_UAS):
    # Asummed intensity in Jy!! -> mJy per uas2
    intensity = intensity/10**3 /pixel_size_UAS**2
    return intensity


def extract_model_intensity_profile(path):
    model = eh.image.load_fits(path)
    image_array = model.imarr()
    pixel_size_UAS = model.psize*1/eh.RADPERUAS
    xlen, ylen = np.shape(image_array)

    image_array = intensity_unit_conversion(image_array, pixel_size_UAS)

    if xlen != ylen:
        raise Exception('The image is not symmetric and the center profile may not be the center!')
    
    center = int(ylen/2)
    center_axis_profile = image_array[center]

    pixal_array_in_UAS = np.arange(-center, center, 1)*pixel_size_UAS 

    profile = np.column_stack((pixal_array_in_UAS, center_axis_profile))
    return profile


def vis_amp(path, file_names):
    fig, ax = plt.subplots()
    angle=0
    u_domain = (1*10**9, 200*10**9)

    for name in file_names:
        full_path = path + name
        uv_dist, vis_amp = extract_model_vis_amp(full_path, angle, u_domain)
        ax.plot(uv_dist/10**9, vis_amp, label=f'{name[:3]}')

    ax.set_xlabel('$uv$-distance (G$\lambda$)', fontsize=10)
    ax.set_ylabel('Visibility Amplitude (Jy)', fontsize=10)
    ax.set_yscale('log')

    ax.axvline(20, color='grey', linestyle='--', linewidth=1)
    ax.legend()

    ax.set_title(f'Kgeo model visibility amplitude for different n at $\phi$={angle}')
    plt.show()


def intensity_profile(path, file_names):
    fig, ax = plt.subplots()

    for name in file_names:
        full_path = path + name
        profile = extract_model_intensity_profile(full_path)
        ax.plot(profile[:,0], profile[:,1], label=f'{name[:3]}')

    ax.set_xlabel('size in $\mu as$', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.legend()

    ax.set_title(f'Some profile')
    plt.show()


def main():
    path = '/net/vdesk/data2/tmayer/kgeo_project/m87_folder/vary_n=ring/'
    names = ['n=0/m87_model_n=0.fits',
            'n=1/m87_model_n=1.fits',
            'n>=2/m87_model_n>=2.fits',
            'n=all/m87_model_n=all.fits']

    vis_amp(path, names)
    intensity_profile(path, names)


main()