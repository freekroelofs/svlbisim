import ehtim as eh
import numpy as np
import os
from svlbisim_genuvcov import calc_positions
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import matplotlib.image as image

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    
    return artists

def animate_orbits_uvcov(uvfile, radius1, radius2, inc, pa, tstop, out, radius_earth = 6378., nframes = 5000):

    uvdata = eh.obsdata.load_uvfits(uvfile)

    t_orig = uvdata.data['time']*3600
    times = np.linspace(t_orig[0], tstop, num=nframes)
    positions1 = calc_positions(radius1, inc, pa, times)
    x1 = positions1[0]
    y1 = positions1[1]
    z1 = positions1[2]
    positions2 = calc_positions(radius2, inc, pa, times)
    x2 = positions2[0]
    y2 = positions2[1]
    z2 = positions2[2]

    us = np.zeros(len(times))
    vs = np.zeros(len(times))
    for i in range(len(times)):
        index = np.abs(t_orig-times[i]).argmin()
        us[i] = uvdata.data['u'][index]/1e9
        vs[i] = uvdata.data['v'][index]/1e9

    maxuvdist = np.max(np.sqrt(us**2+vs**2))

    for i in range(len(times)):

        fig=plt.figure()
        ax1=fig.add_subplot(121)
        ax2=fig.add_subplot(122)

        stars_image_path = os.path.join(os.path.dirname(__file__)) + '/img/stars.png'
        im_earth = image.imread(stars_image_path)
        ax1.imshow(im_earth, extent=(-1.25*radius2, 1.25*radius2, -1.25*radius2, 1.25*radius2), zorder=-1)

        ax1.plot(x1, y1, color= 'grey', linewidth=0.1, zorder=3)
        ax1.plot(x2, y2, color= 'grey', linewidth=0.1, zorder=3)

        if np.min(np.abs(t_orig-times[i])) <= 1.5*(times[1]-times[0]):
            ax1.plot([x1[i], x2[i]], [y1[i], y2[i]], color='r', zorder=3)
        

        sat_image_path = os.path.join(os.path.dirname(__file__)) + '/img/satellite.png'
        imscatter(x1[i], y1[i], sat_image_path, zoom=0.03, ax=ax1)
        imscatter(x2[i], y2[i], sat_image_path, zoom=0.03, ax=ax1)

        earth_image_path = os.path.join(os.path.dirname(__file__)) + '/img/earth.png'
        im_earth = image.imread(earth_image_path)
        ax1.imshow(im_earth, extent=(-radius_earth, radius_earth, -radius_earth, radius_earth), zorder=4)


        ax2.scatter(us[:i], vs[:i], s=1, color='b')
        ax2.scatter(-us[:i], -vs[:i], s=1, color='b')
        if np.min(np.abs(t_orig-times[i])) <= 1.5*(times[1]-times[0]):
            ax2.scatter(us[i], vs[i], s=1, color='r')
            ax2.scatter(-us[i], -vs[i], s=1, color='r')
        else:
            ax2.scatter(us[i], vs[i], s=1, color='b')
            ax2.scatter(-us[i], -vs[i], s=1, color='b')

        ax1.set_xlim(-1.25*radius2, 1.25*radius2)
        ax1.set_ylim(-1.25*radius2, 1.25*radius2)
        ax2.set_xlim(-1.1*maxuvdist, 1.1*maxuvdist)
        ax2.set_ylim(-1.1*maxuvdist, 1.1*maxuvdist)
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')

        ax1.set_xlabel(r'$x$ (km)', fontsize=12)
        ax1.set_ylabel(r'$y$ (km)', fontsize=12)
        ax2.set_xlabel(r'$u$ (G$\lambda$)', fontsize=12)
        ax2.set_ylabel(r'$v$ (G$\lambda$)', fontsize=12)
        ax2.invert_xaxis()
        plt.suptitle('$t=$%s h' %(round(times[i]/3600, 1)), y=0.8)
        plt.tight_layout()
        plt.savefig(out + 'frame%05d.png'%i, bbox_inches='tight', dpi=400)
        plt.close()

    return 0

def animate_orbits_uvcov_3sat(uvfile, radius1, radius2, radius3, inc, pa, tstop, out, radius_earth = 6378., nframes = 5000):

    uvdata = eh.obsdata.load_uvfits(uvfile)
    t_orig = uvdata.data['time']*3600
    times = np.linspace(t_orig[0], tstop, num=nframes)

    uvdata12 = uvdata.flag_sites(['SAT3']).data
    t_orig12 = uvdata12['time']*3600

    uvdata23 = uvdata.flag_sites(['SAT1']).data
    t_orig23 = uvdata23['time']*3600

    uvdata31 = uvdata.flag_sites(['SAT2']).data
    t_orig31 = uvdata31['time']*3600
              
    positions1 = calc_positions(radius1, inc, pa, times)
    x1 = positions1[0]
    y1 = positions1[1]
    z1 = positions1[2]
    positions2 = calc_positions(radius2, inc, pa, times)
    x2 = positions2[0]
    y2 = positions2[1]
    z2 = positions2[2]
    positions3 = calc_positions(radius3, inc, pa, times)
    x3 = positions3[0]
    y3 = positions3[1]
    z3 = positions3[2]
    
    us12 = np.zeros(len(times))
    vs12 = np.zeros(len(times))
    for i in range(len(times)):
        index = np.abs(t_orig12-times[i]).argmin()
        us12[i] = uvdata12['u'][index]/1e9
        vs12[i] = uvdata12['v'][index]/1e9

    us23 = np.zeros(len(times))
    vs23 = np.zeros(len(times))
    for i in range(len(times)):
        index = np.abs(t_orig23-times[i]).argmin()
        us23[i] = uvdata23['u'][index]/1e9
        vs23[i] = uvdata23['v'][index]/1e9

    us31 = np.zeros(len(times))
    vs31 = np.zeros(len(times))
    for i in range(len(times)):
        index = np.abs(t_orig31-times[i]).argmin()
        us31[i] = uvdata31['u'][index]/1e9
        vs31[i] = uvdata31['v'][index]/1e9

    maxuvdist = np.max(np.sqrt(us12**2+vs12**2))

    for i in range(len(times)):

        fig=plt.figure()
        ax1=fig.add_subplot(121)
        ax2=fig.add_subplot(122)

        stars_image_path = os.path.join(os.path.dirname(__file__)) + '/img/stars.png'
        im_earth = image.imread(stars_image_path)
        ax1.imshow(im_earth, extent=(-1.25*radius2, 1.25*radius2, -1.25*radius2, 1.25*radius2), zorder=-1)

        ax1.plot(x1, y1, color= 'grey', linewidth=0.1, zorder=3)
        ax1.plot(x2, y2, color= 'grey', linewidth=0.1, zorder=3)
        ax1.plot(x3, y3, color= 'grey', linewidth=0.1, zorder=3)

        if np.min(np.abs(t_orig12-times[i])) <= 3*(times[1]-times[0]):
            ax1.plot([x1[i], x2[i]], [y1[i], y2[i]], color='lightblue', zorder=3)
            
        if np.min(np.abs(t_orig23-times[i])) <= 3*(times[1]-times[0]):
            ax1.plot([x2[i], x3[i]], [y2[i], y3[i]], color='darkorange', zorder=3)

        if np.min(np.abs(t_orig31-times[i])) <= 3*(times[1]-times[0]):
            ax1.plot([x3[i], x1[i]], [y3[i], y1[i]], color='lightgreen', zorder=3)
        

        sat_image_path = os.path.join(os.path.dirname(__file__)) + '/img/satellite.png'
        imscatter(x1[i], y1[i], sat_image_path, zoom=0.03, ax=ax1)
        imscatter(x2[i], y2[i], sat_image_path, zoom=0.03, ax=ax1)
        imscatter(x3[i], y3[i], sat_image_path, zoom=0.03, ax=ax1)

        earth_image_path = os.path.join(os.path.dirname(__file__)) + '/img/earth.png'
        im_earth = image.imread(earth_image_path)
        ax1.imshow(im_earth, extent=(-radius_earth, radius_earth, -radius_earth, radius_earth), zorder=4)


        s = 0.5

        ax2.scatter(us31[:i], vs31[:i], s=s, color='green')
        ax2.scatter(-us31[:i], -vs31[:i], s=s, color='green')
        if np.min(np.abs(t_orig31-times[i])) <= 3*(times[1]-times[0]):
            ax2.scatter(us31[i], vs31[i], s=s, color='lightgreen')
            ax2.scatter(-us31[i], -vs31[i], s=s, color='lightgreen')
        else:
            ax2.scatter(us31[i], vs31[i], s=s, color='green')
            ax2.scatter(-us31[i], -vs31[i], s=s, color='green')

        ax2.scatter(us23[:i], vs23[:i], s=s, color='crimson')
        ax2.scatter(-us23[:i], -vs23[:i], s=s, color='crimson')
        if np.min(np.abs(t_orig23-times[i])) <= 3*(times[1]-times[0]):
            ax2.scatter(us23[i], vs23[i], s=s, color='darkorange')
            ax2.scatter(-us23[i], -vs23[i], s=s, color='darkorange')
        else:
            ax2.scatter(us23[i], vs23[i], s=s, color='crimson')
            ax2.scatter(-us23[i], -vs23[i], s=s, color='crimson')
            
        ax2.scatter(us12[:i], vs12[:i], s=s, color='b')
        ax2.scatter(-us12[:i], -vs12[:i], s=s, color='b')
        if np.min(np.abs(t_orig12-times[i])) <= 3*(times[1]-times[0]):
            ax2.scatter(us12[i], vs12[i], s=s, color='lightblue')
            ax2.scatter(-us12[i], -vs12[i], s=s, color='lightblue')
        else:
            ax2.scatter(us12[i], vs12[i], s=s, color='b')
            ax2.scatter(-us12[i], -vs12[i], s=s, color='b')



        ax1.set_xlim(-1.25*radius2, 1.25*radius2)
        ax1.set_ylim(-1.25*radius2, 1.25*radius2)
        ax2.set_xlim(-1.1*maxuvdist, 1.1*maxuvdist)
        ax2.set_ylim(-1.1*maxuvdist, 1.1*maxuvdist)
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')

        ax1.set_xlabel(r'$x$ (km)', fontsize=12)
        ax1.set_ylabel(r'$y$ (km)', fontsize=12)
        ax2.set_xlabel(r'$u$ (G$\lambda$)', fontsize=12)
        ax2.set_ylabel(r'$v$ (G$\lambda$)', fontsize=12)
        ax2.invert_xaxis()
        plt.suptitle('$t=$%s h' %(round(times[i]/3600, 1)), y=0.8)
        plt.tight_layout()
        plt.savefig(out + 'frame%05d.png'%i, bbox_inches='tight', dpi=400)
        plt.close()

    return 0

    
def make_gif(out):
    filestr = out + 'frame' + '%05d'
    outstr = out + 'orbits_uvcov'                                            

    os.system('ffmpeg -f image2 -y -framerate 40 -r 50 -y -i %s.png -vcodec libx264 -pix_fmt yuv420p -crf 22 %s.mp4'%(filestr, outstr))

    return 0
    
def main(params):

    out = params['outdir'] + '/' + params['outtag']
    uvfile = out + '_uvcoords.uvfits'
    radius1 = params['radius1']
    radius3 = params['radius1'] + params['delta_r']
    nsat = int(params['nsat'])
    if nsat == 3:
        radius2 = radius1 + (radius3-radius1)/3
    inc = params['inclination']*np.pi/180
    pa = params['positionangle']*np.pi/180
    tstop = params['tstop']*24*3600
    
    out = params['outdir'] + '/animation_' + params['outtag'] + '/'
    if not os.path.exists(out):
        os.makedirs(out)

    if nsat == 2:
        animate_orbits_uvcov(uvfile, radius1, radius3, inc, pa, tstop, out)
    elif nsat == 3:
        animate_orbits_uvcov_3sat(uvfile, radius1, radius2, radius3, inc, pa, tstop, out)

    make_gif(out)

    return 0




