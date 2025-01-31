import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import os

def calc_positions(radius, inc, pa, times, GM=3.986004418e14):
    omega = np.sqrt(GM / (1000. * radius)) / (1000. * radius)
    positions = np.array([np.cos(times * omega) * radius, np.sin(times * omega) * radius, np.zeros(len(times))])
    temp = np.array([np.cos(inc) * positions[0] - np.sin(inc) * positions[2], positions[1], np.cos(inc) * positions[2] + np.sin(inc) * positions[0]])
    positions = np.array([np.cos(pa) * temp[0] - np.sin(pa) * temp[1], np.cos(pa) * temp[1] + np.sin(pa) * temp[0], temp[2]])

    return positions

def mask_earthshadow(positions, radius_earth=6378.):
    # Take care of Earth's shadow being in the way: mask all positions where this is the case.
    # Condition: if z < 0 and sqrt(x^2 + y^2) is smaller than radius_earth.
    mask_1 = np.sqrt(positions[0]**2. + positions[1]**2.) < radius_earth
    mask_2 = positions[2] < 0.
    mask = ~(mask_1 * mask_2)

    return mask

def calc_uv(r1, r2, inc, pa, timerange, nu, fov, GM=3.986004418e14, radius_earth=6378.):
    
    # Find timestamps using uv-smearing limit
    times = np.array([0.])
    finaltimes = [0.]
    positions1=[]
    positions2=[]

    while times[0] < timerange[1]:
        positions1 = calc_positions(r1, inc, pa, times)
        positions2 = calc_positions(r2, inc, pa, times)

        # Calculate uv coordinates
        us12 = (positions1[0] - positions2[0]) * 1000. / (3e8/nu)
        vs12 = (positions1[1] - positions2[1]) * 1000. / (3e8/nu)

        # Calculate tint using uv smearing limit
        omega = np.sqrt(GM / (1000. * r1)) / (1000. * r1)
        period = (2. * np.pi) / omega       
        baseline = np.sqrt(us12**2+vs12**2)        
        tint = period/(2*np.pi*fov*baseline)

        # Have at least 10 points per orbit (for short baselines)        
        if tint > period/36.:
            tint = period/36.
        times[0] += tint
        finaltimes.append(times[0])
        
    # Set integration times
    integrationtimes = np.zeros(len(finaltimes))
    for i in range(1,len(finaltimes)):
        integrationtimes[i] = finaltimes[i]-finaltimes[i-1]
    integrationtimes[0]=integrationtimes[1]
    times = np.array(finaltimes)

    # Calculate positions
    positions1 = calc_positions(r1, inc, pa, times)
    positions2 = calc_positions(r2, inc, pa, times)

    # Mask out Earth Shadow
    mask1 = mask_earthshadow(positions1)
    mask2 = mask_earthshadow(positions2)
    mask12 = (mask1 * mask2)
    positions12 = np.array([positions1[0][mask12], positions1[1][mask12], positions1[2][mask12]])
    positions21 = np.array([positions2[0][mask12], positions2[1][mask12], positions2[2][mask12]])
    times12 = times[mask12]
    integrationtimes12 = integrationtimes[mask12]
    
    # Mask out ISL occlusion
    rcrit = np.sqrt(r1**2. - radius_earth**2.) + np.sqrt(r2**2. - radius_earth**2.)
    mask12 = np.sqrt((positions12[0] - positions21[0])**2. + (positions12[1] - positions21[1])**2. + (positions12[2] - positions21[2])**2.) < rcrit
    positions12 = np.array([positions12[0][mask12], positions12[1][mask12], positions12[2][mask12]])
    positions21 = np.array([positions21[0][mask12], positions21[1][mask12], positions21[2][mask12]])
    times12 = times12[mask12]
    integrationtimes12 = integrationtimes12[mask12]

    # Calculate final uv coordinates
    us12 = (positions12[0] - positions21[0]) * 1000. / (3e8/nu)
    vs12 = (positions12[1] - positions21[1]) * 1000. / (3e8/nu)

    uvdata = np.array([us12, vs12, times12, integrationtimes12])
    
    return uvdata

def export_uvfits(uvdata, nsat, source, ra, dec, nu, bw, out):
    if nsat == 2:
        uvdata12 = uvdata
    elif nsat == 3:
        uvdata12 = uvdata[0]
        uvdata23 = uvdata[1]
        uvdata31 = uvdata[2]
        
    us12 = uvdata12[0]
    vs12 = uvdata12[1]
    times12 = uvdata12[2]
    integrationtimes12 = uvdata12[3]
    
    if nsat == 3:
        us23 = uvdata23[0]
        vs23 = uvdata23[1]
        times23 = uvdata23[2]
        integrationtimes23 = uvdata23[3]
        us31 = uvdata31[0]
        vs31 = uvdata31[1]
        times31 = uvdata31[2]
        integrationtimes31 = uvdata31[3]
    
    with open(out + '_uvcoords.txt' ,'w') as f:
        print('# SRC: %s'%source, file=f) 
        print('# RA: %s'%ra, file=f)
        print('# DEC: %s'%dec, file=f)
        print('# MJD: 0', file=f)
        print('# RF: %s.0000 GHz'%int(nu/1e9), file=f)
        print('# BW: %s GHz'%bw, file=f)
        print('# PHASECAL: 1', file=f)
        print('# AMPCAL: 1', file=f)
        print('# OPACITYCAL: 1', file=f)
        print('# DCAL: 1', file=f)
        print('# FRCAL: 1', file=f)
        print('# ----------------------------------------------------------------------------------------------------------------------------------------', file=f)
        print('# Site       X(m)             Y(m)             Z(m)           SEFDR      SEFDL     FR_PAR   FR_EL   FR_OFF  DR_RE    DR_IM    DL_RE    DL_IM', file=f)
        print('# SAT1       0.00             0.00             0.00           0.00       0.00      0.00     0.00    0.00    0.00     0.00     0.00     0.00', file=f) 
        print('# SAT2       0.00             0.00             0.00           0.00       0.00      0.00     0.00    0.00    0.00     0.00     0.00     0.00', file=f)
        if nsat == 3:
            print('# SAT3       0.00             0.00             0.00           0.00       0.00      0.00     0.00    0.00    0.00     0.00     0.00     0.00', file=f)
        print('# ----------------------------------------------------------------------------------------------------------------------------------------', file=f)
        print('# time (hr) tint    T1     T2    Tau1   Tau2   U (lambda)       V (lambda)         Iamp (Jy)    Iphase(d)  Qamp (Jy)    Qphase(d)   Uamp (Jy)    Uphase(d)   Vamp (Jy)    Vphase(d)   Isigma (Jy)   Qsigma (Jy)   Usigma (Jy)   Vsigma (Jy)', file=f)

        for i in range(0, len(us12)):
            print('%f %f SAT1 SAT2 0. 0. %f %f 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.'%(times12[i]/3600., integrationtimes12[i], us12[i], vs12[i]), file=f)
        if nsat == 3:
            for i in range(0, len(us23)):
                print('%f %f SAT2 SAT3 0. 0. %f %f 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.'%(times23[i]/3600., integrationtimes23[i], us23[i], vs23[i]), file=f)
            for i in range(0, len(us31)):
                print('%f %f SAT3 SAT1 0. 0. %f %f 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.'%(times31[i]/3600., integrationtimes31[i], us31[i], vs31[i]), file=f)
    f.close()
    obs = eh.obsdata.load_txt(out + '_uvcoords.txt')
    obs.save_uvfits(out + '_uvcoords.uvfits')
    os.system('rm %s'%(out + '_uvcoords.txt'))
    #obs.plotall('u','v')
    #plt.show()

    return 0

def main(params):
    # Set inputs
    nsat = params['nsat']
    radius1 = params['radius1']
    radius3 = radius1 + params['delta_r']
    if nsat == 3:
        radius2 = radius1 + (radius3-radius1)/3
    inclination = params['inclination']*np.pi/180 
    positionangle = params['positionangle']*np.pi/180
    timerange = [params['tstart']*3600*24, params['tstop']*3600*24]
    source = params['source']
    ra = params['ra']
    dec = params['dec'] 
    nu = params['nu']
    bw = params['bw']
    fov = params['fov']*eh.RADPERUAS
    out = params['outdir'] + '/' + params['outtag']
    
    # Generate uv coordinates
    uvdata31 = calc_uv(radius3, radius1, inclination, positionangle, timerange, nu, fov)
    if nsat == 3:
        uvdata12 = calc_uv(radius1, radius2, inclination, positionangle, timerange, nu, fov)
        uvdata23 = calc_uv(radius2, radius3, inclination, positionangle, timerange, nu, fov)
        uvdata = [uvdata12, uvdata23, uvdata31]
        export_uvfits(uvdata, nsat, source, ra, dec, nu, bw, out)
    elif params['nsat'] == 2:
        export_uvfits(uvdata31, nsat, source, ra, dec, nu, bw, out)

    return 0
        

