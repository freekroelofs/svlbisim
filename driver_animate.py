import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import svlbisim_input_reader
import svlbisim_genuvcov
import svlbisim_obs
import svlbisim_plotting
import svlbisim_image

infile = sys.argv[1]
params = svlbisim_input_reader.load_yaml(infile)

if not os.path.exists(params['outdir']):
    os.makedirs(params['outdir'])

os.system('cp %s %s'%(infile, params['outdir']+'/used_inputs.yaml'))

svlbisim_genuvcov.main(params)

svlbisim_animate_orbits_uvcov.main(params)

