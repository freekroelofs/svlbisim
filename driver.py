import os
import sys
from svlbisim import input_reader, genuvcov, observe, griduv, plotting, image

infile = sys.argv[1]
params = input_reader.load_yaml(infile)

if not os.path.exists(params['outdir']):
    os.makedirs(params['outdir'])

os.system('cp %s %s'%(infile, params['outdir']+'/used_inputs.yaml'))

genuvcov.main(params)

observe.main(params)

if params['grid_uv'] == 'True':
    griduv.main(params)

if params['make_plots'] == 'True':
    plotting.main(params)

if params['make_image'] == 'True':
    image.main(params)
