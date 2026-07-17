import os
import sys
from svlbisim import input_reader, genuvcov, animate_orbits_uvcov

infile = sys.argv[1]
params = input_reader.load_yaml(infile)

if not os.path.exists(params['outdir']):
    os.makedirs(params['outdir'])

os.system('cp %s %s'%(infile, params['outdir']+'/used_inputs.yaml'))

genuvcov.main(params)

animate_orbits_uvcov.main(params)

