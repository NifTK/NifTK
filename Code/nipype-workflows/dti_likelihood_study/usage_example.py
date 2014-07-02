#! /usr/bin/env python

import dti_likelihood_study as dmri
import argparse
import math

parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-b','--basedir', help='Base directory that contains the diffusion data in the required format',
                    required=True)
args = parser.parse_args()

basedir = args.basedir

bvals = basedir + 'chicago-template.bval'
bvecs = basedir + 'chicago-template.bvec'
T1    = basedir + 'chicago-template-T1.nii.gz'
tensors = basedir + 'chicago-template-tensors.nii.gz'
B0 = basedir + 'chicago-template-b0.nii.gz'

r = dmri.create_dti_reproducibility_study_workflow(name='dti_likelihood_study')
r.base_dir = basedir

r.inputs.input_node.in_tensors_file = tensors
r.inputs.input_node.in_bvec_file = bvecs
r.inputs.input_node.in_bval_file = bvals
r.inputs.input_node.in_B0_file = B0
r.inputs.input_node.in_T1_file = T1
# default values before 
# r.inputs.input_node.in_stddev_translation = 0.4
# r.inputs.input_node.in_stddev_rotation = 0.02*math.pi/180
# r.inputs.input_node.in_stddev_shear = 0.0006
# r.inputs.input_node.in_noise_sigma = 10.0

# reasonable values from literature 
# r.inputs.input_node.in_stddev_translation = 0.0
# r.inputs.input_node.in_stddev_rotation = 0.5*math.pi/180
# r.inputs.input_node.in_stddev_shear = 0.075
# r.inputs.input_node.in_noise_sigma = 0.0

# testing values 
r.inputs.input_node.in_stddev_translation = 1.0
r.inputs.input_node.in_stddev_rotation = 0.7*math.pi/180
r.inputs.input_node.in_stddev_shear = 0.075
r.inputs.input_node.in_noise_sigma = 10.0

r.write_graph(graph2use = 'colored')

qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

#r.run()
#r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
#r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
r.run(plugin='MultiProc')

