#! /usr/bin/env python

import argparse

import diffusion_mri_processing as dmri

parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-b','--basedir', help='Base directory that contains the diffusion data in the required format',
                    required=True)
args = parser.parse_args()

basedir = args.basedir

dwis  = basedir + 'dwi-1.nii.gz'
bvals = basedir + 'dwi-1.bval'
bvecs = basedir + 'dwi-1.bvec'
T1    = basedir + 'dwi-1-mprage.nii.gz'
fmmag = basedir + 'dwi-1-fieldmap-magnitude.nii.gz'
fmph  = basedir + 'dwi-1-fieldmap-phase.nii.gz'

r = dmri.create_diffusion_mri_processing_workflow('dmri_workflow_log_2', resample_in_t1 = True, log_data = True)
r.base_dir = basedir

r.inputs.input_node.in_dwi_4d_file = dwis
r.inputs.input_node.in_bvec_file = bvecs
r.inputs.input_node.in_bval_file = bvals
r.inputs.input_node.in_fm_magnitude_file = fmmag
r.inputs.input_node.in_fm_phase_file = fmph
r.inputs.input_node.in_T1_file = T1

r.write_graph(graph2use = 'exec')

qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

#r.run()
#r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
#r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
r.run(plugin='MultiProc')

