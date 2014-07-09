#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import diffusion_mri_processing         as dmri
import argparse
import os

parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-i', '--dwis',
                    dest='dwis',
                    metavar='dwis',
                    help='Diffusion Weighted Images in a 4D nifti file',
                    required=True)
parser.add_argument('-l','--bvals',
                    dest='bvals',
                    metavar='bvals',
                    help='bval file to be associated with the DWIs',
                    required=True)
parser.add_argument('-c','--bvecs',
                    dest='bvecs',
                    metavar='bvecs',
                    help='bvec file to be associated with the DWIs',
                    required=True)
parser.add_argument('-t','--t1',
                    dest='t1',
                    metavar='t1',
                    help='T1 file to be associated with the DWIs',
                    required=True)
parser.add_argument('-m','--fieldmapmag',
                    dest='fieldmapmag',
                    metavar='fieldmapmag',
                    help='Field Map Magnitude image file to be associated with the DWIs',
                    required=True)
parser.add_argument('-p','--fieldmapphase',
                    dest='fieldmapphase',
                    metavar='fieldmapphase',
                    help='Field Map Phase image file to be associated with the DWIs',
                    required=True)

args = parser.parse_args()

result_dir = os.path.join(os.getcwd(),'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
r = dmri.create_diffusion_mri_processing_workflow('dmri_workflow', 
                                                  resample_in_t1 = True, 
                                                  log_data = True)
r.base_dir = os.getcwd()

r.inputs.input_node.in_dwi_4d_file = os.path.abspath(args.dwis)
r.inputs.input_node.in_bvec_file = os.path.abspath(args.bvecs)
r.inputs.input_node.in_bval_file = os.path.abspath(args.bvals)
r.inputs.input_node.in_fm_magnitude_file = os.path.abspath(args.fieldmapmag)
r.inputs.input_node.in_fm_phase_file = os.path.abspath(args.fieldmapphase)
r.inputs.input_node.in_t1_file = os.path.abspath(args.t1)

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = result_dir
ds.inputs.parameterization = False

r.connect(r.get_node('output_node'), 'tensor', ds, '@tensors')
r.connect(r.get_node('output_node'), 'FA', ds, '@fa')
r.connect(r.get_node('output_node'), 'MD', ds, '@md')
r.connect(r.get_node('output_node'), 'COL_FA', ds, '@colfa')
r.connect(r.get_node('output_node'), 'V1', ds, '@v1')
r.connect(r.get_node('output_node'), 'predicted_image', ds, '@img')
r.connect(r.get_node('output_node'), 'residual_image', ds, '@res')
r.connect(r.get_node('output_node'), 'parameter_uncertainty_image', ds, '@unc')
r.connect(r.get_node('output_node'), 'dwis', ds, '@dwis')
r.connect(r.get_node('output_node'), 'transformations', ds, 'transformations')
r.connect(r.get_node('output_node'), 'average_b0', ds, '@b0')

r.write_graph(graph2use = 'colored')

qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

#r.run()
#r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
#r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
r.run(plugin='MultiProc')

