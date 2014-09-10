#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import diffusion_mri_processing         as dmri
from distutils                          import spawn
import argparse
import os

help_message = \
'Perform Diffusion Model Fitting with pre-processing steps. \n\n' + \
'Mandatory Inputs are the Diffusion Weighted Images and the bval/bvec pair. \n' + \
'as well as a T1 image for reference space. \n\n' + \
'If the Field maps are provided then Susceptibility correction is applied. \n' + \
'Use the --model option to control which diffusion model to use (tensor or noddi)' 

model_choices = ['tensor', 'noddi']

parser = argparse.ArgumentParser(description=help_message)
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
                    required=False)
parser.add_argument('-p','--fieldmapphase',
                    dest='fieldmapphase',
                    metavar='fieldmapphase',
                    help='Field Map Phase image file to be associated with the DWIs',
                    required=False)
parser.add_argument('--model',
                    dest='model',
                    metavar='model',
                    help='Diffusion Model to use, choices are ' + str(model_choices) + ' default: tensor',
                    required=False,
                    choices = model_choices,
                    default = model_choices[0])
parser.add_argument('--output_dir',dest='output_dir', type=str, \
                    metavar='directory', help='Output directory containing the registration result\n' + \
                    'Default is a directory called results', \
                    default=os.path.abspath('results'), required=False)

args = parser.parse_args()

result_dir = os.path.abspath(args.output_dir)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

if args.fieldmapmag is None or args.fieldmapphase is None:
    do_susceptibility_correction = False
else:
    do_susceptibility_correction = True

if do_susceptibility_correction == True:
    if not os.path.exists(os.path.abspath(args.fieldmapmag)) or not os.path.exists(os.path.abspath(args.fieldmapphase)):
        do_susceptibility_correction = False

r = dmri.create_diffusion_mri_processing_workflow(name = 'dmri_workflow',
                                                  resample_in_t1 = True,
                                                  log_data = True,
                                                  correct_susceptibility = do_susceptibility_correction,
                                                  dwi_interp_type = 'CUB',
                                                  t1_mask_provided = False,
                                                  ref_b0_provided = False,
                                                  model = args.model)

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
r.connect(r.get_node('output_node'), 'T1toB0_transformation', ds, '@transformation')

if (args.model == 'noddi'):
    r.connect(r.get_node('output_node'), 'mcmap', ds, '@mcmap')

r.write_graph(graph2use = 'colored')

qsub_exec=spawn.find_executable('qsub')
if not qsub_exec == None:
    qsubargs='-l h_rt=02:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    r.run(plugin='MultiProc')


