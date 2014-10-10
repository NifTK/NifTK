#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import diffusion_mri_processing         as dmri
from distutils                          import spawn
import argparse
import os

import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')

help_message = \
'Perform Diffusion Model Fitting with pre-processing steps. \n\n' + \
'Mandatory Inputs are the Diffusion Weighted Images and the bval/bvec pair. \n' + \
'as well as a T1 image for reference space. \n\n' + \
'If the Field maps are provided then Susceptibility correction is applied.'

parser = argparse.ArgumentParser(description=help_message)
parser.add_argument('-i', '--dwis',
                    dest='dwis',
                    metavar='dwis',
                    help='Diffusion Weighted Images in a 4D nifti file',
                    required=True)
parser.add_argument('-a','--bvals',
                    dest='bvals',
                    metavar='bvals',
                    help='bval file to be associated with the DWIs',
                    required=True)
parser.add_argument('-e','--bvecs',
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

# extracting basename of the input file (list)
input_file = os.path.basename(args.dwis)
# extracting the directory where the input file is (are)
input_directory = os.path.abspath(os.path.dirname(args.dwis))
# extracting the 'subject name simply for output name purposes
subject_name = input_file.replace('.nii.gz','')
subject_t1_name = os.path.basename(args.t1).replace('.nii.gz','')


r = dmri.create_diffusion_mri_processing_workflow(name = 'dmri_workflow',
                                                  resample_in_t1 = False,
                                                  log_data = True,
                                                  correct_susceptibility = do_susceptibility_correction,
                                                  dwi_interp_type = 'CUB',
                                                  t1_mask_provided = True,
                                                  ref_b0_provided = False,
                                                  wls_tensor_fit = False,
                                                  set_op_basename = True)

r.base_dir = os.getcwd()

r.inputs.input_node.in_dwi_4d_file = os.path.abspath(args.dwis)
r.inputs.input_node.in_bvec_file = os.path.abspath(args.bvecs)
r.inputs.input_node.in_bval_file = os.path.abspath(args.bvals)
r.inputs.input_node.in_fm_magnitude_file = os.path.abspath(args.fieldmapmag)
r.inputs.input_node.in_fm_phase_file = os.path.abspath(args.fieldmapphase)
r.inputs.input_node.in_t1_file = os.path.abspath(args.t1)
r.inputs.input_node.op_basename = subject_name

# the input image is registered to the MNI for masking purpose
mni_to_input = pe.Node(interface=niftyreg.RegAladin(), 
                       name='mni_to_input')
mni_to_input.inputs.ref_file = os.path.abspath(args.t1)
mni_to_input.inputs.flo_file = mni_template
mask_resample  = pe.Node(interface = niftyreg.RegResample(), 
                         name = 'mask_resample')
mask_resample.inputs.inter_val = 'NN'
mask_resample.inputs.ref_file = os.path.abspath(args.t1)
mask_resample.inputs.flo_file = mni_template_mask
mask_eroder = pe.Node(interface = niftyseg.BinaryMaths(), 
                         name = 'mask_eroder')
mask_eroder.inputs.operation = 'ero'
mask_eroder.inputs.operand_value = 3

r.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
r.connect(mask_resample, 'res_file', mask_eroder, 'in_file')
r.connect(mask_eroder, 'out_file', r.get_node('input_node'), 'in_t1_mask')

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = result_dir
ds.inputs.parameterization = False

subs = []
subs.append(('vol0000_res_merged_maths', subject_name + '_corrected'))
subs.append(('average_output_res_maths', subject_name + '_average_b0'))
subs.append((subject_t1_name+ '_aff_reg_transform', subject_name + '_t1_transform'))
ds.inputs.regexp_substitutions = subs

r.connect(r.get_node('output_node'), 'tensor', ds, '@tensors')
r.connect(r.get_node('output_node'), 'FA', ds, '@fa')
r.connect(r.get_node('output_node'), 'MD', ds, '@md')
r.connect(r.get_node('output_node'), 'COL_FA', ds, '@colfa')
r.connect(r.get_node('output_node'), 'V1', ds, '@v1')
r.connect(r.get_node('output_node'), 'predicted_image_tensor', ds, '@img')
r.connect(r.get_node('output_node'), 'residual_image_tensor', ds, '@res')
r.connect(r.get_node('output_node'), 'dwis', ds, '@dwis')
r.connect(r.get_node('output_node'), 'transformations', ds, 'transformations')
r.connect(r.get_node('output_node'), 'average_b0', ds, '@b0')
r.connect(r.get_node('output_node'), 'T1toB0_transformation', ds, '@transformation')

r.write_graph(graph2use = 'colored')

qsub_exec=spawn.find_executable('qsub')
if not qsub_exec == None:
    qsubargs='-l h_rt=02:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    r.run(plugin='MultiProc')


