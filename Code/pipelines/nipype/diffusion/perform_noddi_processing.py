#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import diffusion_mri_processing         as dmri
import nipype.interfaces.fsl            as fsl

import noddi as noddi
from distutils                          import spawn
import argparse
import os, sys

import registration

import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg


def find_shell_files(input_directory):

    import os, glob
    dwis = glob.glob(os.path.join(input_directory, '*_corrected_dwi.nii.gz'))
    average_b0 = glob.glob(os.path.join(input_directory, '*_average_b0.nii.gz'))
    transformations = glob.glob(os.path.join(input_directory, 'transformations', '*.txt'))
    return dwis[0], average_b0[0], transformations


mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')

help_message = \
'Perform NODDI Model Fitting with pre-processing steps. \n\n' + \
'Mandatory Inputs are the Diffusion Weighted Images and the bval/bvec pair. \n' + \
'as well as a T1 image for reference space. \n\n' + \
'If the Field maps are provided then Susceptibility correction is applied.'

parser = argparse.ArgumentParser(description=help_message)
parser.add_argument('-i', '--dwis',
                    dest='dwis',
                    metavar='dwis',
                    nargs='+',
                    help='Diffusion Weighted Images in a 4D nifti file',
                    required=True)
parser.add_argument('-a','--bvals',
                    dest='bvals',
                    metavar='bvals',
                    nargs='+',
                    help='bval file to be associated with the DWIs',
                    required=True)
parser.add_argument('-e','--bvecs',
                    dest='bvecs',
                    metavar='bvecs',
                    nargs='+',
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
parser.add_argument('-o', '--output_dir', 
                    dest='output_dir', 
                    type=str,
                    metavar='output_dir', 
                    help='Output directory containing the registration result\n' + \
                    'Default is a directory called results',
                    default=os.path.abspath('results'), 
                    required=False)

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
input_file = os.path.basename(args.dwis[0])
# extracting the 'subject name simply for output name purposes
subject_name = input_file.replace('.nii.gz','')
subject_t1_name = os.path.basename(args.t1).replace('.nii.gz','')

number_of_shells = len(args.dwis)

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

diffusion_preprocs = []
diffusion_preprocs_sinks = []

for i in range(number_of_shells):
    
    r = dmri.create_diffusion_mri_processing_workflow(name = 'dmri_workflow_shell_'+str(i+1),
                                                      resample_in_t1 = False,
                                                      log_data = True,
                                                      correct_susceptibility = do_susceptibility_correction,
                                                      dwi_interp_type = 'CUB',
                                                      t1_mask_provided = True,
                                                      ref_b0_provided = False,
                                                      wls_tensor_fit = False,
                                                      set_op_basename = True)
    
    r.base_dir = os.getcwd()
    
    r.inputs.input_node.in_dwi_4d_file = os.path.abspath(args.dwis[i])
    r.inputs.input_node.in_bval_file = os.path.abspath(args.bvals[i])
    r.inputs.input_node.in_bvec_file = os.path.abspath(args.bvecs[i])
    
    r.inputs.input_node.in_fm_magnitude_file = os.path.abspath(args.fieldmapmag)
    r.inputs.input_node.in_fm_phase_file = os.path.abspath(args.fieldmapphase)
    r.inputs.input_node.in_t1_file = os.path.abspath(args.t1)
    r.inputs.input_node.op_basename = subject_name
    
    r.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
    r.connect(mask_resample, 'res_file', mask_eroder, 'in_file')
    r.connect(mask_eroder, 'out_file', r.get_node('input_node'), 'in_t1_mask')
    
    shell_outputdir = os.path.join(result_dir, 'shell_'+str(i+1))
    if not os.path.exists(shell_outputdir):
        os.mkdir(shell_outputdir)
    
    ds = pe.Node(nio.DataSink(), name='ds')
    ds.inputs.base_directory = shell_outputdir
    ds.inputs.parameterization = False

    subs = []
    subs.append(('vol0000_maths_res_merged_thresh_maths', subject_name + '_corrected_dwi'))
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
        
    diffusion_preprocs.append(r)   
    diffusion_preprocs_sinks.append(shell_outputdir)   
    








name = 'noddi_estimation'

workflow = pe.Workflow(name=name)
workflow.base_dir=os.getcwd()

output_merger_dwis = pe.Node(interface = niu.Merge(numinputs=number_of_shells),
                             name = 'output_merger_dwis')
output_merger_b0s = pe.Node(interface = niu.Merge(numinputs=number_of_shells),
                             name = 'output_merger_b0s')

shell_files_finders = []
for i in range(number_of_shells):

    shell_files_finder = pe.Node(interface = niu.Function(input_names = ['input_directory'],
                                                          output_names = ['dwis', 'average_b0', 'transformations'],
                                                          function = find_shell_files),
                                 name = 'shell_files_finder_shell_'+str(i+1))

    shell_files_finder.inputs.input_directory = diffusion_preprocs_sinks[i]

    workflow.connect(shell_files_finder, 'dwis', 
                     output_merger_dwis, 'in'+str(i+1))
    workflow.connect(shell_files_finder, 'average_b0', 
                     output_merger_b0s, 'in'+str(i+1))

    shell_files_finders.append(shell_files_finder)


# Perform rigid groupwise registration
groupwise_B0_coregistration = registration.create_atlas('groupwise_B0_coregistration', 
                                                        initial_ref = False, 
                                                        itr_rigid = 2, 
                                                        itr_affine = 0, 
                                                        itr_non_lin = 0)

workflow.connect(output_merger_b0s, 'out', 
                 groupwise_B0_coregistration, 'input_node.in_files')


# the input image is registered to the MNI for masking purpose
mni_to_b0 = pe.Node(interface=niftyreg.RegAladin(), 
                       name='mni_to_b0')
workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', 
                 mni_to_b0, 'ref_file')
mni_to_b0.inputs.flo_file = mni_template
b0_mask_resample  = pe.Node(interface = niftyreg.RegResample(), 
                         name = 'b0_mask_resample')
b0_mask_resample.inputs.inter_val = 'NN'
workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', 
                 b0_mask_resample, 'ref_file')
b0_mask_resample.inputs.flo_file = mni_template_mask
b0_mask_eroder = pe.Node(interface = niftyseg.BinaryMaths(), 
                         name = 'b0_mask_eroder')
b0_mask_eroder.inputs.operation = 'ero'
b0_mask_eroder.inputs.operand_value = 3
workflow.connect(b0_mask_resample, 'res_file', b0_mask_eroder, 'in_file')

output_merger_transformations = pe.Node(interface = niu.Merge(numinputs=number_of_shells),
                             name = 'output_merger_transformations')

for i in range(number_of_shells):
    
    transformation_selector = pe.Node(niu.Select(index = [i]),
                                      name = 'transformation_selector_shell'+str(i+1))
    workflow.connect(groupwise_B0_coregistration, 'output_node.trans_files', 
                     transformation_selector, 'inlist')
    
    transformation_composition = pe.MapNode(niftyreg.RegTransform(),
                                            name = 'transformation_composition_shell'+str(i+1), 
                                            iterfield=['comp_input2'])

    workflow.connect(transformation_selector, 'out', 
                     transformation_composition, 'comp_input')
    workflow.connect(shell_files_finders[i], 'transformations', 
                     transformation_composition, 'comp_input2')

    workflow.connect(transformation_composition, 'out_file', 
                     output_merger_transformations, 'in'+str(i+1))

    
    
# Resample the DWI and B0s
resampling = pe.MapNode(niftyreg.RegResample(), 
                        name = 'resampling', 
                        iterfield=['trans_file', 'flo_file'])
resampling.inputs.inter_val = 'CUB'

workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', 
                 resampling, 'ref_file')
workflow.connect(output_merger_dwis, 'out', 
                 resampling, 'flo_file')
workflow.connect(output_merger_transformations, 'out', 
                 resampling, 'trans_file')

# Remerge all the DWIs
merge_dwis_images = pe.Node(interface = fsl.Merge(dimension = 't'), 
                            name = 'merge_dwis_images')
workflow.connect(resampling, 'res_file',
                 merge_dwis_images, 'in_files')

merge_dwis_files = pe.Node(interface = niu.Function(input_names = ['input_dwis', 
                                                             'input_bvals', 
                                                             'input_bvecs'],
                                              output_names = ['dwis', 'bvals', 'bvecs'],
                                              function = dmri.merge_dwi_function),
                     name = 'merge_dwis_files')

merge_dwis_files.inputs.input_dwis = [os.path.abspath(f) for f in args.dwis]
merge_dwis_files.inputs.input_bvals = [os.path.abspath(f) for f in args.bvals]
merge_dwis_files.inputs.input_bvecs = [os.path.abspath(f) for f in args.bvecs]



noddi_fitting = pe.Node(interface = noddi.Noddi(),
                        name = 'noddi_fitting')
                 
workflow.connect(merge_dwis_images, 'merged_file', noddi_fitting, 'in_dwis')
workflow.connect(b0_mask_eroder, 'out_file', noddi_fitting, 'in_mask')
workflow.connect(merge_dwis_files, 'bvals', noddi_fitting, 'in_bvals')
workflow.connect(merge_dwis_files, 'bvecs', noddi_fitting, 'in_bvecs')

workflow.write_graph(graph2use = 'colored')

qsub_exec=spawn.find_executable('qsub')
if not qsub_exec == None:
    qsubargs='-l h_rt=02:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    workflow.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    workflow.run(plugin='Linear')
