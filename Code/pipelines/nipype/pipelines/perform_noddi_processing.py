#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import nipype.interfaces.fsl            as fsl
from distutils                          import spawn
import argparse
import os, sys

import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg

import niftk


def gen_substitutions(op_basename):    
    subs = []    

    subs.append(('average_output_res_maths', op_basename+'_average_b0'))
    subs.append(('vol0000_maths_res_merged_thresh_maths', op_basename+'_corrected_dwi'))
    subs.append(('MNI152_T1_2mm_brain_mask_dil_res_maths_res', op_basename+'_mask'))
    subs.append((r'/([^/;]+)_merged.bval', '/' + op_basename + '_corrected_dwi.bval'))
    subs.append((r'/([^/;]+)_merged.bvec', '/' + op_basename + '_corrected_dwi.bvec'))
    subs.append((r'/([^/;]+)_aff_reg_transform.txt', '/' + op_basename + '_T1_to_B0.txt'))
    subs.append((r'/transformations/vol', '/transformations/' + op_basename + '_dwi_to_b0_'))
    subs.append(('_log_maths_aff.txt', '_aff.txt'))
    subs.append(('vol0000_aff_rotation', op_basename + '_dwi_to_b0_rotation'))

    return subs

def merge_bv_function (input_bvals, input_bvecs):

    import os, glob, sys, errno
    import nipype.interfaces.fsl as fsl
    import nibabel as nib
    import numpy as np 
    
    def merge_vector_files(input_files):
        import numpy as np
        import os
        result = np.array([])
        files_base, files_ext = os.path.splitext(os.path.basename(input_files[0]))
        for f in input_files:
            if result.size == 0:
                result = np.loadtxt(f)
            else:
                result = np.hstack((result, np.loadtxt(f)))
        output_file = os.path.abspath(files_base + '_merged' + files_ext)
        if len(result.shape) == 1:
            np.savetxt(output_file, result, fmt = '%.3f', newline=" ")
        else:
            np.savetxt(output_file, result, fmt = '%.3f')
            
        return output_file
    
    if len(input_bvals) == 0 or len(input_bvecs) == 0:
        print 'I/O One of the dwis merge input is empty, exiting.'
        sys.exit(errno.EIO)

    if not type(input_bvals) == list:
        input_bvals = [input_bvals]
        input_bvecs = [input_bvecs]
    
    # Set the default values of these variables as the first index, 
    # in case we only have one image and we don't do a merge
    bvals = input_bvals[0]
    bvecs = input_bvecs[0]
    if len(input_bvals) > 1:
        bvals = merge_vector_files(input_bvals)
        bvecs = merge_vector_files(input_bvecs)

    return bvals, bvecs



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
    
    r = niftk.diffusion.create_diffusion_mri_processing_workflow(name = 'dmri_workflow_shell_'+str(i+1),
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
    
    if do_susceptibility_correction:
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

    interslice_qc = pe.Node(interface = niftk.qc.InterSliceCorrelationPlot(), 
                            name = 'interslice_qc')
    
    matrixrotation_qc = pe.Node(interface = niftk.qc.MatrixRotationPlot(), 
                                name = 'matrixrotation_qc')
    
    r.connect(r.get_node('output_node'), 'dwis', interslice_qc, 'in_file')
    interslice_qc.inputs.bval_file = os.path.abspath(args.bvals[i])

    r.connect(r.get_node('output_node'), 'transformations', matrixrotation_qc, 'in_files')
    

    ds = pe.Node(nio.DataSink(), name='ds')
    ds.inputs.base_directory = shell_outputdir
    ds.inputs.parameterization = False

    subsgen = pe.Node(interface = niu.Function(input_names = ['op_basename'], 
                                               output_names = ['substitutions'], 
                                               function = gen_substitutions), 
                      name = 'subsgen')
    subsgen.inputs.op_basename =  subject_name

    r.connect(subsgen, 'substitutions', ds, 'regexp_substitutions')

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
    r.connect(r.get_node('output_node'), 'dwi_mask', ds, '@dwi_mask')

    r.connect(interslice_qc, 'out_file', ds, '@interslice_qc')
    r.connect(matrixrotation_qc, 'out_file', ds, '@matrixrotation_qc')

    dot_exec=spawn.find_executable('dot')   
    if not dot_exec == None:
        r.write_graph(graph2use='colored')
    
    qsub_exec=spawn.find_executable('qsub')

    # Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
    try:    
        qsubargs=os.environ['QSUB_OPTIONS']
    except KeyError:                
        print 'The environtment variable QSUB_OPTIONS is not set up, we cannot queue properly the process. Using the default script options.'
      	qsubargs='-l h_rt=02:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
       	print qsubargs

    # We can use qsub or not depending on this environment variable, by default we use it.
    try:    
        run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
    except KeyError:                
        run_qsub=True

    if not qsub_exec == None and run_qsub:
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

    split_dwis = pe.Node(interface = fsl.Split(dimension="t"), name = 'split_dwis_shell'+str(i+1))

    workflow.connect(shell_files_finder, 'dwis', 
                     split_dwis, 'in_file')
    workflow.connect(split_dwis, 'out_files', 
                     output_merger_dwis, 'in'+str(i+1))
    workflow.connect(shell_files_finder, 'average_b0', 
                     output_merger_b0s, 'in'+str(i+1))

    shell_files_finders.append(shell_files_finder)


# Perform rigid groupwise registration
groupwise_B0_coregistration = niftk.registration.create_atlas('groupwise_B0_coregistration', 
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

merge_bv_files = pe.Node(interface = niu.Function(input_names = ['input_bvals', 
                                                                 'input_bvecs'],
                                                  output_names = ['bvals', 'bvecs'],
                                                  function = merge_bv_function),
                         name = 'merge_bv_files')

merge_bv_files.inputs.input_bvals = [os.path.abspath(f) for f in args.bvals]
merge_bv_files.inputs.input_bvecs = [os.path.abspath(f) for f in args.bvecs]

noddi_fitting = pe.Node(interface = niftk.Noddi(),
                        name = 'noddi_fitting')

workflow.connect(merge_dwis_images, 'merged_file', noddi_fitting, 'in_dwis')
workflow.connect(b0_mask_eroder, 'out_file', noddi_fitting, 'in_mask')
workflow.connect(merge_bv_files, 'bvals', noddi_fitting, 'in_bvals')
workflow.connect(merge_bv_files, 'bvecs', noddi_fitting, 'in_bvecs')

data_sink = pe.Node(nio.DataSink(), name='data_sink')
data_sink.inputs.base_directory = result_dir
data_sink.inputs.parameterization = False

workflow.connect(noddi_fitting, 'out_neural_density', data_sink, '@out_neural_density')
workflow.connect(noddi_fitting, 'out_orientation_dispersion_index', data_sink, '@out_orientation_dispersion_index')
workflow.connect(noddi_fitting, 'out_csf_volume_fraction', data_sink, '@out_csf_volume_fraction')
workflow.connect(noddi_fitting, 'out_objective_function', data_sink, '@out_objective_function')
workflow.connect(noddi_fitting, 'out_kappa_concentration', data_sink, '@out_kappa_concentration')
workflow.connect(noddi_fitting, 'out_error', data_sink, '@out_error')
workflow.connect(noddi_fitting, 'out_fibre_orientations_x', data_sink, '@out_fibre_orientations_x')
workflow.connect(noddi_fitting, 'out_fibre_orientations_y', data_sink, '@out_fibre_orientations_y')
workflow.connect(noddi_fitting, 'out_fibre_orientations_z', data_sink, '@out_fibre_orientations_z')    
workflow.connect(b0_mask_eroder, 'out_file', data_sink, '@dwi_mask')
workflow.connect(merge_dwis_images, 'merged_file', data_sink, '@corrected_dwis')
workflow.connect(merge_bv_files, 'bvals', data_sink, '@bvals')
workflow.connect(merge_bv_files, 'bvecs', data_sink, '@bvecs')

dot_exec=spawn.find_executable('dot')   
if not dot_exec == None:
    workflow.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')

# Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
try:
    qsubargs=os.environ['QSUB_OPTIONS']
except KeyError:
    qsubargs='-l h_rt=02:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

# We can use qsub or not depending of this environment variable, by default we use it.
try:
    run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
except KeyError:   
    run_qsub=True

if not qsub_exec == None and run_qsub:
    workflow.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    workflow.run(plugin='Linear')
