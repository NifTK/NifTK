#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyfit       as niftyfit
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.susceptibility as susceptibility

import inspect

#from groupwise_registration_workflow    import *
from registration    import *
from susceptibility_correction import *

'''
This file provides the creation of the whole workflow necessary for 
processing diffusion MRI images.
'''

def get_B0s_from_bvals_bvecs(bvals, bvecs):
    import dipy.core.gradients as gradients
    gtab = gradients.gradient_table(bvals, bvecs, b0_threshold=20)
    masklist = list(gtab.b0s_mask)
    ret_val = []
    for i in range(len(masklist)):
        if masklist[i] == True:
            ret_val.append(i)
    return ret_val

def get_DWIs_from_bvals_bvecs(bvals, bvecs):
    import dipy.core.gradients as gradients
    gtab = gradients.gradient_table(bvals, bvecs, b0_threshold=20)
    masklist = list(gtab.b0s_mask)
    ret_val = []
    for i in range(len(masklist)):
        if masklist[i] == False:
            ret_val.append(i)
    return ret_val

def reorder_list_from_bval_bvecs(B0s, DWIs, bvals, bvecs):
    import dipy.core.gradients as gradients
    gtab = gradients.gradient_table(bvals, bvecs, b0_threshold=20)
    masklist = list(gtab.b0s_mask)
    B0s_indices  = []
    DWIs_indices = []
    for i in range(len(masklist)):
        if masklist[i] == True:
            B0s_indices.append(i)
        else:
            DWIs_indices.append(i)
    total_list_length = len(B0s) + len(DWIs)
    ret_val = [''] * total_list_length
    i = 0
    for index in B0s_indices:
        ret_val[index] = B0s[i]
        i = i+1
    i = 0
    for index in DWIs_indices:
        ret_val[index] = DWIs[i]
        i = i+1
    return ret_val


def merge_dwi_function (input_dwis, input_bvals, input_bvecs):

    import os, glob, sys, errno
    import nipype.interfaces.fsl as fsl
    import nibabel as nib
    import numpy as np 
    
    qdiff_eps = 1.0

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
        np.savetxt(output_file, result, fmt = '%.3f')
        return output_file
    
    if len(input_bvals) == 0 or len(input_bvecs) == 0 or len(input_dwis) == 0:
        print 'I/O One of the dwis merge input is empty, exiting.'
        sys.exit(errno.EIO)

    if not type(input_dwis) == list:
        input_dwis =  [input_dwis]
        input_bvals = [input_bvals]
        input_bvecs = [input_bvecs]

    qforms = []
    for dwi in input_dwis:
        im = nib.load(dwi)
        qform = im.get_qform()
        qforms.append(qform)

    if len(input_dwis) > 1:
        # Walk backwards through the dwis, assume later scans are more likely to be correctly acquired!
        for file_index in range(len(input_dwis)-2,-1,-1):
            # If the qform doesn't match that of the last scan, throw it away
            if np.linalg.norm(qforms[len(input_dwis)-1] - qforms[file_index]) > qdiff_eps:
                print '** WARNING ** : The qform of the DWIs dont match, denoting a re-scan error, throwing ', input_dwis[file_index]
                input_dwis.pop(file_index)
                input_bvals.pop(file_index)
                input_bvecs.pop(file_index)
    
    # Set the default values of these variables as the first index, 
    # in case we only have one image and we don't do a merge
    dwis = input_dwis[0]
    bvals = input_bvals[0]
    bvecs = input_bvecs[0]
    if len(input_dwis) > 1:
        merger = fsl.Merge(dimension = 't')
        merger.inputs.in_files = input_dwis
        res = merger.run()
        dwis = res.outputs.merged_file
        bvals = merge_vector_files(input_bvals)
        bvecs = merge_vector_files(input_bvecs)

    return dwis, bvals, bvecs


def create_diffusion_mri_processing_workflow(name='diffusion_mri_processing', 
                                             correct_susceptibility = True, 
                                             resample_in_t1 = False, 
                                             log_data = False, 
                                             t1_mask_provided = False,
                                             ref_b0_provided = False,
                                             dwi_interp_type = 'CUB',
                                             wls_tensor_fit = True,
                                             set_op_basename = False):

    """Creates a diffusion processing workflow. This initially performs a groupwise registration
    of all the B=0 images, subsequently each of the DWI is registered to the averageB0.
    If enabled, the averageB0 is corrected for magnetic susceptibility distortion. 
    Tensor, and derivative images, are estimated using niftyfit 
    
    Example
    -------

    >>> dmri_proc = create_diffusion_mri_processing_workflow(name='dmri_proc')
    >>> dmri_proc.inputs.


    Inputs::

        input_node.in_dwi_4d_file - The original 4D DWI image file
        input_node.in_bvec_file - The bvector file of the DWI 
        input_node.in_bval_file - The bvalue of the DWI images
        input_node.in_fm_magnitude_file - The field map magnitude image 
            (only required if correct_susceptibility = True)
        input_node.in_fm_phase_file - The field map phase image
            (only required if correct_susceptibility = True)
        input_node.in_t1_file - The T1 image 
        input_node.in_t1_mask - A mask provided in the T1 space 
            (Only require if t1_mask_provided = True)
           
    Outputs::

        output_node.


    Optional arguments::
        correct_susceptibility - Correct for magnetic susceptibility distortion
            (default = True) 
        resample_in_t1 - Resample the DTI derivative images in the T1 space (default = False)
        log_data - Register the log of the DWI to the log average B0 (default = False)
        t1_mask_provided - A T1 image mask is provided externally, otherwise bet is used (default = False)

    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_dwi_4d_file',
                    'in_bvec_file',
                    'in_bval_file',
                    'in_fm_magnitude_file',
                    'in_fm_phase_file',
                    'in_t1_file',
                    'in_t1_mask',
                    'in_ref_b0',
                    'op_basename'], mandatory_inputs=False),
        name='input_node')

    #Node using fslsplit() to split the 4D file, separate the B0 and DWIs
    split_dwis = pe.Node(interface = fsl.Split(dimension="t"), name = 'split_dwis')
    
    # Node using fslsplit to split the two fieldmap magnitude images    
    split_fm_mag = pe.Node(interface = fsl.Split(dimension="t"), name='split_fm_mag')
    select_first_fm_mag = pe.Node(interface = niu.Select(index = 0), name = 'select_first_fm_mag')

    # Node using niu.Select() to select only the B0 files
    function_find_B0s = niu.Function(input_names=['bvals', 'bvecs'], output_names=['out'])
    function_find_B0s.inputs.function_str = str(inspect.getsource(get_B0s_from_bvals_bvecs))
    find_B0s = pe.Node(interface = function_find_B0s, name = 'find_B0s')
    select_B0s = pe.Node(interface = niu.Select(), name = 'select_B0s')

    # Node using niu.Select() to select only the DWIs files
    function_find_DWIs = niu.Function(input_names=['bvals', 'bvecs'], output_names=['out'])
    function_find_DWIs.inputs.function_str = str(inspect.getsource(get_DWIs_from_bvals_bvecs))
    find_DWIs   = pe.Node(interface = function_find_DWIs, name = 'find_DWIs')
    select_DWIs = pe.Node(interface = niu.Select(), name = 'select_DWIs')
    
    # Perform rigid groupwise registration
    groupwise_B0_coregistration = create_atlas('groupwise_B0_coregistration', 
                                               initial_ref = ref_b0_provided, 
                                               itr_rigid = 2, 
                                               itr_affine = 0, 
                                               itr_non_lin = 0)
    
    # Perform susceptibility correction, where we already have a mask in the b0 space
    susceptibility_correction = create_fieldmap_susceptibility_workflow('susceptibility_correction',
                                                                        mask_exists = True,
                                                                        reg_to_t1 = True)
    susceptibility_correction.inputs.input_node.etd = 2.46

    # rot value for the DRC
    susceptibility_correction.inputs.input_node.rot = 34.56
    # rot value for the 1946
    # susceptibility_correction.inputs.input_node.rot = 25.92

    susceptibility_correction.inputs.input_node.ped = '-y'
    
    # As we're trying to estimate an affine transformation, and rotations and shears are confounded
    # easier just to optimise an affine directly for the DWI 
    dwi_to_B0_registration = pe.MapNode(niftyreg.RegAladin(aff_direct_flag = True), 
                                        name = 'dwi_to_B0_registration',
                                        iterfield=['flo_file'])
    
    reorder_transformations = pe.Node(interface = niu.Function(
        input_names = ['B0s', 'DWIs', 'bvals', 'bvecs'], 
        output_names = ['out'],
        function = reorder_list_from_bval_bvecs), 
                                      name = 'reorder_transformations')
    
    #TODO: do the node for the gradient reorientation
    #gradient_reorientation = pe.Node()
    
    # Compose the nonlinear and linear deformations to correct the DWI
    transformation_composition = pe.MapNode(niftyreg.RegTransform(),
                                            name = 'transformation_composition', 
                                            iterfield=['comp_input2'])

    change_datatype = pe.MapNode(interface=niftyseg.BinaryMaths(operation='add', 
                                                                operand_value = 0.0, 
                                                                output_datatype = 'float'), 
                                 name='change_datatype',
                                 iterfield = ['in_file'])
    
    # Resample the DWI and B0s
    resampling = pe.MapNode(niftyreg.RegResample(), name = 'resampling', iterfield=['trans_file', 'flo_file'])
    resampling.inputs.inter_val = dwi_interp_type
    
    # Remerge all the DWIs
    merge_dwis = pe.Node(interface = fsl.Merge(dimension = 't'), name = 'merge_dwis')
    
    # Divide the corrected merged DWIs by the distortion Jacobian image to dampen compression effects
    divide_dwis = pe.Node(interface = niftyseg.BinaryMaths(operation='div'), name = 'divide_dwis')
    
    # The masks in T1 mask needs to be rigidly registered to the B0 space
    T1_to_b0_registration = pe.Node(niftyreg.RegAladin(rig_only_flag = True, nac_flag = True), 
                                    name='T1_to_b0_registration')
    T1_to_b0_registration.inputs.rig_only_flag = True
    # Use resampling for the T1 image
    T1_resampling = pe.Node(niftyreg.RegResample(), name = 'T1_resampling')
    # Use nearest neighbour resampling for the mask image
    T1_mask_resampling = pe.Node(niftyreg.RegResample(inter_val = 'NN'), name = 'T1_mask_resampling')

    # Threshold the DWIs to 0: if we use cubic or sync interpolation we may end up with
    # negative DWi values at sharp edges, which is not physically possible.
#    threshold_dwis = pe.Node(interface=niftyseg.BinaryMaths(operation='thr', 
#                                                            operand_value = 0.0), 
#                             name='threshold_dwis')
    threshold_dwis = pe.Node(interface=fsl.maths.Threshold(thresh = 0.0, direction = 'below'), 
                             name='threshold_dwis')

    # Fit the tensor model
    diffusion_model_fitting_tensor = pe.Node(interface=niftyfit.FitDwi(),name='diffusion_model_fitting_tensor')
    diffusion_model_fitting_tensor.inputs.dti_flag = True
    diffusion_model_fitting_tensor.inputs.wls_flag = wls_tensor_fit

    # The reorientation of the tensors is not necessary.
    # It is taken care of in the tensor resampling process itself (reg_resample)
    # We force the reorientation flag to 0 just to be sure
    diffusion_model_fitting_tensor.inputs.rotsform_flag = 0
    
    # Output node
    output_node = pe.Node( interface=niu.IdentityInterface(
        fields=['tensor',
                'FA', 
                'MD', 
                'COL_FA', 
                'V1', 
                'predicted_image_tensor',
                'residual_image_tensor',
                'dwis',
                'transformations',
                'average_b0',
                'mcmap',
                'T1toB0_transformation',
                'dwi_mask']),
                           name="output_node" )

    #############################################################
    # Find the Diffusion data and separate between DWIs and B0s #
    #############################################################

    workflow.connect(input_node, 'in_dwi_4d_file', split_dwis, 'in_file')
    workflow.connect(input_node, 'in_bval_file', find_B0s,  'bvals')
    workflow.connect(input_node, 'in_bvec_file', find_B0s,  'bvecs')
    workflow.connect(input_node, 'in_bval_file', find_DWIs, 'bvals')
    workflow.connect(input_node, 'in_bvec_file', find_DWIs, 'bvecs')
    
    workflow.connect(find_B0s,   'out',       select_B0s, 'index')
    workflow.connect(find_DWIs,  'out',       select_DWIs, 'index')
    
    workflow.connect(split_dwis, 'out_files', select_B0s, 'inlist')
    workflow.connect(split_dwis, 'out_files', select_DWIs,'inlist')
    
    #############################################################
    #             groupwise atlas of B0 images                  #
    #############################################################
    workflow.connect(select_B0s, 'out', groupwise_B0_coregistration, 'input_node.in_files')
    
    if ref_b0_provided == True:
        workflow.connect(input_node, 'in_ref_b0', groupwise_B0_coregistration, 'input_node.ref_file')
    
    
    #############################################################
    #    T1 to B0 rigid registration and resampling             #
    #############################################################
    workflow.connect(groupwise_B0_coregistration, 'output_node.average_image',T1_to_b0_registration, 'ref_file')
    workflow.connect(input_node, 'in_t1_file',T1_to_b0_registration, 'flo_file')
    
    workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', T1_resampling, 'ref_file')
    workflow.connect(input_node, 'in_t1_file', T1_resampling, 'flo_file')
    workflow.connect(T1_to_b0_registration, 'aff_file', T1_resampling, 'trans_file')
    
    workflow.connect(T1_to_b0_registration, 'aff_file', T1_mask_resampling, 'trans_file')
    workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', T1_mask_resampling, 'ref_file')
    
    # If we have a proper T1 mask, we can use that, otherwise make one using BET
    if t1_mask_provided == True:
        workflow.connect(input_node, 'in_t1_mask', T1_mask_resampling, 'flo_file')
    else:
        T1_mask = pe.Node(interface=fsl.BET(mask=True), name='T1_mask')
        workflow.connect(input_node, 'in_t1_file', T1_mask, 'in_file')
        workflow.connect(T1_mask, 'mask_file', T1_mask_resampling, 'flo_file')

    #############################################################
    #             DWI to B0 affine registration                 #
    #############################################################

    if log_data == True:
        # If we're logging the DWI before registration, need to connect the logged images
        # rather than the split dwi into the dwi_to_b0_registration
        # Make a log images node
        log_ims = pe.MapNode(interface = fsl.UnaryMaths(operation = 'log', output_datatype = 'float'),
                             name = 'log_ims', iterfield=['in_file'])
        log_b0 = pe.Node(interface = fsl.UnaryMaths(operation = 'log'), name = 'log_b0')
        # The amount to smooth the logged diffusion weighted images by (in voxels)        
        smooth_log_sigma = 0.75

        smooth_ims = pe.MapNode(interface = niftyseg.BinaryMaths(operation = 'smo',operand_value = smooth_log_sigma),
                                name = 'smooth_ims', iterfield=['in_file'])
        smooth_b0 = pe.Node(interface = niftyseg.BinaryMaths(operation = 'smo',operand_value = smooth_log_sigma),
                                name = 'smooth_b0')
        
        workflow.connect(select_DWIs,'out', log_ims, 'in_file')
        workflow.connect(log_ims, 'out_file', smooth_ims, 'in_file')
        
        workflow.connect(groupwise_B0_coregistration, 'output_node.average_image',
                         log_b0, 'in_file')
        workflow.connect(log_b0, 'out_file', smooth_b0, 'in_file')

        workflow.connect(smooth_b0, 'out_file',
                         dwi_to_B0_registration, 'ref_file')
        workflow.connect(smooth_ims, 'out_file', dwi_to_B0_registration, 'flo_file')

    else:
        
        workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', dwi_to_B0_registration, 'ref_file')
        workflow.connect(select_DWIs,                 'out',                       dwi_to_B0_registration, 'flo_file')

    # Once we have a resampled mask from the T1 space, this can be used as a ref mask for the B0 registration
    # the susceptibility correction and for the tensor fitting    
    workflow.connect(T1_mask_resampling, 'res_file', dwi_to_B0_registration, 'rmask_file')
    

    #############################################################
    #             Susceptibility Correction                     #
    #############################################################
    
    if correct_susceptibility == True:
        workflow.connect(input_node, 'in_fm_magnitude_file', split_fm_mag,'in_file')
        workflow.connect(split_fm_mag, 'out_files', select_first_fm_mag, 'inlist')
        workflow.connect(select_first_fm_mag, 'out', susceptibility_correction, 'input_node.mag_image')
        workflow.connect(input_node, 'in_fm_phase_file', susceptibility_correction, 'input_node.phase_image')
        workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', susceptibility_correction, 'input_node.epi_image')
        workflow.connect(T1_resampling, 'res_file', susceptibility_correction, 'input_node.t1')
        workflow.connect(susceptibility_correction, 'output_node.out_field', transformation_composition, 'comp_input')
        workflow.connect(reorder_transformations, 'out', transformation_composition, 'comp_input2')
        workflow.connect(T1_mask_resampling, 'res_file', susceptibility_correction, 'input_node.mask_image')

    
    #############################################################
    # Reorder the B0 and DWIs transformations to match the bval #
    #############################################################
    
    workflow.connect(groupwise_B0_coregistration, 'output_node.trans_files', reorder_transformations, 'B0s')
    workflow.connect(dwi_to_B0_registration, 'aff_file', reorder_transformations, 'DWIs')
    workflow.connect(input_node, 'in_bval_file', reorder_transformations, 'bvals')
    workflow.connect(input_node, 'in_bvec_file', reorder_transformations, 'bvecs')

    #############################################################
    #   Resample the DWIs with affine (+susceptibility)         #
    #   transformations and merge back into a 4D image          #
    #############################################################
    
    workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', resampling, 'ref_file')
    workflow.connect(split_dwis, 'out_files', change_datatype, 'in_file')
    workflow.connect(change_datatype, 'out_file', resampling, 'flo_file')
    
    if correct_susceptibility ==True:
        workflow.connect(transformation_composition, 'out_file', resampling, 'trans_file')
    else:
        workflow.connect(reorder_transformations, 'out', resampling, 'trans_file')        
    
    workflow.connect(resampling, 'res_file',   merge_dwis, 'in_files')

    #############################################################
    #   Fit the tensor model from the DWI data                  #
    #############################################################
    # Set the op basename for the tensor
    if set_op_basename == True:
        workflow.connect(input_node, 'op_basename', diffusion_model_fitting_tensor, 'op_basename')
    else:
        diffusion_model_fitting_tensor.inputs.op_basename = 'dti'
        
    workflow.connect(merge_dwis, 'merged_file', threshold_dwis, 'in_file')    
    
    if correct_susceptibility == True:
        # If we're correcting for susceptibility distortions, need to divide by the
        # jacobian of the distortion field
        # Connect up the correct image to the tensor fitting software
        workflow.connect(threshold_dwis, 'out_file', divide_dwis, 'in_file')
        workflow.connect(susceptibility_correction, 'output_node.out_jac', divide_dwis, 'operand_file')
        workflow.connect(divide_dwis,'out_file',  diffusion_model_fitting_tensor, 'source_file')
    else:
        workflow.connect(threshold_dwis, 'out_file', diffusion_model_fitting_tensor, 'source_file')

    workflow.connect(T1_mask_resampling, 'res_file', diffusion_model_fitting_tensor, 'mask_file')    
    workflow.connect(input_node, 'in_bvec_file', diffusion_model_fitting_tensor, 'bvec_file')
    workflow.connect(input_node, 'in_bval_file', diffusion_model_fitting_tensor, 'bval_file')        
    
    #############################################################
    #         Prepare data for output_node                      #
    #############################################################

    inv_T1_aff = pe.Node(niftyreg.RegTransform(), name = 'inv_T1_aff')
    workflow.connect(T1_to_b0_registration, 'aff_file', inv_T1_aff, 'inv_aff_input')
    workflow.connect(inv_T1_aff, 'out_file', output_node, 'T1toB0_transformation')
    
    if resample_in_t1 == True:

        resamp_tensors = pe.Node(niftyreg.RegResample(), name='resamp_tensors')
        resamp_tensors.inputs.tensor_flag = True
        dwi_tool = pe.Node(interface = niftyfit.DwiTool(dti_flag2 = True), name = 'dwi_tool')
        
        if set_op_basename == True:
            workflow.connect(input_node, 'op_basename', dwi_tool, 'op_basename')
        else:
            dwi_tool.inputs.op_basename = 'dti'     
            
        workflow.connect(input_node, 'in_t1_file', resamp_tensors, 'ref_file')
        workflow.connect(diffusion_model_fitting_tensor, 'tenmap_file', resamp_tensors, 'flo_file')
        workflow.connect(inv_T1_aff, 'out_file', resamp_tensors, 'trans_file')
        workflow.connect(resamp_tensors, 'res_file', dwi_tool, 'source_file')
        workflow.connect(resamp_tensors, 'res_file', output_node, 'tensor')
        workflow.connect(dwi_tool, 'famap_file', output_node, 'FA')
        workflow.connect(dwi_tool, 'mdmap_file', output_node, 'MD')
        workflow.connect(dwi_tool, 'rgbmap_file', output_node, 'COL_FA')
        workflow.connect(dwi_tool, 'v1map_file', output_node, 'V1')

    else:
        
        workflow.connect(diffusion_model_fitting_tensor, 'tenmap_file', output_node, 'tensor')
        workflow.connect(diffusion_model_fitting_tensor, 'mdmap_file', output_node, 'MD')
        workflow.connect(diffusion_model_fitting_tensor, 'famap_file', output_node, 'FA')
        workflow.connect(diffusion_model_fitting_tensor, 'rgbmap_file', output_node, 'COL_FA')
        workflow.connect(diffusion_model_fitting_tensor, 'v1map_file', output_node, 'V1')

    workflow.connect(diffusion_model_fitting_tensor, 'res_file', output_node, 'residual_image_tensor')
    workflow.connect(diffusion_model_fitting_tensor, 'syn_file', output_node, 'predicted_image_tensor')
    
    if correct_susceptibility == True:
        workflow.connect(susceptibility_correction, 'output_node.out_epi', output_node, 'average_b0')
        workflow.connect(divide_dwis,'out_file',  output_node, 'dwis')
    else:
        workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', output_node, 'average_b0')    
        workflow.connect(threshold_dwis, 'out_file', output_node, 'dwis')

    workflow.connect(reorder_transformations, 'out', output_node, 'transformations')
    workflow.connect(T1_mask_resampling, 'res_file', output_node, 'dwi_mask')

    return workflow    

    # node explanation:
    
    # input_node
    # contains the entry point for the following inputs: 
    # - in_dwi_4d_file
    # - in_bvec_file
    # - in_bval_file
    # - in_fm_magnitude_file
    # - in_fm_phase_file
    # - in_t1_file

    # split_dwis
    # Node using fsl.Split() to split the 4D image (in_dwi_4d_file) into 3D volume files

    # select_B0s
    # Node using niu.Select() to select only B0s from the list
    
    # select_DWIs 
    # Node using niu.Select() to select only DWIs from the list
    
    # groupwise_B0_coregistration
    # the groupwise registration workflow (IVOR), iterated over the 'flo_file'
    
    # susceptibility_correction
    # the susceptibility correction workflow, TODO
    
    # dwi_to_B0_registration
    # the groupwise registration node, it's iterated on 'flo_file'

    # gradient_reorientation
    # gradient reorientation node, TODO

    # transformation_composition
    # node using RegTransform() to compose between a linear transformation 
    # and a deformation field, it's iterated over 'file1' and 'file2 and 'file3'

    # resampling
    # node to resample every DWI and B0 with composed transformation, 
    # iterated on 'ref_file' 'flo_file' and 'res_file'
    
    # merge_dwis_list
    #do the node for merging all the DWIs and B0s files into one list
    
    # merge_dwis
    #do the node for merging all the files into one 4D
    
