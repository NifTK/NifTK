#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
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
    gtab = gradients.gradient_table(bvals, bvecs)
    masklist = list(gtab.b0s_mask)
    ret_val = []
    for i in range(len(masklist)):
        if masklist[i] == True:
            ret_val.append(i)
    return ret_val

def get_DWIs_from_bvals_bvecs(bvals, bvecs):
    import dipy.core.gradients as gradients
    gtab = gradients.gradient_table(bvals, bvecs)
    masklist = list(gtab.b0s_mask)
    ret_val = []
    for i in range(len(masklist)):
        if masklist[i] == False:
            ret_val.append(i)
    return ret_val

def reorder_list_from_bval_bvecs(B0s, DWIs, bvals, bvecs):
    import dipy.core.gradients as gradients
    gtab = gradients.gradient_table(bvals, bvecs)
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

def create_diffusion_mri_processing_workflow(name='diffusion_mri_processing', correct_susceptibility = True):

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
                    'in_T1_file']),
        name='inputnode')
    
    #Node using fslsplit() to split the 4D file, separate the B0 and DWIs
    split_dwis = pe.Node(interface = fsl.Split(dimension="t"), name = 'split_dwis')
    
    # Node using fslsplit to split the two fieldmap magnitude images    
    split_fm_mag = pe.Node(interface = fsl.Split(dimension="t"), name='split_fm_mag')
    select_first_fm_mag = pe.Node(interface = niu.Select(), name = 'select_first_fm_mag')
    select_first_fm_mag.inputs.index = 0

    #Node using niu.Select() to select only the B0 files
    function_find_B0s = niu.Function(input_names=['bvals', 'bvecs'], output_names=['out'])
    function_find_B0s.inputs.function_str = str(inspect.getsource(get_B0s_from_bvals_bvecs))
    find_B0s = pe.Node(interface = function_find_B0s, name = 'find_B0s')
    select_B0s = pe.Node(interface = niu.Select(), name = 'select_B0s')

    select_first_B0 = pe.Node(interface = niu.Select(), name = 'select_first_B0')
    select_first_B0.inputs.index = 0

    #Node using niu.Select() to select only the DWIs files
    function_find_DWIs = niu.Function(input_names=['bvals', 'bvecs'], output_names=['out'])
    function_find_DWIs.inputs.function_str = str(inspect.getsource(get_DWIs_from_bvals_bvecs))
    find_DWIs   = pe.Node(interface = function_find_DWIs, name = 'find_DWIs')
    select_DWIs = pe.Node(interface = niu.Select(), name = 'select_DWIs')

    groupwise_B0_coregistration = create_linear_coregistration_workflow('groupwise_B0_coregistration')
    
    susceptibility_correction = create_fieldmap_susceptibility_workflow('susceptibility_correction')
    susceptibility_correction.inputs.input_node.etd = 2.46
    susceptibility_correction.inputs.input_node.rot = 34.56
    susceptibility_correction.inputs.input_node.ped = '-y'
    
    dwi_to_B0_registration = pe.MapNode(niftyreg.RegAladin(), name = 'dwi_to_B0_registration', iterfield=['flo_file'])

    #Node using niu.Merge() to put back together the list of B0s and DWIs
    function_reorder_files = niu.Function(input_names=['B0s', 'DWIs', 'bvals', 'bvecs'], output_names=['out'])
    function_reorder_files.inputs.function_str = str(inspect.getsource(reorder_list_from_bval_bvecs))

    reorder_transformations = pe.Node(interface = function_reorder_files, name = 'reorder_transformations')
    
    #do the node for the gradient reorientation
    #gradient_reorientation = pe.Node()
    
    transformation_composition = pe.MapNode(niftyreg.RegTransform(), name = 'transformation_composition', iterfield=['comp_input2'])
    
    resampling = pe.MapNode(niftyreg.RegResample(), name = 'resampling', iterfield=['trans_file', 'flo_file'])

    merge_dwis = pe.Node(interface = fsl.Merge(), name = 'merge_dwis')
    merge_dwis.inputs.dimension = 't'

    T1_mask = pe.Node(interface=fsl.BET(), name='T1_mask')
    T1_mask.inputs.mask = True

    T1_mask_resampling = pe.Node(niftyreg.RegResample(), name = 'T1_mask_resampling')

    tensor_fitting = pe.Node(interface=fsl.DTIFit(),name='tensor_fitting')
    
    outputnode = pe.Node( interface=niu.IdentityInterface(fields=['tensor', 'FA', 'MD', 'MO', 'S0', 'L1', 'L2', 'L3', 'V1', 'V2', 'V3']),
                          name="outputnode" )

    workflow.connect(input_node, 'in_dwi_4d_file', split_dwis, 'in_file')

    workflow.connect(input_node, 'in_bval_file', find_B0s,  'bvals')
    workflow.connect(input_node, 'in_bvec_file', find_B0s,  'bvecs')
    workflow.connect(input_node, 'in_bval_file', find_DWIs, 'bvals')
    workflow.connect(input_node, 'in_bvec_file', find_DWIs, 'bvecs')

    workflow.connect(split_dwis, 'out_files', select_B0s, 'inlist')
    workflow.connect(find_B0s,   'out',       select_B0s, 'index')

    workflow.connect(split_dwis, 'out_files', select_DWIs,'inlist')
    workflow.connect(find_DWIs,  'out',       select_DWIs, 'index')

    workflow.connect(select_B0s,       'out', select_first_B0, 'inlist')

    workflow.connect(select_B0s,       'out', groupwise_B0_coregistration, 'input_node.in_files')
    workflow.connect(select_first_B0,  'out', groupwise_B0_coregistration, 'input_node.ref_file')
    
    if correct_susceptibility == True:
        # Need to insert an fslsplit
        workflow.connect(input_node, 'in_fm_magnitude_file', split_fm_mag, 'in_file')
        workflow.connect(split_fm_mag, 'out_files', select_first_fm_mag, 'inlist')
        workflow.connect(select_first_fm_mag,                  'out',      susceptibility_correction, 'input_node.mag_image')
        workflow.connect(input_node,                  'in_fm_phase_file',          susceptibility_correction, 'input_node.phase_image')
        workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', susceptibility_correction, 'input_node.average_b0')
        #    workflow.connect(input_node,                  'in_T1_file',                susceptibility_correction, 'input_node.in_T1_file')
    
    workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', dwi_to_B0_registration, 'ref_file')
    workflow.connect(select_DWIs,                 'out',                       dwi_to_B0_registration, 'flo_file')

    workflow.connect(groupwise_B0_coregistration, 'output_node.aff_files', reorder_transformations, 'B0s')
    workflow.connect(dwi_to_B0_registration,      'aff_file',              reorder_transformations, 'DWIs')
    workflow.connect(input_node,                  'in_bval_file',          reorder_transformations, 'bvals')
    workflow.connect(input_node,                  'in_bvec_file',          reorder_transformations, 'bvecs')
    
    if correct_susceptibility == True:
        workflow.connect(susceptibility_correction, 'output_node.out_field', transformation_composition, 'comp_input')
        workflow.connect(reorder_transformations,   'out',                   transformation_composition, 'comp_input2')

    workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', resampling, 'ref_file')
    workflow.connect(split_dwis,                  'out_files',                 resampling, 'flo_file')
    
    if correct_susceptibility ==True:
        workflow.connect(transformation_composition,  'out_file',                  resampling, 'trans_file')
    else:
        workflow.connect(reorder_transformations,     'out',                       resampling, 'trans_file')

    workflow.connect(resampling, 'res_file',   merge_dwis, 'in_files')
    
    workflow.connect(input_node, 'in_T1_file', T1_mask, 'in_file')

    workflow.connect(groupwise_B0_coregistration, 'output_node.average_image', T1_mask_resampling, 'ref_file')
    workflow.connect(T1_mask,                     'mask_file',                 T1_mask_resampling, 'flo_file')

    workflow.connect(merge_dwis, 'merged_file',      tensor_fitting, 'dwi')
    workflow.connect(input_node, 'in_bvec_file',     tensor_fitting, 'bvecs')
    workflow.connect(input_node, 'in_bval_file',     tensor_fitting, 'bvals')
    workflow.connect(T1_mask_resampling, 'res_file', tensor_fitting, 'mask')
    
    workflow.connect(tensor_fitting, 'tensor', outputnode, 'tensor')
    workflow.connect(tensor_fitting, 'FA',     outputnode, 'FA')
    workflow.connect(tensor_fitting, 'MD',     outputnode, 'MD')
    workflow.connect(tensor_fitting, 'MO',     outputnode, 'MO')
    workflow.connect(tensor_fitting, 'S0',     outputnode, 'S0')
    workflow.connect(tensor_fitting, 'L1',     outputnode, 'L1')
    workflow.connect(tensor_fitting, 'L2',     outputnode, 'L2')
    workflow.connect(tensor_fitting, 'L3',     outputnode, 'L3')
    workflow.connect(tensor_fitting, 'V1',     outputnode, 'V1')
    workflow.connect(tensor_fitting, 'V2',     outputnode, 'V2')
    workflow.connect(tensor_fitting, 'V3',     outputnode, 'V3')
    
    return workflow
    

    # node explanation:
    
    # input_node
    # contains the entry point for the following inputs: 
    # - in_dwi_4d_file
    # - in_bvec_file
    # - in_bval_file
    # - in_fm_magnitude_file
    # - in_fm_phase_file
    # - in_T1_file

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
    
