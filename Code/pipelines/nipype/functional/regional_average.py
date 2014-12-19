#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.pipeline.engine           as pe           # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg     # NiftySeg
import nipype.interfaces.niftyreg       as niftyreg     # NiftyReg
from extract_roi_statistics             import ExtractRoiStatistics
from normalise_roi_average_values       import NormaliseRoiAverageValues
from write_array_to_csv                 import WriteArrayToCsv

import os

def create_mask_from_functional():
    workflow = pe.Workflow(name='mask_func')
    workflow.base_dir = os.getcwd()
    workflow.base_output_dir='mask_func'
    # Create all the required nodes
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['functional_files']),
            name='input_node')
    otsu_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='otsu_filter', iterfield=['in_file'])
    erosion_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='erosion_filter', iterfield=['in_file'])
    lconcomp_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='lconcomp_filter', iterfield=['in_file'])
    dilation_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='dilation_filter', iterfield=['in_file'])
    fill_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='fill_filter', iterfield=['in_file'])
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['mask_files']),
            name='output_node')
    # Define the node options
    otsu_filter.inputs.operation='otsu'
    erosion_filter.inputs.operation='ero'
    erosion_filter.inputs.operand_value=1
    lconcomp_filter.inputs.operation='lconcomp'
    dilation_filter.inputs.operation='dil'
    dilation_filter.inputs.operand_value=5
    fill_filter.inputs.operation='fill'
    fill_filter.inputs.output_datatype='char'
    # Create the connections
    workflow.connect(input_node, 'functional_files', otsu_filter, 'in_file')
    workflow.connect(otsu_filter, 'out_file', erosion_filter, 'in_file')
    workflow.connect(erosion_filter, 'out_file', lconcomp_filter, 'in_file')
    workflow.connect(lconcomp_filter, 'out_file', dilation_filter, 'in_file')
    workflow.connect(dilation_filter, 'out_file', fill_filter, 'in_file')
    workflow.connect(fill_filter, 'out_file', output_node, 'mask_files')    
    
    return workflow
    
def create_mask_from_parcelation():
    workflow = pe.Workflow(name='mask_mri')
    workflow.base_dir = os.getcwd()
    workflow.base_output_dir='mask_mri'
    # Create all the required nodes
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['par_files']),
            name='input_node')
    binary_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='binary_filter', iterfield=['in_file'])
    erosion_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='erosion_filter', iterfield=['in_file'])
    lconcomp_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='lconcomp_filter', iterfield=['in_file'])
    dilation_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='dilation_filter', iterfield=['in_file'])
    fill_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='fill_filter', iterfield=['in_file'])
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['mask_files']),
            name='output_node')
    # Define the node options
    binary_filter.inputs.operation='bin'
    erosion_filter.inputs.operation='ero'
    erosion_filter.inputs.operand_value=1
    lconcomp_filter.inputs.operation='lconcomp'
    dilation_filter.inputs.operation='dil'
    dilation_filter.inputs.operand_value=5
    fill_filter.inputs.operation='fill'
    fill_filter.inputs.output_datatype='char'
    # Create the connections
    workflow.connect(input_node, 'par_files', binary_filter, 'in_file')
    workflow.connect(binary_filter, 'out_file', erosion_filter, 'in_file')
    workflow.connect(erosion_filter, 'out_file', lconcomp_filter, 'in_file')
    workflow.connect(lconcomp_filter, 'out_file', dilation_filter, 'in_file')
    workflow.connect(dilation_filter, 'out_file', fill_filter, 'in_file')
    workflow.connect(fill_filter, 'out_file', output_node, 'mask_files')    
    
    return workflow
    
def regroup_roi(in_file, roi_list):
    import nibabel as nib
    import numpy as np
    import os
    # Load the parcelation
    parcelation=nib.load(in_file)
    data=parcelation.get_data()
    # Create a new empy segmentation
    new_roi_data=np.zeros(data.shape)
    # Extract the relevant roi from the initial parcelation
    for i in roi_list:
        new_roi_data=new_roi_data + np.equal(data,i*np.ones(data.shape))
    new_roi_data = new_roi_data!=0
    # Create a new image based on the initial parcelation
    out_img = nib.Nifti1Image(np.uint8(new_roi_data), parcelation.get_affine())
    out_file=os.getcwd()+'/roi_'+os.path.basename(in_file)
    out_img.set_data_dtype('uint8')
    out_img.set_qform(parcelation.get_qform())
    out_img.set_sform(parcelation.get_sform())
    out_img.set_filename(out_file)
    out_img.to_filename(out_file)
    return out_file

def create_reg_avg_value_pipeline(reg_avg_value_var):
    workflow = pe.Workflow(name='reg_avg_value_pipeline')
    workflow.base_dir = os.getcwd()
    workflow.base_output_dir='reg_avg_value'
    # Create the input node interface
    if len(reg_avg_value_var.input_freesurfer_par) > 0:
        input_node = pe.Node(
            interface = niu.IdentityInterface(
                fields=['functional_files',
                        'mri_files',
                        'par_files',
                        'seg_files',
                        'trans_files',
                        'freesurfer_files']),
                        name='input_node')
        input_node.inputs.freesurfer_files=reg_avg_value_var.input_freesurfer_par
    else:
        input_node = pe.Node(
            interface = niu.IdentityInterface(
                fields=['functional_files',
                        'mri_files',
                        'par_files',
                        'seg_files',
                        'trans_files']),
                        name='input_node')
    input_node.inputs.functional_files=reg_avg_value_var.input_img
    input_node.inputs.mri_files=reg_avg_value_var.input_mri
    input_node.inputs.par_files=reg_avg_value_var.input_par
    input_node.inputs.seg_files=reg_avg_value_var.input_seg
    # Perform the registration if none is specified
    if len(reg_avg_value_var.input_trans) > 0:
        input_node.inputs.trans_files=reg_avg_value_var.input_trans
        # The input transformations are inverted
        invert_affine=pe.MapNode(interface = niftyreg.RegTransform(), \
            name='invert_affine', iterfield=['inv_aff_input'])
        workflow.connect(input_node, 'trans_files', invert_affine, 'inv_aff_input')
    else:
        # First step is to generate quick masks for both MRI and functional images
        func_mask = create_mask_from_functional()
        mri_mask = create_mask_from_parcelation()
        # Connections
        workflow.connect(input_node, 'functional_files', func_mask, 'input_node.functional_files')
        workflow.connect(input_node, 'par_files', mri_mask, 'input_node.par_files')
        # The MRI and functional images are globally registered
        aladin=pe.MapNode(interface = niftyreg.RegAladin(), name='aladin', \
            iterfield=['ref_file', 'flo_file', 'rmask_file', 'fmask_file'])
        if reg_avg_value_var.func_mri_use_affine==False:
            aladin.inputs.rig_only_flag=True
        aladin.inputs.verbosity_off_flag=True
        aladin.inputs.v_val=80
        aladin.inputs.nosym_flag=False
        # Connections
        workflow.connect(input_node, 'functional_files', aladin, 'ref_file')
        workflow.connect(input_node, 'mri_files', aladin, 'flo_file')
        workflow.connect(func_mask, 'output_node.mask_files', aladin, 'rmask_file')
        workflow.connect(mri_mask, 'output_node.mask_files', aladin, 'fmask_file')
    if reg_avg_value_var.roi=='gm_cereb':
        # The grey matter segmentation is exracted
        extract_timepoint=pe.MapNode(interface = niftyseg.BinaryMaths(), \
            name='extract_timepoint', iterfield=['in_file'])
        extract_timepoint.inputs.operation='tp'
        extract_timepoint.inputs.operand_value=int(2)
        workflow.connect(input_node, 'seg_files', extract_timepoint, 'in_file')
        # The segmentation is resampled in the space of functional image
        resample_seg=pe.MapNode(interface = niftyreg.RegResample(), name='resample_seg', \
            iterfield=['ref_file', 'flo_file', 'trans_file'])
        resample_seg.inputs.inter_val='LIN'
        resample_seg.inputs.verbosity_off_flag=True
        workflow.connect(input_node, 'functional_files', resample_seg, 'ref_file')
        workflow.connect(extract_timepoint, 'out_file', resample_seg, 'flo_file')
        if len(reg_avg_value_var.input_trans) > 0:
            workflow.connect(invert_affine, 'out_file', resample_seg, 'trans_file')
        else:
            workflow.connect(aladin, 'aff_file', resample_seg, 'trans_file')
    # The parcelation is resampled in the space of functional images
    resample_par=pe.MapNode(interface = niftyreg.RegResample(), name='resample_par', \
        iterfield=['ref_file', 'flo_file', 'trans_file'])
    resample_par.inputs.inter_val='NN'
    resample_par.inputs.verbosity_off_flag=True
    workflow.connect(input_node, 'functional_files', resample_par, 'ref_file')
    workflow.connect(input_node, 'par_files', resample_par, 'flo_file')
    if len(reg_avg_value_var.input_trans) > 0:
        workflow.connect(invert_affine, 'out_file', resample_par, 'trans_file')
    else:
        workflow.connect(aladin, 'aff_file', resample_par, 'trans_file')
    # Extract all the regional update values
    extract_uptakes = pe.MapNode(interface = ExtractRoiStatistics(), name='extract_uptakes',
        iterfield=['in_file','roi_file'])
    workflow.connect(input_node, 'functional_files', extract_uptakes, 'in_file')
    workflow.connect(resample_par, 'res_file', extract_uptakes, 'roi_file')
    # The gray matter cerebellum ROI used for normalisation is extracted if required
    if reg_avg_value_var.roi=='gm_cereb':
        # Extract the cerebellum information
        extract_cerebellum = pe.MapNode(interface = 
                                            niu.Function(input_names = ['in_file',
                                                                        'roi_list'],
                                           output_names = ['out_file'],
                                           function=regroup_roi),
                                        name='extract_cerebellum',
                                        iterfield=['in_file'])
        extract_cerebellum.inputs.roi_list=[39,40,41,42,72,73,74]
        workflow.connect(resample_par, 'res_file', extract_cerebellum, 'in_file')
        # Binarise the grey matter segmentation
        binarise_gm_seg=pe.MapNode(interface = niftyseg.BinaryMaths(), \
            name='binarise_gm_seg', iterfield=['in_file'])
        binarise_gm_seg.inputs.operation='thr'
        binarise_gm_seg.inputs.operand_value=reg_avg_value_var.seg_threshold
        workflow.connect(resample_seg, 'res_file', binarise_gm_seg, 'in_file')
        binarise2_gm_seg=pe.MapNode(interface = niftyseg.UnaryMaths(), \
            name='binarise2_gm_seg', iterfield=['in_file'])
        binarise2_gm_seg.inputs.operation='bin'
        workflow.connect(binarise_gm_seg, 'out_file', binarise2_gm_seg, 'in_file')
        # Extract the interaction between cerebellum and grey matter segmentation
        get_gm_cereb=pe.MapNode(interface = niftyseg.BinaryMaths(), \
            name='get_gm_cereb', iterfield=['in_file', 'operand_file'])
        get_gm_cereb.inputs.operation='mul'
        workflow.connect(binarise2_gm_seg, 'out_file', get_gm_cereb, 'in_file')
        workflow.connect(extract_cerebellum, 'out_file', get_gm_cereb, 'operand_file')
        
        extract_gm_cereb_uptake = pe.MapNode(interface = ExtractRoiStatistics(), \
            name='extract_gm_cereb_uptake', iterfield=['in_file','roi_file'])
        workflow.connect(input_node, 'functional_files', extract_gm_cereb_uptake, 'in_file')
        workflow.connect(get_gm_cereb, 'out_file', extract_gm_cereb_uptake, 'roi_file')
        
        # Normalise the uptake values
        norm_uptakes = pe.MapNode(interface = NormaliseRoiAverageValues(), name='norm_uptakes',
            iterfield=['in_file','in_array', 'cereb_array'])
        workflow.connect(input_node, 'functional_files', norm_uptakes, 'in_file')
        workflow.connect(extract_uptakes, 'out_array', norm_uptakes, 'in_array')
        workflow.connect(extract_gm_cereb_uptake, 'out_array', norm_uptakes, 'cereb_array')
        norm_uptakes.inputs.roi=reg_avg_value_var.roi
    else:
        # Normalise the uptake values
        norm_uptakes = pe.MapNode(interface = NormaliseRoiAverageValues(), name='norm_uptakes',
            iterfield=['in_file','in_array'])
        workflow.connect(input_node, 'functional_files', norm_uptakes, 'in_file')
        workflow.connect(extract_uptakes, 'out_array', norm_uptakes, 'in_array')
        norm_uptakes.inputs.roi=reg_avg_value_var.roi

    # Create the output node interface
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['out_files',
                    'norm_files',
                    'aff_files',
                    'freesurfer_out_files']),
            name='output_node')
    # Export the result
    workflow.connect(norm_uptakes, 'out_csv_file', output_node, 'out_files')
    workflow.connect(norm_uptakes, 'out_file', output_node, 'norm_files')
    if len(reg_avg_value_var.input_trans) > 0:
        workflow.connect(invert_affine, 'out_file', output_node, 'aff_files')
    else:
        workflow.connect(aladin, 'aff_file', output_node, 'aff_files')
    
    # Use the freesurfer parcelation if specified
    if len(reg_avg_value_var.input_freesurfer_par) > 0:
        fs_resampling = pe.MapNode(interface = niftyreg.RegResample(), name='fs_resampling',
                                   iterfield=['ref_file', 'flo_file', 'trans_file'])
        workflow.connect(input_node, 'functional_files', fs_resampling, 'ref_file')
        workflow.connect(input_node, 'freesurfer_files', fs_resampling, 'flo_file')
        if len(reg_avg_value_var.input_trans) > 0:
            workflow.connect(invert_affine, 'out_file', fs_resampling, 'trans_file')
        else:
            workflow.connect(aladin, 'aff_file', fs_resampling, 'trans_file')
        fs_resampling.inputs.inter_val='NN'
        fs_resampling.inputs.verbosity_off_flag=True
        extract_fs_uptakes = pe.MapNode(interface = ExtractRoiStatistics(),
                                         name='extract_fs_uptakes',
                                         iterfield=['in_file', 'roi_file'])
        workflow.connect(norm_uptakes, 'out_file', extract_fs_uptakes, 'in_file')
        workflow.connect(fs_resampling, 'res_file', extract_fs_uptakes, 'roi_file')
        fs_write_array = pe.MapNode(interface = WriteArrayToCsv(), name='fs_write_array',
                                   iterfield=['in_array', 'in_name'])
        workflow.connect(extract_fs_uptakes, 'out_array', fs_write_array, 'in_array')
        name_array=[]
        for i in range(0,len(reg_avg_value_var.input_freesurfer_par)):
            name_array.append('freesurfer_array_'+str(i))
        fs_write_array.inputs.in_name=name_array
        workflow.connect(fs_write_array, 'out_file', output_node, 'freesurfer_out_files')
        
    # Return the created workflow
    return workflow
