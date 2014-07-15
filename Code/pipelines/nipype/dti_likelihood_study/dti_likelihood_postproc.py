#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyfit       as niftyfit
import nipype.interfaces.fsl            as fsl
from extract_roi_statistics import ExtractRoiStatistics
from write_array_to_csv import WriteArrayToCsv
from calculate_distance_between_affines import CalculateAffineDistances

def create_dti_likelihood_post_proc_workflow(name='dti_likelihood_post_proc'):

    """
    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['simulated_dwis',
                    'estimated_dwis',
                    'simulated_tensors',
                    'estimated_tensors',
                    'labels_file',
                    'simulated_affines',
                    'estimated_affines',
                    'proc_residual_image']),
        name='input_node')

    # TODO: Compare estimated transformation matrices!

    logger1 = pe.Node(interface = niftyfit.DwiTool(dti_flag2 = True), 
                      name = 'logger1')
    logger2 = pe.Node(interface = niftyfit.DwiTool(dti_flag2 = True), 
                      name = 'logger2')
    
    tensor_subtracter = pe.Node(interface = niftyseg.maths.BinaryMaths(), 
                          name = 'tensor_subtracter')
    tensor_subtracter.inputs.operation = 'sub'
    
    tensor_sqr = pe.Node(interface=niftyseg.maths.BinaryMaths(),
                         name = 'tensor_squarer')
    tensor_sqr.inputs.operation = 'mul'
    
    tensor_tmean = pe.Node(interface= niftyseg.maths.UnaryMaths(), 
                           name = 'tensor_meaner')
    tensor_tmean.inputs.operation = 'tmean'
    
    dwi_subtracter = pe.Node(interface = fsl.maths.BinaryMaths(), 
                          name = 'dwi_substracter')
    dwi_subtracter.inputs.operation = 'sub'
    
    dwi_sqr = pe.Node(interface=niftyseg.maths.BinaryMaths(),
                         name = 'dwi_squarer')
    dwi_sqr.inputs.operation = 'mul'
    
    residual_sqr = pe.Node(interface=niftyseg.maths.BinaryMaths(),
                         name = 'residual_squarer')
    residual_sqr.inputs.operation = 'mul'
    
    # We need to resample the parcellation for each of the simulated DWI images
    resample_parc = pe.MapNode(interface=niftyreg.RegResample(), name = 'resample_atlas', iterfield = 'trans_file')
    resample_parc.inputs.inter_val = 'NN'

    # We need to merge the resampled parcellations before calculating the statistics    
    merge_parc = pe.Node(interface = fsl.Merge(dimension = 't'), name = 'merge_parc')
    
    calculate_affine_distance = pe.Node(interface=CalculateAffineDistances(), name="calculate_affine_distance")
    array_affine_writer = pe.Node(interface=WriteArrayToCsv(),name='affine_stat_writer')
    array_affine_writer.inputs.in_name = 'affine_stats'
    
    array_tensor_writer = pe.Node(interface=WriteArrayToCsv(),name='tensor_stat_writer')
    array_tensor_writer.inputs.in_name = 'tensor_stats'
    
    array_dwi_writer = pe.Node(interface=WriteArrayToCsv(),name='dwi_stat_writer')
    array_dwi_writer.inputs.in_name = 'dwi_stats'
    
    array_residual_writer = pe.Node(interface=WriteArrayToCsv(),name='residual_stat_writer')
    array_residual_writer.inputs.in_name = 'proc_residual_stats'
    
    
    
    roi_tensor_stats = pe.Node(interface=ExtractRoiStatistics(), name='roi_tensor_stats')
    roi_dwi_stats = pe.Node(interface=ExtractRoiStatistics(), name='roi_dwi_stats')
    roi_residual_stats = pe.Node(interface=ExtractRoiStatistics(), name='roi_residual_stats')  
    array_dwi_writer = pe.Node(interface=WriteArrayToCsv(),name='dwi_stat_writer')
    array_dwi_writer.inputs.in_name = 'dwi_stats'
    
    output_node = pe.Node(
        niu.IdentityInterface(
            fields=['tensor_metric_map',
                    'tensor_metric_ROI_statistics',
                    'dwi_metric_map',
                    'dwi_metric_ROI_statistics',
                    'dwi_likelihood',
                    'affine_distances',
                    'proc_residual_metric_ROI_statistics']),
        name='output_node')
    
    workflow.connect(input_node, 'simulated_tensors', logger1, 'source_file')
    workflow.connect(input_node, 'estimated_tensors', logger2, 'source_file')
    
    workflow.connect(logger1, 'logdti_file',  tensor_subtracter, 'in_file')
    workflow.connect(logger2, 'logdti_file',  tensor_subtracter, 'operand_file')
    
    workflow.connect(tensor_subtracter, 'out_file', tensor_sqr, 'in_file')
    workflow.connect(tensor_subtracter, 'out_file', tensor_sqr, 'operand_file')
    workflow.connect(tensor_sqr, 'out_file', tensor_tmean,'in_file')
    
    workflow.connect(input_node,'simulated_dwis',  dwi_subtracter, 'in_file')
    workflow.connect(input_node, 'estimated_dwis',  dwi_subtracter, 'operand_file')
    workflow.connect(dwi_subtracter, 'out_file', dwi_sqr, 'in_file')
    workflow.connect(dwi_subtracter, 'out_file', dwi_sqr, 'operand_file')
    
    # Square the residual of the processed image
    workflow.connect(input_node, 'proc_residual_image', residual_sqr, 'in_file')
    workflow.connect(input_node, 'proc_residual_image', residual_sqr, 'operand_file')
    # Calculate the statistics of the processed residual image based on the atlas
    workflow.connect(residual_sqr, 'out_file', roi_residual_stats, 'in_file')
    workflow.connect(input_node, 'labels_file', roi_residual_stats, 'roi_file')
    
    workflow.connect(tensor_tmean, 'out_file', roi_tensor_stats, 'in_file')
    workflow.connect(input_node, 'labels_file', roi_tensor_stats, 'roi_file')
    
    
    workflow.connect(input_node, 'labels_file', resample_parc, 'flo_file')
    workflow.connect(input_node, 'estimated_affines', resample_parc, 'trans_file')
    workflow.connect(input_node, 'labels_file', resample_parc, 'ref_file')
    
    workflow.connect(resample_parc, 'res_file', merge_parc, 'in_files')
    workflow.connect(dwi_sqr, 'out_file', roi_dwi_stats, 'in_file')
    workflow.connect(merge_parc, 'merged_file', roi_dwi_stats, 'roi_file')
    
    
    workflow.connect(roi_tensor_stats, 'out_array', array_tensor_writer, 'in_array')
    workflow.connect(roi_dwi_stats, 'out_array', array_dwi_writer, 'in_array')
    workflow.connect(roi_residual_stats, 'out_array', array_residual_writer, 'in_array')
    
    workflow.connect( tensor_subtracter, 'out_file', output_node, 'tensor_metric_map')
    workflow.connect(array_tensor_writer, 'out_file', output_node, 'tensor_metric_ROI_statistics')
    
    workflow.connect(dwi_subtracter, 'out_file', output_node, 'dwi_metric_map')
    workflow.connect(array_dwi_writer, 'out_file', output_node, 'dwi_metric_ROI_statistics')
    workflow.connect(array_residual_writer, 'out_file', output_node, 'proc_residual_metric_ROI_statistics')
    
    workflow.connect(input_node, 'simulated_affines', calculate_affine_distance,'transformation1_list')
    workflow.connect(input_node, 'estimated_affines', calculate_affine_distance,'transformation2_list')
    workflow.connect(calculate_affine_distance, 'out_array', array_affine_writer, 'in_array')
    workflow.connect(array_affine_writer, 'out_file', output_node, 'affine_distances')
    #   workflow.connect(??, output_node, 'tensor_metric_ROI_statistics')
    
    return workflow
