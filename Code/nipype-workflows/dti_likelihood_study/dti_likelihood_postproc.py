#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyfit       as niftyfit
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.ttk            as ttk
from extract_roi_statistics import ExtractRoiStatistics
from write_array_to_csv import WriteArrayToCsv

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
                    'labels_file']),
        name='input_node')



    logger1 = pe.Node(interface = ttk.utils.TensorLog(), 
                      name = 'logger1')
    logger1.inputs.use_fsl_style = True

    logger2 = pe.Node(interface = ttk.utils.TensorLog(), 
                      name = 'logger2')
    logger2.inputs.use_fsl_style = True
    
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
    
    dwi_tmean = pe.Node(interface= niftyseg.maths.UnaryMaths(), 
                           name = 'dwi_meaner')
    dwi_tmean.inputs.operation = 'tmean'
    
    array_tensor_writer = pe.Node(interface=WriteArrayToCsv(),name='tensor_stat_writer')
    array_tensor_writer.inputs.in_name = 'tensor_stats'
    
    array_dwi_writer = pe.Node(interface=WriteArrayToCsv(),name='dwi_stat_writer')
    array_dwi_writer.inputs.in_name = 'dwi_stats'
    
    
    roi_tensor_stats = pe.Node(interface=ExtractRoiStatistics(), name='roi_tensor_stats')
    roi_dwi_stats = pe.Node(interface=ExtractRoiStatistics(), name='roi_dwi_stats')  
    
    output_node = pe.Node(
        niu.IdentityInterface(
            fields=['tensor_metric_map',
                    'tensor_metric_ROI_statistics',
                    'dwi_metric_map',
                    'dwi_metric_ROI_statistics',
                    'dwi_likelihood']),
        name='output_node')
    
    workflow.connect(input_node, 'simulated_tensors', logger1, 'in_file')
    workflow.connect(input_node, 'estimated_tensors', logger2, 'in_file')
    
    workflow.connect(logger1, 'out_file',  tensor_subtracter, 'in_file')
    workflow.connect(logger2, 'out_file',  tensor_subtracter, 'operand_file')
    
    workflow.connect(tensor_subtracter, 'out_file', tensor_sqr, 'in_file')
    workflow.connect(tensor_subtracter, 'out_file', tensor_sqr, 'operand_file')
    workflow.connect(tensor_sqr, 'out_file', tensor_tmean,'in_file')
    
    workflow.connect(input_node,'simulated_dwis',  dwi_subtracter, 'in_file')
    workflow.connect(input_node, 'estimated_dwis',  dwi_subtracter, 'operand_file')
    workflow.connect(dwi_subtracter, 'out_file', dwi_sqr, 'in_file')
    workflow.connect(dwi_subtracter, 'out_file', dwi_sqr, 'operand_file')
    workflow.connect(dwi_sqr, 'out_file', dwi_tmean,'in_file')
    
    workflow.connect(tensor_tmean, 'out_file', roi_tensor_stats, 'in_file')
    workflow.connect(input_node, 'labels_file', roi_tensor_stats, 'roi_file')
    
    workflow.connect(dwi_tmean, 'out_file', roi_dwi_stats, 'in_file')
    workflow.connect(input_node, 'labels_file', roi_dwi_stats, 'roi_file')
    
    workflow.connect(roi_tensor_stats, 'out_array', array_tensor_writer, 'in_array')
    workflow.connect(roi_dwi_stats, 'out_array', array_dwi_writer, 'in_array')
    
    workflow.connect( tensor_subtracter, 'out_file', output_node, 'tensor_metric_map')
    workflow.connect(array_tensor_writer, 'out_file', output_node, 'tensor_metric_ROI_statistics')
    
    workflow.connect(dwi_subtracter, 'out_file', output_node, 'dwi_metric_map')
    workflow.connect(array_dwi_writer, 'out_file', output_node, 'dwi_metric_ROI_statistics')
    #   workflow.connect(??, output_node, 'tensor_metric_ROI_statistics')
    
    return workflow
