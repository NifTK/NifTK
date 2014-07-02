#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyfit       as niftyfit
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.ttk            as ttk

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
    
    substracter = pe.Node(interface = fsl.maths.BinaryMaths(), 
                          name = 'substracter')
    substracter.inputs.operation = 'sub'
    
    output_node = pe.Node(
        niu.IdentityInterface(
            fields=['tensor_metric_map',
                    'tensor_metric_ROI_statistics']),
        name='output_node')
    
    workflow.connect(input_node, 'simulated_tensors', logger1, 'in_file')
    workflow.connect(input_node, 'estimated_tensors', logger2, 'in_file')
    
    workflow.connect(logger1, 'out_file', substracter, 'in_file')
    workflow.connect(logger2, 'out_file', substracter, 'operand_file')
    
    workflow.connect(substracter, 'out_file', output_node, 'tensor_metric_map')
    #   workflow.connect(??, output_node, 'tensor_metric_ROI_statistics')
    
    return workflow
