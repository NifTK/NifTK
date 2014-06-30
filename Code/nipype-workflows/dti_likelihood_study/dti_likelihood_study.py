#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyfit       as niftyfit
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.susceptibility as susceptibility

import diffusion_mri_processing         as dmri
import nipype.interfaces.ttk            as ttk
import diffusion_distortion_simulation        as distortion_sim
import math

'''
This file provides the creation of the whole workflow necessary for 
processing diffusion MRI images.
'''

def create_dti_reproducibility_study_workflow(name='create_dti_reproducibility_study'):

    """
    Example
    -------

    >>> dti_repr_st = create_dti_reproducibility_study_workflow(name='dti_repr_st')
    >>> dti_repr_st.inputs.


    Inputs::

        input_node.
    Outputs::

        output_node.


    Optional arguments::
        
    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_tensors_file',
                    'in_T1_file',
                    'in_B0_file',
                    'in_bvec_file',
                    'in_bval_file']),
        name='input_node')
    
    distortion_generator = pe.Node(interface = distortion_sim.DistortionGenerator(), 
                                   name = 'distortion_generator')
    distortion_generator.inputs.stddev_translation_val = 0.2
    distortion_generator.inputs.stddev_rotation_val = 0.01*math.pi/180
    distortion_generator.inputs.stddev_shear_val = 0.0003
    
    tensor_resampling = pe.MapNode(interface = niftyreg.RegResample(), 
                                   name = 'tensor_resampling', 
                                   iterfield = ['trans_file'])
                                   
    b0_resampling = pe.MapNode(interface=niftyreg.RegResample(),
                                  name = 'b0_resampling',
                                  iterfield = ['trans_file'])
                                   

    tensor_2_dwi = pe.MapNode(interface = niftyfit.DwiTool(), 
                              name = 'tensor_2_dwi', 
                              iterfield = ['source_file','b0_file'])

    merge_dwis = pe.Node(interface = fsl.Merge(), 
                         direction = 't',
                         name = 'merge_dwis')
    
    r = dmri.create_diffusion_mri_processing_workflow(name = 'dmri_workflow', correct_susceptibility = False)

    inv_estimated_distortions = pe.MapNode(interface = niftyreg.RegTransform(), 
                                           name = 'inv_estimated_distortions', 
                                           iterfield = ['inv_aff_input'])

    tensor_resampling_2 = pe.MapNode(interface = niftyreg.RegResample(), 
                                     name = 'tensor_resampling_2', 
                                     iterfield = ['trans_file'])
    b0_resampling_2 = pe.MapNode(interface=niftyreg.RegResample(),
                                  name = 'b0_resampling_2',
                                  iterfield = ['trans_file'])

    tensor_2_dwi_2 = pe.MapNode(interface = niftyfit.DwiTool(dtiFlag2 = True), 
                                name = 'tensor_2_dwi_2', 
                                iterfield = ['in_file'])

    merge_dwis_2 = pe.Node(interface = fsl.Merge(), 
                          direction = 't',
                          name = 'merge_dwis_2')

    # Output node
    output_node = pe.Node( interface=niu.IdentityInterface(
        fields=['simulated_dwis',
                'estimated_dwis',
                'simulated_tensors',
                'estimated_tensors']),
                           name="output_node" )
                           
    
    
    #TODO: How to separate bvec/bvalue pairs for the DWI generation?
    workflow.connect(input_node, 'in_bval_file', distortion_generator, 'bval_file')
    workflow.connect(input_node, 'in_bvec_file', distortion_generator, 'bvec_file')
    # Resample tensors
    workflow.connect(input_node, 'in_tensors_file', tensor_resampling, 'ref_file')
    workflow.connect(input_node, 'in_tensors_file', tensor_resampling, 'flo_file')
    workflow.connect(distortion_generator, 'aff_files', tensor_resampling, 'trans_file')
    # Resample B0s the same way as the distorted tensor
    workflow.connect(input_node, 'in_B0_file', b0_resampling, 'flo_file')
    workflow.connect(input_node, 'in_B0_file', b0_resampling, 'ref_file')
    workflow.connect(distortion_generator, 'aff_files', tensor_resampling, 'trans_file')
    
    # Make distortedDWI using the the affine distorted tensors and B0s
    workflow.connect(tensor_resampling, 'res_file', tensor_2_dwi, 'source_file')
    workflow.connect(input_node, 'in_bvec_file',    tensor_2_dwi, 'bvec_file')
    workflow.connect(input_node, 'in_bval_file',    tensor_2_dwi, 'bval_file')
    workflow.connect(input_node, 'in_B0_file',      tensor_2_dwi, 'b0_file')
    
    #TODO: Need to make rigidly distorted B0s for 'psuedo-observations'
    
    #TODO: Need to add noise to the B0 and the distorted DWI!

    # Merge distorted DWI
    workflow.connect(tensor_2_dwi, 'syn_file', merge_dwis, 'in_files')
    
    #TODO:  Can we merge DWI and b0 images at the same time?!?
    workflow.connect(merge_dwis, 'merged_file', r, 'input_node.in_dwi_4d_file')
    
    # Now perform the diffusion pre-processing pipeline
    workflow.connect(input_node, 'in_bvec_file', r, 'input_node.in_bvec_file')
    workflow.connect(input_node, 'in_bval_file', r, 'input_node.in_bval_file')
    workflow.connect(input_node, 'in_T1_file', r, 'input_node.in_T1_file')
    
    # Take the final estimated transformation of the DWI, and invert the transformation
    workflow.connect(r, 'output_node.transformations', inv_estimated_distortions, 'inv_aff_input')
    
    
    # Resample the tensor to the space of each of the observed DWI
    workflow.connect(r, 'output_node.tensor', tensor_resampling_2, 'ref_file')
    workflow.connect(r, 'output_node.tensor', tensor_resampling_2, 'flo_file')
    workflow.connect(inv_estimated_distortions, 'out_file', tensor_resampling_2, 'trans_file')

    # Resample the averageB0 to the sapce of the observed DWI for prediction    
    workflow.connect(r, 'output_node.average_b0', b0_resampling_2, 'ref_file')
    workflow.connect(r, 'output_node.average_b0', b0_resampling_2, 'flo_file')
    workflow.connect(inv_estimated_distortions, 'out_file', b0_resampling_2, 'trans_file')
    
    # Predict the DWI for this particular b-vector/b-value pair, ignore 
    workflow.connect(tensor_resampling_2, 'res_file', tensor_2_dwi_2, 'source_file')
    workflow.connect(input_node, 'in_bvec_file', tensor_2_dwi_2, 'bvec_file')
    # TODO: Need to pass the correct bval/bvec pair
    workflow.connect(input_node, 'in_bval_file', tensor_2_dwi_2, 'bval_file')
    workflow.connect(b0_resampling_2, 'res_file', tensor_2_dwi_2, 'B0_file')
    
    workflow.connect(tensor_2_dwi_2, 'syn_file', merge_dwis_2, 'in_files')
    workflow.connect(merge_dwis,'merged_file',output_node, 'simulated_dwis')
    workflow.connect(merge_dwis_2, 'merged_file', output_node, 'estimated_dwis')
    workflow.connect(input_node, 'in_tensors_file', output_node, 'simulated_tensors')
    workflow.connect(r, 'output_node.tensor', output_node, 'estimated_tensors')
    
    return workflow
    

    # node explanation:
    
