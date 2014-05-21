#! /usr/bin/env python

import nipype.interfaces.utility as niu     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.niftyreg as niftyreg

'''This file provides some common registration routines useful for a variety of pipelines. Including linear
and non-linear image co-registration '''

# Do a single iteration of an average b0 image from rigid registration and averaging
# Options include rig_only 
# TODO:Aladin options hash?
def create_linear_coregistration_workflow(name="linear_registration_niftyreg", rig_only=False):
    # We need to create an input node for the workflow    
    input_node = pe.Node(niu.IdentityInterface(
            fields=['in_files', 'ref_file']),
                        name='input_node')
    
    # Rigidly register each of the images to the average
    # flo_file can take a list of files
    lin_reg = pe.MapNode(interface=niftyreg.RegAladin(), name="lin_reg", iterfield=['flo_file'])
        
    # Select whether to do rigid registration only
    lin_reg.inputs.rig_only_flag = rig_only

    # Average the images
    ave_ims = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims")
    
    # We have a new centered average image, the resampled original images and the affine 
    # transformations, which are returned as an output node. 
    output_node = pe.Node(
        niu.IdentityInterface(
            fields=['average_image', 'aff_files']),
                        name='output_node')

    pipeline = pe.Workflow(name=name)
    pipeline.base_output_dir=name

    # Connect the inputs to the lin_reg node, which is split over in_files
    pipeline.connect([(input_node, lin_reg,[('in_files','flo_file')]),
                     (input_node, lin_reg,[('ref_file', 'ref_file')]),
                     (input_node, lin_reg,[('rmask_file','rmask_file')])])
    
    if initial_affines == True:
        pipeline.connect(input_node, 'in_aff_files', lin_reg,'in_aff_file')
    

    # Join the outputs from lin_reg (as conveniently outputted from the RegAladin wrapper) 
    # and pass to ave_ims   
    pipeline.connect([(lin_reg, ave_ims, [('avg_output', 'demean_files')]),
                      (input_node, ave_ims, [('ref_file','demean1_ref_file')])])
                    
    # Connect up the output node
    pipeline.connect([(lin_reg, output_node,[('aff_file', 'aff_files')]),
                      (ave_ims, output_node,[('out_file', 'average_image')])
                      ])
    return pipeline


def create_nonlinear_coregistration_workflow(name="nonlinear_registration_niftyreg"):
    pipeline = pe.Workflow(name=name)
    
    return pipeline

    
