#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyfit       as niftyfit
import nipype.interfaces.fsl            as fsl

import inspect

from registration import *

'''
This file provides the creation of the whole workflow necessary for 
processing ASL MRI images.
'''

def pop_first_item_function(in_list):
    out_list = list(in_list).pop(0)
    return out_list


def create_asl_processing_workflow(name='asl_processing',
                                   asl_interp_type = 'CUB'):

    """Creates a ASL processing workflow. Each of the ASL is registered to the M0 map.
    
    Example
    -------

    >>> asl_proc = create_asl_processing_workflow(name='asl_proc')
    >>> asl_proc.inputs.in_source = aslsequence.nii.gz
    >>> asl_proc.run()    

    Inputs::

        input_node.in_source - The original 4D DWI image file
           
    Outputs::

        output_node.out_cbf

    Optional arguments::

    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_source']),
        name='input_node')
    
    #Node using fslsplit() to split the 4D file, separate the B0 and DWIs
    split_asls = pe.Node(interface = fsl.Split(dimension="t"), name = 'split_asls')
    
    # Node using fslsplit to get the m0 map   
    select_m0map = pe.Node(interface = niu.Select(index = 0), name = 'select_m0map')
    
    # Node using fslsplit to get the m0 map   
    select_aslmaps = pe.Node(interface = niu.Function(input_names = ['in_list'], 
                                                      output_names = ['out_list'],
                                                      function = pop_first_item_function), 
                                      name = 'select_aslmaps')
    
    # As we're trying to estimate an affine transformation, and rotations and shears are confounded
    # easier just to optimise an affine directly for the DWI 
    asl_to_m0_registration = pe.MapNode(niftyreg.RegAladin(aff_direct_flag = True), 
                                        name = 'asl_to_m0_registration',
                                        iterfield=['flo_file'])
    
    # Resample the DWI and B0s
    resampling = pe.MapNode(niftyreg.RegResample(), name = 'resampling', iterfield=['trans_file', 'flo_file'])
    resampling.inputs.inter_val = asl_interp_type
    
    # Remerge all the asls
    merge_asls = pe.Node(interface = fsl.Merge(dimension = 't'), name = 'merge_asls')
    
    # Fit the model
    asl_model_fitting = pe.Node(interface=niftyfit.FitAsl(),name='asl_model_fitting')
    asl_model_fitting.inputs.pasl = True
    asl_model_fitting.inputs.Tinv1 = 800.0
    asl_model_fitting.inputs.Tinv2 = 50.5722
    asl_model_fitting.inputs.mul = 1.0
    asl_model_fitting.inputs.out = 2.5

    # Output node
    output_node = pe.Node( interface=niu.IdentityInterface(fields=['cbf_file', 'error_file', 'syn_file']),
                           name="output_node" )

    #############################################################
    # Find the ASL data and separate between ASL and M0 #
    #############################################################

    workflow.connect(input_node, 'in_source', split_asls, 'in_file')
    workflow.connect(split_asls, 'out_files', select_m0map, 'inlist')
    workflow.connect(split_asls, 'out_files', select_aslmaps, 'in_list')

    #############################################################
    #             ASL to M0 affine registration                 #
    #############################################################

    workflow.connect(select_m0map, 'out', asl_to_m0_registration, 'ref_file')
    workflow.connect(split_asls, 'out_files', asl_to_m0_registration, 'flo_file')
    
    #############################################################
    #   Resample the ASL with affine                            #
    #   transformations and merge back into a 4D image          #
    #############################################################
    
    workflow.connect(select_m0map, 'out', resampling, 'ref_file')
    workflow.connect(select_aslmaps, 'out_list', resampling, 'flo_file')
    workflow.connect(asl_to_m0_registration, 'aff_file', resampling, 'trans_file')
    
    workflow.connect(resampling, 'res_file',   merge_asls, 'in_files')

    #############################################################
    #   Fit the ASL model from the ASL data         #
    #############################################################
    
    workflow.connect(merge_asls, 'merged_file', asl_model_fitting, 'source_file')
    workflow.connect(select_m0map, 'out', asl_model_fitting, 'm0map')
    
    #############################################################
    #         Prepare data for output_node                      #
    #############################################################

    workflow.connect(asl_model_fitting, 'cbf_file', output_node, 'cbf_file')
    workflow.connect(asl_model_fitting, 'cbf_file', output_node, 'error_file')
    workflow.connect(asl_model_fitting, 'cbf_file', output_node, 'syn_file')
    
    return workflow    

    
