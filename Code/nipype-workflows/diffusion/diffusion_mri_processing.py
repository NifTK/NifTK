#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.susceptibility as susceptibility

from groupwise_registration_workflow    import *
from susceptibility_correction_workflow import *

'''
This file provides the creation of the whole workflow necessary for 
processing diffusion MRI images.
'''

def create_diffusion_mri_processing_workflow(name='diffusion_mri_processing'):

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
    split_dwis = pe.Node(interface = fsl.Split(), name = 'split_dwis')
    
    #Node using niu.Select() to select only the B0 files
    select_B0s = pe.Node(interface = niu.Select(), name = 'select_B0s')
    #select_B0s.inputs.index=?
    select_DWIs = pe.Node(interface = niu.Select(), name = 'select_DWIs')
    #select_DWIs.inputs.index=?
    groupwise_B0_coregistration = create_linear_coregistration_workflow('groupwise_B0_coregistration')
    
    susceptibility_correction = create_susceptibility_correction_workflow('susceptibility_correction')

    dwi_to_B0_registration = pe.Node(niftyreg.RegAladin(), name = 'dwi_to_B0_registration', iterfield=['flo_file'])
    
    #do the node for the gradient reorientation
    gradient_reorientation = pe.Node()
    
    #do the join node for composing a linear transformation and a deformation field
    transformation_composition = pe.Node(niftyreg.RegTransform(), name = 'transformation_composition', iterfield=['in_comp_transformation_file1'])
    
    #do the node for resampling
    resampling = pe.Node(niftyreg.RegResample(), name = 'resampling', iterfield=['trans_file', 'flo_file'])

    #do the node for merging all the files into one
    merge_dwis = pe.Node(interface = niu.Select(), name = 'merge_dwis')
    
    workflow.connect([input_node, split_dwis,                  ['in_dwi_4d_file', 'in_file']))
    workflow.connect([split_dwis, select_B0s,                  ['out_files',      'inlist']))
    workflow.connect([select_B0s, groupwise_B0_coregistration, ['out',            'inputnode.in_files']))

    workflow.connect([groupwise_B0_coregistration, susceptibility_correction, ['outputnode.mean_image','inputnode.in_B0_file']))
    workflow.connect([inputnode, susceptibility_correction,                   ['in_T1_file', 'inputnode.in_T1_file']))

    workflow.connect([groupwise_B0_coregistration, dwi_to_B0_registration,    ['outputnode.mean_image','ref_file']))
    workflow.connect([select_DWIs, dwi_to_B0_registration,     ['out','flo_file']))

    workflow.connect([groupwise_B0_coregistration, transformation_composition, ['out_transformation',   'in_comp_transformation_file1']))
    workflow.connect([dwi_to_B0_registration,      transformation_composition, ['out_transformation',   'in_comp_transformation_file1']))
    workflow.connect([susceptibility_correction,   transformation_composition, ['out_deformation_field','in_comp_transformation_file2']))
    
    workflow.connect([groupwise_B0_coregistration, resampling, ['mean_image',              'ref_file']))
    workflow.connect([split_dwis, resampling,                  ['out_files',               'flo_file']))
    workflow.connect([transformation_composition, resampling,  ['out_comp_transformation', 'trans_file']))

    workflow.connect([resampling, merge_dwis,                  ['res_file', 'in_file']))

    workflow.connect([merge_dwis, tensor_fitting,              ['out_file',     'in_dwi_file']))
    workflow.connect([inputnode,  tensor_fitting,              ['in_bvec_file', 'in_bvec_file']))
    workflow.connect([inputnode,  tensor_fitting,              ['in_bval_file', 'in_bvec_file']))
