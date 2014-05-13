#! /usr/bin/env python

import nipype.interfaces.utility as niu     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg


'''This file provides some common registration routines useful for a variety of pipelines. Including linear
and non-linear image registration '''

# Do a single iteration of an average b0 image from rigid registration and averaging
# Options include rig_only 
# TODO:Aladin options hash?
def create_linear_coregistration_workflow(name="b0_registration_niftyreg", rig_only=False):
    # We need to create an input node for the workflow    
    input_node = pe.Node(niu.IdentityInterface(
            fields=['in_files', 'ref_file']),
                        name='inputnode')
    
    # Average the images    
    
    # Rigidly register each of the images to the average
    lin_reg = pe.MapNode(interface=niftyreg.RegAladin, name="lin_reg", iterfield=['flo_file'])
    # flo_file can now take a list of files, pass that list
    # List of input files which this node is iterated over
    lin_reg.iterables('flo_file', ["file1.nii.gz, file2.nii.gz"])
        
    # Affine output file
    lin_reg.inputs.rig_only_flag = rig_only    
    
    resamp_im = pe.Node(interface=niftyreg.RegResample, name='reg_resamp')
    
    

    # Join node to make string to pass to reg_average. Needs to contain ref image, transformation files and floating image files
    merge_interface = niu.Merge(2)
    
    # Merge the filenames from lin_reg for passing to reg_average
    merge_filenames = pe.Node(interface=merge_interface, name='merge_filenames')
    

    # Average the images, this needs to be a join node as the lin-reg node is iterable
    ave_ims = pe.JoinNode(interface=niftyreg.RegAverage, name="ave_ims", joinfield='in_files', joinsource='lin_reg')
    
    
    # We have a new centered average image, the resampled original images and the affine 
    # transformations
    output_node = pe.Node(
        niu.IdentityInterface(
            fields=['resampled_images', 'average_image', 'aff_files']),
                        name='outputnode')

    

    pipeline = pe.Workflow(name=name)
    # Connect the pipeline
    pipeline.connect([(input_node, lin_reg,['in_files','flo_file']),
                     (input_node, lin_reg,['ref_file', 'ref_file']),
                     (lin_reg, output_node,['aff_file', 'aff_files']),
                     (lin_reg, resamp_im, ['aff_file', 'trans_file']),
                        (lin_reg, resamp_im, ['flo_file', 'flo_file']),
                    (lin_reg, resamp_im, ['ref_file', 'ref_file']),
                    (resamp_im, ave_ims, ['res_file', 'in_files']),
                    (lin_reg, merge_filenames, ['aff_file', 'in1']),
                    ()
                    (ave_affs, inv_avg_aff, ['out_file','inv_aff_val[0]']),
                    (ave_ims, resamp_ave, ['out_file', 'flo_file']),
                    (ave_ims, resamp_ave, ['out_file', 'trans_file'])
                    ])
    
    
    
    return reg_worflow
    
