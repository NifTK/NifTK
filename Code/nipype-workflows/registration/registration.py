#! /usr/bin/env python

import nipype.interfaces.utility as niu     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.niftyreg as niftyreg
from nipype.interfaces.base import isdefined

'''This file provides some common registration routines useful for a variety of pipelines. Including linear
and non-linear image co-registration '''

# Do a single iteration of an average b0 image from rigid registration and averaging
# Options include rig_only 
# TODO:Aladin options hash?
def create_linear_coregistration_workflow(name="linear_registration_niftyreg", demean=True, linear_options_hash = None, initial_affines = False):
    """Creates a workflow that perform linear co-registration of a set of images using RegAladin, 
    producing an affine average image and a set of affine transformation matrices linking each
    of the floating images to the average.

    Example
    -------
    >>> linear_coreg = create_linear_coregistration_workflow('my_linear_coreg')
    >>> linear_coreg.inputs.input_node.in_files = ['file1.nii.gz', 'file2.nii.gz']
    >>> linear_coreg.inputs.input_node.ref_file = ['initial_ref']
    >>> linear_coreg.inputs.input_node
    >>> linear_coreg.run() # doctest: +SKIP

    Inputs::

        input_node.in_files - The input files to be registered
        input_node.ref_file - The initial reference image that the input files are registered to
        input_node.rmask_file - Mask of the reference image
        input_node.in_aff_files - Initial affine transformation files
        

    Outputs::

        output_node.average_image - The average image
        output_node.aff_files - The affine transformation files


    Optional arguments::
        linear_options_hash - An options dictionary containing a list of parameters for RegAladin that take the same form as given in the interface (default None) 
        demean - Selects whether to demean the transformation matrices when performing the averaging (default True)
        initial_affines - Selects whether to iterate over initial affine images, which we generally won't have (default False)


    """
    # We need to create an input node for the workflow    
    input_node = pe.Node(niu.IdentityInterface(
            fields=['in_files', 'ref_file', 'rmask_file', 'in_aff_files']),
                        name='input_node')
    
    # Rigidly register each of the images to the average
    # flo_file can take a list of files
    # Need to be able to iterate over input affine files, but what about the cases where we have no input affine files?
    # Passing empty strings are not valid filenames, and undefined fields can not be iterated over.
    # Current simple solution, as this is not generally required, is to use a flag which specifies wherther to iterate
    if initial_affines == False:
        lin_reg = pe.MapNode(interface=niftyreg.RegAladin(options_hash=linear_options_hash), name="lin_reg", iterfield=['flo_file'])
    else:
        lin_reg = pe.MapNode(interface=niftyreg.RegAladin(options_hash=linear_options_hash), name="lin_reg", iterfield=['flo_file','in_aff_file'])
    # Synchronize over the iterfields    
    lin_reg.synchronize = True

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

# Creates an atlas image by iterative registration. An initial reference image can be provided, otherwise one will be made. 
#
def create_atlas(name="atlas_creation", itr_rigid = 1, itr_affine = 1, itr_nl = 1, initial_ref = True, linear_options_hash=None, nonlinear_options_hash=None):
    pipeline = pe.Workflow(name=name)
    input_node = pe.Node(niu.IdentityInterface(
            fields=['in_files', 'ref_file']),
                        name='input_node')
    
    output_node = pe.Node(
        niu.IdentityInterface(
            fields=['average_image', 'aff_files']),
                        name='output_node')
                        
    lin_workflows = []
    nonlin_workflows = []
    if linear_options_hash == None:
        linear_options_hash = dict()
    # Assume we're doing rigid, but change it after the rigid iterations are done
    demean_arg = True
    initial_affines_arg = False
    for i in range(itr_rigid+itr_affine):
        w = None
        
        if( i >= itr_rigid ):
            linear_options_hash['rig_only_flag'] = False
        if( i < itr_rigid ):
            linear_options_hash['rig_only_flag'] = True
        if (i == (itr_rigid-1)) or (i == (itr_affine-1)):
            demean_arg = False
        else:
            demean_arg = True
        if i > 0:
            initial_affines_arg = True
        w = create_linear_coregistration_workflow('lin_reg'+str(i), linear_options_hash=linear_options_hash\
        , initial_affines=initial_affines_arg, demean=demean_arg)
        lin_workflows.append(w)
        # Connect up the input data to the workflows
        pipeline.connect(input_node, 'in_files', w, 'input_node.in_files')
        
        if i > 0:
            pipeline.connect(lin_workflows[i-1], 'output_node.average_image', w, 'input_node.ref_file' )
            pipeline.connect(lin_workflows[i-1], 'output_node.aff_files', w, 'input_node.in_aff_files' )
            
        
    # Make the nonlinear coregistration workflows
    '''for i in range(itr_nl):
        w = None
        if i < (itr_nl - 1):
            w = create_nonlinear_coregistration_workflow('nonlin'+str(i), nonlinear_options_hash=nonlinear_options_hash)
        else:
            w = create_nonlinear_coregistration_workflow('nonlin'+str(i), demean=False, nonlinear_options_hash=nonlinear_options_hash)
        # Connect up the input data to the workflows
        pipeline.connect(input_node, 'in_files', w, 'input_node.in_files')
        # Take the final linear registration results and use them to initialise the NR
        pipeline.connect(lin_workflows[len(lin_workflows)-1], 'output_node.aff_files', w, 'input_node.aff_files' )
        nonlin_workflows.append(w)'''
       
    # Set the reference image if we have one, else make a node to generate
    # it and connect it up
    if initial_ref == False:
        ave_ims = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims_initial")
        pipeline.connect(input_node, 'data', ave_ims, 'in_files')
        pipeline.connect(ave_ims, 'out_file', lin_workflows[0], 'input_node.ref_file')
    else:
        pipeline.connect(input_node, 'ref_file', lin_workflows[0], 'input_node.ref_file')
        
    pipeline.connect(lin_workflows[len(lin_workflows)-1], 'output_node.aff_files', output_node, 'aff_files')
    pipeline.connect(lin_workflows[len(lin_workflows)-1], 'output_node.average_image', output_node, 'average_image')
            
    
    return pipeline


def create_nonlinear_coregistration_workflow(name="nonlinear_registration_niftyreg"):
    pipeline = pe.Workflow(name=name)
    
    return pipeline

    
