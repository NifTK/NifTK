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
def create_linear_coregistration_workflow(name="linear_registration_niftyreg", 
                                          demean=True, 
                                          linear_options_hash = dict(), 
                                          initial_affines = False):
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
            fields=['in_files', 
                    'ref_file', 
                    'rmask_file', 
                    'in_aff_files']),
                        name='input_node')
    
    # Rigidly register each of the images to the average
    # flo_file can take a list of files
    # Need to be able to iterate over input affine files, but what about the cases where we have no input affine files?
    # Passing empty strings are not valid filenames, and undefined fields can not be iterated over.
    # Current simple solution, as this is not generally required, is to use a flag which specifies wherther to iterate
    if initial_affines == False:
        lin_reg = pe.MapNode(interface=niftyreg.RegAladin(**linear_options_hash), name="lin_reg", iterfield=['flo_file'])
    else:
        lin_reg = pe.MapNode(interface=niftyreg.RegAladin(**linear_options_hash), name="lin_reg", iterfield=['flo_file','in_aff_file'])
    # Synchronize over the iterfields    
    lin_reg.synchronize = True

    # Average the images
    ave_ims = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims")
    
    # We have a new centered average image, the resampled original images and the affine 
    # transformations, which are returned as an output node. 
    output_node = pe.Node(niu.IdentityInterface(
        fields=['average_image', 
                'trans_files']),
                          name='output_node')
    
    pipeline = pe.Workflow(name=name)
    pipeline.base_output_dir=name

    # Connect the inputs to the lin_reg node, which is split over in_files
    pipeline.connect([(input_node, lin_reg,[('in_files','flo_file')]),
                     (input_node, lin_reg,[('ref_file', 'ref_file')]),
                     (input_node, lin_reg,[('rmask_file','rmask_file')])])
    
    # If we have initial affine transforms, we need to connect them in
    if initial_affines == True:
        pipeline.connect(input_node, 'in_aff_files', lin_reg,'in_aff_file')
    
    if demean == True:   
        pipeline.connect(input_node, 'ref_file', ave_ims, 'demean1_ref_file')
    else:
        pipeline.connect(input_node, 'ref_file', ave_ims, 'avg_tran_ref_file')
    
    # Either way we do the averaging, we need to connect the files in
    # Join the outputs from lin_reg (as conveniently outputted from the RegAladin wrapper) 
    # and pass to ave_ims
    pipeline.connect(lin_reg, 'avg_output',ave_ims, 'demean_files')
                    
    # Connect up the output node
    pipeline.connect([(lin_reg, output_node,[('aff_file', 'trans_files')]),
                      (ave_ims, output_node,[('out_file', 'average_image')])
                      ])
    return pipeline


def create_nonlinear_coregistration_workflow(name="nonlinear_registration_niftyreg", 
                                             demean=True, 
                                             nonlinear_options_hash = dict(), 
                                             initial_affines = False, 
                                             initial_cpps = False):
    """Creates a workflow that perform non-linear co-registration of a set of images using RegF3d, 
    producing an non-linear average image and a set of cpp transformation linking each
    of the floating images to the average.

    Example
    -------
    >>> nonlinear_coreg = create_nonlinear_coregistration_workflow('my_linear_coreg')
    >>> nonlinear_coreg.inputs.input_node.in_files = ['file1.nii.gz', 'file2.nii.gz']
    >>> nonlinear_coreg.inputs.input_node.ref_file = ['initial_ref']
    >>> nonlinear_coreg.inputs.input_node
    >>> nonlinear_coreg.run() # doctest: +SKIP

    Inputs::

        input_node.in_files - The input files to be registered
        input_node.ref_file - The initial reference image that the input files are registered to
        input_node.rmask_file - Mask of the reference image
        input_node.in_trans_files - Initial transformation files (affine or cpps)
        

    Outputs::

        output_node.average_image - The average image
        output_node.cpp_files - The bspline transformation files


    Optional arguments::
        nonlinear_options_hash - An options dictionary containing a list of parameters for RegAladin that take the same form as given in the interface (default None) 
        initial_affines - Selects whether to iterate over initial affine images, which we generally won't have (default False)


    """
    # We need to create an input node for the workflow    
    input_node = pe.Node(niu.IdentityInterface(
        fields=['in_files', 
                'ref_file', 
                'rmask_file', 
                'in_trans_files']),
                         name='input_node')
    
    # Rigidly register each of the images to the average
    # flo_file can take a list of files
    # Need to be able to iterate over input affine files, but what about the cases where we have no input affine files?
    # Passing empty strings are not valid filenames, and undefined fields can not be iterated over.
    # Current simple solution, as this is not generally required, is to use a flag which specifies wherther to iterate
    if initial_affines == True:
        nonlin_reg = pe.MapNode(interface=niftyreg.RegF3d(**nonlinear_options_hash), 
                                name="nonlin_reg", 
                                iterfield=['flo_file','aff_file'])
    elif initial_cpps == True:
        nonlin_reg = pe.MapNode(interface=niftyreg.RegF3d(**nonlinear_options_hash), 
                                name="nonlin_reg", 
                                iterfield=['flo_file','incpp_file'])
    else:
        nonlin_reg = pe.MapNode(interface=niftyreg.RegF3d(**nonlinear_options_hash), 
                                name="nonlin_reg",
                                iterfield=['flo_file'])
            
    # Average the images
    ave_ims = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims")
    
    # We have a new centered average image, the resampled original images and the affine 
    # transformations, which are returned as an output node. 
    output_node = pe.Node( niu.IdentityInterface(
        fields=['average_image', 
                'trans_files']),
                           name='output_node')

    pipeline = pe.Workflow(name=name)
    pipeline.base_output_dir=name

    # Connect the inputs to the lin_reg node, which is split over in_files
    pipeline.connect([(input_node, nonlin_reg,[('in_files','flo_file')]),
                      (input_node, nonlin_reg,[('ref_file', 'ref_file')]),
                      (input_node, nonlin_reg,[('rmask_file','rmask_file')])])
    
    # If we have initial affine transforms, we need to connect them in
    if initial_affines == True:
        pipeline.connect(input_node, 'in_trans_files', nonlin_reg,'aff_file')
    elif initial_cpps == True:
        pipeline.connect(input_node, 'in_trans_files', nonlin_reg,'incpp_file')
    
    if demean == True:   
        pipeline.connect(input_node, 'ref_file', ave_ims, 'demean2_ref_file')
    else:
        pipeline.connect(input_node, 'ref_file', ave_ims, 'avg_tran_ref_file')
    
    # Either way we do the averaging, we need to connect the files in
    # Join the outputs from lin_reg (as conveniently outputted from the RegAladin wrapper) 
    # and pass to ave_ims
    pipeline.connect(nonlin_reg, 'avg_output',ave_ims, 'demean_files')
    
    # Connect up the output node
    pipeline.connect([(nonlin_reg, output_node,[('cpp_file', 'trans_files')]),
                      (ave_ims, output_node,[('out_file', 'average_image')])
                      ])
    return pipeline


# Creates an atlas image by iterative registration. An initial reference image can be provided, otherwise one will be made. 
#
def create_atlas(name="atlas_creation", 
                 itr_rigid = 1, 
                 itr_affine = 1, 
                 itr_non_lin = 1, 
                 initial_ref = True, 
                 linear_options_hash=dict(), 
                 nonlinear_options_hash=dict()):

    pipeline = pe.Workflow(name=name)

    input_node = pe.Node(niu.IdentityInterface(
            fields=['in_files', 
                    'ref_file']),
                        name='input_node')
    
    output_node = pe.Node(niu.IdentityInterface(
        fields=['average_image', 
                'trans_files']),
                          name='output_node')
                        
    lin_workflows = []
    nonlin_workflows = []
    
    demean_arg = None
    initial_affines_arg = False

    for i in range(itr_rigid+itr_affine):
        w = None
        
        if( i >= itr_rigid ):
            linear_options_hash['rig_only_flag'] = False
        if( i < itr_rigid ):
            linear_options_hash['rig_only_flag'] = True
        if (i < (itr_rigid-1)) or (i == (itr_affine-1)):
            demean_arg = False
        else:
            demean_arg = True
        if i > 0:
            initial_affines_arg = True
        w = create_linear_coregistration_workflow(name = 'lin_reg'+str(i), 
                                                  linear_options_hash = linear_options_hash,
                                                  initial_affines = initial_affines_arg, 
                                                  demean = demean_arg)
        lin_workflows.append(w)
        # Connect up the input data to the workflows
        pipeline.connect(input_node, 'in_files', w, 'input_node.in_files')
        
        if i > 0:
            pipeline.connect(lin_workflows[i-1], 'output_node.average_image', w, 'input_node.ref_file' )
            pipeline.connect(lin_workflows[i-1], 'output_node.trans_files', w, 'input_node.in_aff_files' )
            
    
    demean_arg = True
    initial_affines_arg = False
    initial_cpps_arg = False

    # Make the nonlinear coregistration workflows
    for i in range(itr_non_lin):
        w = None

        if i == 0:
            if len(lin_workflows) > 0:
                initial_affines_arg = True
        else:
            initial_cpps_arg = True
        if (i == (itr_non_lin - 1)):
            demean_arg = False
        
        w = create_nonlinear_coregistration_workflow(name = 'nonlin'+str(i), 
                                                     demean = demean_arg, 
                                                     initial_affines = initial_affines_arg, 
                                                     initial_cpps = initial_cpps_arg, 
                                                     nonlinear_options_hash = nonlinear_options_hash)
        
        # Connect up the input data to the workflows
        pipeline.connect(input_node, 'in_files', w, 'input_node.in_files')

        if i == 0:
            if (len(lin_workflows)):
                # Take the final linear registration results and use them to initialise the NR
                pipeline.connect(lin_workflows[len(lin_workflows)-1], 'output_node.aff_files', w, 'input_node.in_trans_files' )
        else:
            pipeline.connect(nonlin_workflows[i-1], 'output_node.average_image', w, 'input_node.ref_file' )
            pipeline.connect(nonlin_workflows[i-1], 'output_node.trans_files', w, 'input_node.in_trans_files' )

        nonlin_workflows.append(w)
       

    first_workflow = None
    if len(lin_workflows) > 0:
        first_workflow = lin_workflows[0]
    elif len(nonlin_workflows) > 0:
        first_workflow = nonlin_workflows[0]

    last_workflow = None
    if len(nonlin_workflows) > 0:
        last_workflow = nonlin_workflows[len(nonlin_workflows)-1]
    elif len(lin_workflows) > 0:
        last_workflow = lin_workflows[len(lin_workflows)-1]
    
    if initial_ref == True:
        pipeline.connect(input_node, 'ref_file', first_workflow, 'input_node.ref_file')
    else:
        ave_ims = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims_initial")
        pipeline.connect(input_node, 'in_files', ave_ims, 'in_files')
        pipeline.connect(ave_ims, 'out_file', first_workflow, 'input_node.ref_file')
        
    pipeline.connect(last_workflow, 'output_node.trans_files', output_node, 'trans_files')
    pipeline.connect(last_workflow, 'output_node.average_image', output_node, 'average_image')
    
    return pipeline

    
