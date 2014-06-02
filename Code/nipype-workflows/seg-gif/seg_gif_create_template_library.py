#! /usr/bin/env python


import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl

import nipype.interfaces.io             as nio
import inspect

import cropimage as cropimage

import registration as reg
import seg_gif_propagation as gif_propagation

'''
'''


def create_subdir(base_dir, input_filename):
    import os
    basename = os.path.basename(input_filename)
    if basename.endswith('.nii.gz'):
        basename = basename[:-7]
    if basename.endswith('.nii'):
        basename = basename[:-4]
    ret = os.path.join(base_dir, basename)
    if not os.path.exists(ret):
        os.mkdir(ret)
    return ret


def get_subs1(in_paths, out_paths):
    import os
    subs = []

    commonpath = os.path.commonprefix(out_paths)

    for i in range(len(out_paths)):
       basename = os.path.basename(in_paths[i])
       _, filepath = out_paths[i].split(commonpath,1)
       subs.append(('_T1_crop'+filepath, basename))

    return subs

def get_iters(filelist):
    import os
    iters = []
    input_names = ['ref_file', 'flo_file']
    
    for input_name in input_name:
        iters.append((input_name, filelist))

    return iters



def create_seg_gif_preproc(name='seg_gif_preproc'):

    """

    Creates a pipeline that uses seg_GIF label propagation to propagate 
    segmentation towards a target image
    
    Example
    -------

    >>> gif = create_niftyseg_gif_propagation_pipeline("niftyseg_gif")
    >>> gif.inputs.inputnode.input_image = 'T1.nii'
    >>> gif.run()                  # doctest: +SKIP

    Inputs::

        inputnode.input_image

    Outputs::


    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_entries_directory',
                    'out_T1_directory',
                    'out_cpps_directory']),
        name='input_node')

    # create a datagrabber that will take care of extracting all these guys
    grabber = pe.Node(interface = nio.DataGrabber(outfields=['T1s_paths']), name = 'grabber')
    grabber.inputs.template = '*.nii*'
    grabber.inputs.sort_filelist = False

    groupwise_coregistration = reg.create_atlas(name = 'groupwise_coregistration', itr_rigid = 0, itr_affine = 1, itr_nl = 0, initial_ref = False)

    sformupdate = pe.MapNode(interface = niftyreg.RegTransform(), name = 'sformupdate', synchronize = True, iterfield = ['upd_s_form_input', 'upd_s_form_input2'])

    average_mask = pe.Node(interface=fsl.BET(), name='average_mask')
    average_mask.inputs.mask = True
    
    average_mask_dil = pe.Node(interface = niftyseg.BinaryMaths(), name = 'average_mask_dil')
    average_mask_dil.inputs.operation = 'dil'
    average_mask_dil.inputs.operand_value = 10

    average_crop = pe.Node(interface = cropimage.CropImage(), name='average_crop')

    T1_crop = pe.MapNode(interface = cropimage.CropImage(), name='T1_crop', iterfield = ['in_file'])

    T1_sink = pe.Node(nio.DataSink(), name='T1_sink')

    non_linear_registration = pe.Node(interface=niftyreg.RegF3D(), name = 'non_linear_registration', synchronize = False)
    non_linear_registration.inputs.output_type = 'NIFTI_GZ'

    cpp_sink = pe.MapNode(nio.DataSink(), name='cpp_sink', iterfield = ['base_directory'])

    makesubdir = pe.MapNode(niu.Function(input_names=['base_dir', 'input_filename'],
                                         output_names=['cpp_path'],
                                         function=create_subdir),
                            name='makesubdir',
                            iterfield = ['input_filename'])

    subsgen = pe.Node(niu.Function(input_names=['in_paths', 'out_paths'],
                                    output_names=['substitutions'],
                                    function=get_subs1),
                          name='subsgen')

    itersgen = pe.Node(niu.Function(input_names=['filelist'],
                                    output_names=['iters'],
                                     function=get_iters),
                          name='itersgen')

    workflow.connect(input_node, 'in_entries_directory', grabber, 'base_directory')

    workflow.connect(grabber, 'T1s_paths', groupwise_coregistration, 'input_node.in_files')

    workflow.connect(grabber,                  'T1s_paths',             sformupdate, 'upd_s_form_input')
    workflow.connect(groupwise_coregistration, 'output_node.aff_files', sformupdate, 'upd_s_form_input2')

    workflow.connect(groupwise_coregistration, 'output_node.average_image', average_mask, 'in_file')

    workflow.connect(average_mask,                         'mask_file',  average_mask_dil, 'in_file')

    workflow.connect(average_mask_dil,         'out_file',                  average_crop, 'mask_file')
    workflow.connect(groupwise_coregistration, 'output_node.average_image', average_crop, 'in_file')

    workflow.connect(sformupdate,  'out_file', T1_crop, 'in_file')
    workflow.connect(average_crop, 'out_file', T1_crop, 'mask_file')

    workflow.connect(grabber, 'T1s_paths', subsgen, 'in_paths')
    workflow.connect(T1_crop, 'out_file',  subsgen, 'out_paths')

    workflow.connect(subsgen,    'substitutions',    T1_sink, 'substitutions')
    workflow.connect(T1_crop,    'out_file',         T1_sink, '@T1s')
    workflow.connect(input_node, 'out_T1_directory', T1_sink, 'base_directory')

    workflow.connect(input_node, 'out_cpps_directory', makesubdir, 'base_dir')
    workflow.connect(grabber,    'T1s_paths',          makesubdir, 'input_filename')

    workflow.connect(T1_sink, 'out_file',  itersgen, 'filelist')

    workflow.connect(itersgen, 'iters',  non_linear_registration, 'iterables')

    workflow.connect(makesubdir,              'cpp_path',      cpp_sink, 'base_directory')
    workflow.connect(non_linear_registration, 'cpp_file',      cpp_sink, '@cpps')
    workflow.connect(subsgen1,                'substitutions', cpp_sink, 'substitutions')
    
    
    return workflow




def create_seg_gif_create_template_database_workflow(name='gif_create_template_library'):

    """

    Creates a pipeline that uses seg_GIF label propagation to propagate 
    segmentation towards a target image
    
    Example
    -------

    >>> gif = create_niftyseg_gif_propagation_pipeline("niftyseg_gif")
    >>> gif.inputs.inputnode.input_image = 'T1.nii'
    >>> gif.run()                  # doctest: +SKIP

    Inputs::

        inputnode.input_image

    Outputs::


    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_entries_directory',
                    'out_database_directory',
                    'number_of_interations']),
        name='input_node')

    # create a datagrabber that will take care of extracting all these guys
    grabber = pe.Node(interface=nio.DataGrabber(outfields=['T1s']), name='grabber')    
    grabber.inputs.sort_filelist = False
    grabber.inputs.template = '*.nii.gz' 
    
    # Cpps_directory = os.path.join(template_database_directory,'cpps')
    # Labels_directory = os.path.join(template_database_directory,'labels')
    
    input_mask = pe.Node(interface=fsl.BET(), name='input_mask')
    input_mask.inputs.mask = True
    
    registration_to_average = pe.Node(interface=niftyreg.RegAladin(), name="registration_to_average")

    sformupdate = pe.Node(interface = niftyreg.RegTransform(), name = 'sformupdate')

    cropper = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_file', 
                    'in_mask']),
        name='cropper')

    non_linear_registration = pe.MapNode(interface=niftyreg.RegF3D(), name = 'non_linear_registration', iterfield = ['flo_file'])
    non_linear_registration.inputs.output_type = 'NIFTI_GZ'
    
    sinker = pe.Node(nio.DataSink(), name='sinker')

    gif = pe.Node(interface=niftyseg.Gif(), name='gif')

    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['',
                    '']),
        name='output_node')

    
    workflow.connect(input_node, 'template_T1s_directory', grabber, 'base_directory')
    
    workflow.connect(input_node, 'template_average_image', registration_to_average, 'ref_file')
    workflow.connect(input_node, 'in_file',                registration_to_average, 'flo_file')

    workflow.connect(input_node,              'in_file',  sformupdate, 'upd_s_form_input')
    workflow.connect(registration_to_average, 'aff_file', sformupdate, 'upd_s_form_input2')

    workflow.connect(sformupdate, 'out_file',  input_mask, 'in_file')

    workflow.connect(input_mask, 'out_file',  cropper, 'in_file')
    workflow.connect(input_mask, 'mask_file', cropper, 'in_mask')
    
    workflow.connect(cropper, 'in_file', non_linear_registration, 'ref_file')
    workflow.connect(grabber, 'T1s',     non_linear_registration, 'flo_file')

    workflow.connect(input_node,              'out_directory',  sinker, 'base_directory')
    workflow.connect(non_linear_registration, 'cpp_file',       sinker, 'non_linear_registration.cpps')

    workflow.connect(sinker,     'cpps',             gif, 'cpp_dir')    
    workflow.connect(cropper,    'in_file',          gif, 'in_file')
    workflow.connect(input_mask, 'mask_file',        gif, 'mask_file')
    workflow.connect(input_node, 'template_db_file', gif, 'database_file')
    workflow.connect(input_node, 'out_directory',    gif, 'out_dir')
    
    return workflow
