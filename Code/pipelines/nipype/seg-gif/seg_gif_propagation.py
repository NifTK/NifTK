#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.io             as nio
import inspect
import os

import cropimage as cropimage
import n4biascorrection                 as biascorrection
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg




def get_db_file(in_files):
    import os
    db_directory = os.path.dirname(in_files[0])
    return os.path.join(db_directory, 'db.xml')

def find_gif_substitutions_function(in_db_file):
    def find_database_fname(in_file):
        import xml.etree.ElementTree as ET
        tree = ET.parse(in_file)
        root = tree.getroot()
        return root.findall('info')[0].findall('fname')[0].text
    in_fname = find_database_fname (in_db_file)
    subs = []
    subs.append(('_'+in_fname+'_Parcellation', ''))
    subs.append(('_'+in_fname+'_prior', ''))
    subs.append(('_'+in_fname+'_geo', ''))
    subs.append(('_'+in_fname+'_Sinth', ''))
    return subs

def create_niftyseg_gif_propagation_pipeline_simple(name='niftyseg_gif_propagation'):
    """
    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name
    
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_file',
                    'in_mask_file',
                    'in_db_file',
                    'in_cpp_dir',
                    'out_dir']),
        name='input_node')

    '''
    *****************************************************************************
    First step: Perform the GIF propagation from the inputs provided
    *****************************************************************************
    '''

    gif = pe.MapNode(interface=niftyseg.Gif(), 
                     name='gif',
                     iterfield=['in_file', 'mask_file', 'cpp_dir'])

    '''
    *****************************************************************************
    Second step: Sink the GIF outputs
    *****************************************************************************
    '''

    gif_sink = pe.Node(nio.DataSink(), name='gif_sink')
    gif_sink.inputs.parameterization = False

    '''
    *****************************************************************************
    Third step: extract the potential next database file from the output directory
    This is used as entry point for the next iteration
    *****************************************************************************
    '''
    extract_output_database = pe.Node(interface = niu.Function(
        input_names = ['in_files'],
        output_names = ['out_db'],
        function = get_db_file),
                                      name = 'extract_output_database')


    '''
    *****************************************************************************
    First step: Perform the GIF propagation from the inputs provided
    *****************************************************************************
    '''
    workflow.connect(input_node, 'in_file',      gif, 'in_file')
    workflow.connect(input_node, 'in_mask_file', gif, 'mask_file')
    workflow.connect(input_node, 'in_cpp_dir',   gif, 'cpp_dir')
    workflow.connect(input_node, 'in_db_file',   gif, 'database_file')    
    
    '''
    *****************************************************************************
    Second step: Sink the GIF outputs
    *****************************************************************************
    '''
    workflow.connect(input_node, 'out_dir',        gif_sink, 'base_directory')
    workflow.connect(gif,        'parc_file',      gif_sink, 'labels')  
    workflow.connect(gif,        'geo_file',       gif_sink, 'labels_geo')
    workflow.connect(gif,        'prior_file',     gif_sink, 'priors')

    '''
    *****************************************************************************
    Aux step: substitutions for the GIF outputs
    *****************************************************************************
    '''
    find_gif_substitutions = pe.Node(interface = niu.Function(
        input_names = ['in_db_file'],
        output_names = ['substitutions'],
        function=find_gif_substitutions_function),
                                     name = 'find_gif_substitutions')
    workflow.connect(input_node, 'in_db_file', find_gif_substitutions, 'in_db_file')
    workflow.connect(find_gif_substitutions, 'substitutions', gif_sink, 'regexp_substitutions')

    '''
    *****************************************************************************
    Third step: extract the potential next database file from the output directory
    This is used as entry point for the next iteration
    *****************************************************************************
    '''
    workflow.connect(gif_sink, 'out_file', extract_output_database, 'in_files')

    '''
    *****************************************************************************
    Connect the outputs to the output node
    *****************************************************************************
    '''
    
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['parc_file', 'geo_file', 'prior_file', 'out_db']),
        name='output_node')
    
    workflow.connect(extract_output_database, 'out_db', output_node, 'out_db')
    workflow.connect(gif, 'parc_file', output_node, 'parc_file')
    workflow.connect(gif, 'geo_file', output_node, 'geo_file')
    workflow.connect(gif, 'prior_file', output_node, 'prior_file')

    return workflow








'''
Convenient function that outputs the dirname of
a list of files
'''
def get_dirname(in_files):
    import os
    directory = os.path.dirname(in_files[0])
    return directory

'''
Convenient function that generates the substitutions 
between files 
'''
def find_preprocessing_substitutions(in_files, out_files):
    import os
    subs = []
    start_index = out_files[0].rfind('mapflow')+8
    for i in range(len(in_files)):
        subs.append(( out_files[i][start_index:], os.path.basename(in_files[i]) ))
    return subs


def create_niftyseg_gif_propagation_pipeline(name='niftyseg_gif_propagation'):

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
            fields=['in_file',
                    'in_mask',
                    'in_t1s_dir',
                    'in_db_file',
                    'out_cpp_dir']),
        name='input_node')

    '''
    *****************************************************************************
    Initial step: grab the input T1s towards which the input needs to be registered to
    *****************************************************************************
    '''
    grabber_t1s = pe.Node(interface = nio.DataGrabber(outfields=['images']), 
                          name = 'grabber_t1s')
    grabber_t1s.inputs.template = '*.nii*'
    grabber_t1s.inputs.sort_filelist = True

    workflow.connect(input_node, 'in_t1s_dir', grabber_t1s, 'base_directory')
    

    '''
    *****************************************************************************
    First step: Cropping input according to 10 voxels surrounding the skull
    *****************************************************************************
    '''
    dilate_image_mask = pe.Node(interface = niftyseg.BinaryMaths(),
                                name = 'dilate_image_mask')
    dilate_image_mask.inputs.operation = 'dil'
    dilate_image_mask.inputs.operand_value = 10

    crop_image_with_mask = pe.Node(interface = cropimage.CropImage(), 
                                   name='crop_image_with_mask')

    resample_image_mask_to_cropped_image = pe.Node(interface = niftyreg.RegResample(),
                                                   name = 'resample_image_mask_to_cropped_image')
    resample_image_mask_to_cropped_image.inputs.inter_val = 'NN'

    '''
    Bias correct the cropped image.
    '''
    bias_correction = pe.Node(interface = biascorrection.N4BiasCorrection(),
                              name = 'bias_correction')
    bias_correction.inputs.in_downsampling=2

    '''
    *****************************************************************************
    Second step: Affine register all T1s towards the input image
    *****************************************************************************
    '''
    linear_registration = pe.MapNode(interface=niftyreg.RegAladin(), 
                                     name='linear_registration', 
                                     iterfield=['flo_file'])
    
    '''
    *****************************************************************************
    Third step: Non-linear register all T1s towards the input image
    *****************************************************************************
    '''
    non_linear_registration = pe.MapNode(interface=niftyreg.RegF3D(), 
                                         name = 'non_linear_registration', 
                                         iterfield = ['flo_file', 'aff_file'])
    non_linear_registration.inputs.vel_flag  = True
    non_linear_registration.inputs.lncc_val  = -5
    non_linear_registration.inputs.maxit_val = 150
    non_linear_registration.inputs.be_val    = 0.025
    non_linear_registration.inputs.output_type = 'NIFTI_GZ'    

    '''
    *****************************************************************************
    Fourth step: Perform GIF propagation
    *****************************************************************************
    '''

    gif = pe.Node(interface=niftyseg.Gif(), 
                     name='gif')



    '''
    *****************************************************************************
    First step: Cropping input according to 10 voxels surrounding the skull
    *****************************************************************************
    '''
    workflow.connect(input_node, 'in_mask', dilate_image_mask, 'in_file')
    workflow.connect(input_node, 'in_file', crop_image_with_mask, 'in_file')
    workflow.connect(dilate_image_mask, 'out_file', crop_image_with_mask, 'mask_file')
    workflow.connect(crop_image_with_mask, 'out_file', resample_image_mask_to_cropped_image, 'ref_file')
    workflow.connect(input_node, 'in_mask', resample_image_mask_to_cropped_image, 'flo_file')
    workflow.connect(crop_image_with_mask, 'out_file', bias_correction, 'in_file')
    workflow.connect(resample_image_mask_to_cropped_image, 'res_file', bias_correction, 'mask_file')

    '''
    *****************************************************************************
    Second step: Affine register all T1s towards the input image
    *****************************************************************************
    '''
    workflow.connect(bias_correction, 'out_file', linear_registration, 'ref_file')
    workflow.connect(grabber_t1s, 'images', linear_registration, 'flo_file')

    '''
    *****************************************************************************
    Third step: Non-linear register all T1s towards the input image
    *****************************************************************************
    '''
    workflow.connect(bias_correction, 'out_file', non_linear_registration, 'ref_file')
    workflow.connect(grabber_t1s, 'images', non_linear_registration, 'flo_file')
    workflow.connect(linear_registration, 'aff_file', non_linear_registration, 'aff_file')


    '''
    *****************************************************************************
    Aux step: Sink the resulting cpp displacement fields in the output directories
    *****************************************************************************
    '''

    cpp_sink = pe.Node(nio.DataSink(parameterization=True),
                       name = 'cpp_sink')
    workflow.connect(input_node, 'out_cpp_dir', cpp_sink, 'base_directory') 
    workflow.connect(non_linear_registration, 'cpp_file', cpp_sink, '@cpps')
    find_cpp_substitutions = pe.Node(niu.Function(
        input_names = ['in_files', 'out_files'],
        output_names = ['substitutions'],
        function = find_preprocessing_substitutions),
                                     name = 'find_cpp_substitutions')
    
    workflow.connect(grabber_t1s, 'images', find_cpp_substitutions, 'in_files') 
    workflow.connect(non_linear_registration, 'cpp_file', find_cpp_substitutions, 'out_files') 
    workflow.connect(find_cpp_substitutions, 'substitutions', cpp_sink, 'substitutions') 
    
    find_cpp_dir = pe.Node(niu.Function(
        input_names = ['in_files'],
        output_names = ['out_dir'],
        function = get_dirname),
                           name = 'find_cpp_dir')
    workflow.connect(cpp_sink, 'out_file', find_cpp_dir, 'in_files')


    '''
    *****************************************************************************
    Fourth step: Perform GIF propagation
    *****************************************************************************
    '''
    workflow.connect(find_cpp_dir, 'out_dir', gif, 'cpp_dir')
    workflow.connect(bias_correction, 'out_file', gif, 'in_file')
    workflow.connect(resample_image_mask_to_cropped_image, 'res_file', gif, 'mask_file')
    workflow.connect(input_node, 'in_db_file', gif, 'database_file')


    '''
    *****************************************************************************
    Connect the outputs to the output node
    *****************************************************************************
    '''
    
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['out_parc_file', 
                    'out_geo_file', 
                    'out_prior_file',
                    'out_tiv_file',
                    'out_seg_file',
                    'out_brain_file',
                    'out_bias_file']),
        name='output_node')
    
    workflow.connect(gif, 'parc_file', output_node, 'out_parc_file')
    workflow.connect(gif, 'geo_file', output_node, 'out_geo_file')
    workflow.connect(gif, 'prior_file', output_node, 'out_prior_file')
#    workflow.connect(gif, 'out_tiv',   output_node, 'out_tiv_file')
#    workflow.connect(gif, 'out_seg',   output_node, 'out_seg_file')
#    workflow.connect(gif, 'out_brain', output_node, 'out_brain_file')
#    workflow.connect(gif, 'out_bias',  output_node, 'out_bias_file')

    return workflow
