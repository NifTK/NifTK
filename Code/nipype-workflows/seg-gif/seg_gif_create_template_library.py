#! /usr/bin/env python


from seg_gif_propagation import *

'''
'''


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
                    'in_cpps_directory',
                    'out_directory']),
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
