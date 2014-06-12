#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl

import nipype.interfaces.io             as nio
import inspect
import os

import cropimage as cropimage


def ensure_aff_files (basedir):
    import os, glob
    cppslist=glob.glob(basedir+os.sep+'*.nii.gz')
    
    affcontent='1 0 0 0\n\
    0 1 0 0\n\
    0 0 1 0\n\
    0 0 0 1\n'

    afffilelist = []
    
    for f in cppslist:
        afffile = f.replace('.nii.gz', '.txt')
        print 'making file ', afffile
        if not os.path.exists(afffile):
            f1=open(afffile, 'w+')
            f1.write(affcontent)
            f1.close()
        afffilelist.append(afffile)

    return basedir, afffilelist
            

def get_subs1(T1s_paths, Cpps_paths):
    import os
    subs = []

    commonpath = os.path.commonprefix(Cpps_paths)

    for i in range(len(Cpps_paths)):
       T1basename = os.path.basename(T1s_paths[i])
       _, cppfilepath = Cpps_paths[i].split(commonpath,1)
       subs.append(('_non_linear_registration'+cppfilepath, T1basename))

    return subs

def get_subs2(inputfile, outputfile):
    import os
    subs = []
    suffixes = [
        '_labels_Parcellation',
        '_labels_geo',
        '_labels_prior']

    for suffix in suffixes:
        subs.append((suffix, ''))

    subs.append((os.path.basename(inputfile),os.path.basename(outputfile)))

    return subs

def get_basedir(paths):
    mypaths = list(paths)
    import os
    return os.path.dirname(mypaths[0])

def get_common_directory(in_files):
    import os
    commonpath = os.path.commonprefix(list(in_files))    
    return commonpath


def create_niftyseg_gif_propagation_pipeline_simple(name='niftyseg_gif_propagation'):
    """
    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_file',
                    'mask_file',
                    'template_db_file',
                    'cpp_directory',
                    'out_directory']),
        name='input_node')
    
    # make a node to create all subdirectories, that will include the cropped T1s and cpps
    create_aff_files = pe.MapNode(interface = niu.Function(input_names = ['basedir'],
                                                           output_names = ['out_dir', 'afffilelist'],
                                                           function=ensure_aff_files),
                                  name = 'create_aff_files',
                                  iterfield = ['basedir'])

    gif = pe.MapNode(interface=niftyseg.Gif(), 
                     name='gif',
                     iterfield=['in_file', 'mask_file', 'cpp_dir'])

    gif_post_sink = pe.Node(nio.DataSink(), name='gif_post_sink')
    subs = []
    subs.append (('_gif[0-9]+'+os.sep, ''))
    suffixes = [
        '_labels_Parcellation',
        '_labels_geo',
        '_labels_prior']
    for suffix in suffixes:
        subs.append((suffix, ''))
    gif_post_sink.inputs.regexp_substitutions = subs

    workflow.connect(input_node, 'cpp_directory',     create_aff_files, 'basedir')

    workflow.connect(input_node, 'in_file',           gif, 'in_file')
    workflow.connect(input_node, 'mask_file',         gif, 'mask_file')
    workflow.connect(input_node, 'template_db_file',  gif, 'database_file')
    workflow.connect(create_aff_files, 'out_dir',     gif, 'cpp_dir')
    
    workflow.connect(input_node, 'out_directory',  gif_post_sink, 'base_directory')
    workflow.connect(gif,        'parc_file',      gif_post_sink, 'labels')  
    workflow.connect(gif,        'geo_file',       gif_post_sink, 'labels_geo')
    workflow.connect(gif,        'prior_file',     gif_post_sink, 'priors')

    extract_sink_dir = pe.Node(interface = niu.Function(input_names = ['in_files'],
                                                        output_names = ['out_dir'],
                                                        function = get_common_directory),
                               name = 'extract_sink_dir')
    workflow.connect(gif_post_sink, 'out_file', extract_sink_dir, 'in_files')

    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['parc_file', 'geo_file', 'prior_file', 'out_directory']),
        name='output_node')
    
    workflow.connect(extract_sink_dir, 'out_dir',       output_node, 'out_directory')
    workflow.connect(gif,              'parc_file',     output_node, 'parc_file')
    workflow.connect(gif,              'geo_file',      output_node, 'geo_file')
    workflow.connect(gif,              'prior_file',    output_node, 'prior_file')

    return workflow


def create_niftyseg_gif_propagation_pipeline(stand_alone = True, name='niftyseg_gif_propagation'):

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
                    'template_T1s_directory',
                    'template_db_file',
                    'template_average_image',
                    'out_cpp_directory',
                    'out_res_directory']),
        name='input_node')

    # create a datagrabber that will take care of extracting all these guys
    grabber = pe.Node(interface = nio.DataGrabber(outfields=['T1s_paths']), name = 'grabber')
    grabber.inputs.template = '*.nii*'
    grabber.inputs.sort_filelist = False
    
    input_mask = pe.Node(interface=fsl.BET(), name='input_mask')
    input_mask.inputs.mask = True
    
    dilater = pe.Node(interface = niftyseg.BinaryMaths(), name = 'dilater')
    dilater.inputs.operation = 'dil'
    dilater.inputs.operand_value = 10

    registration_to_average = pe.Node(interface=niftyreg.RegAladin(), name="registration_to_average")

    sformupdate = pe.Node(interface = niftyreg.RegTransform(), name = 'sformupdate')

    invaffer = pe.Node(interface = niftyreg.RegTransform(), name = 'invaffer')

    cropper = pe.Node(interface = cropimage.CropImage(), name='cropper')

    resampler_mask  = pe.Node(interface = niftyreg.RegResample(), name = 'resampler_mask')
    resampler_mask.inputs.inter_val = 'NN'

    non_linear_registration = pe.MapNode(interface=niftyreg.RegF3D(), name = 'non_linear_registration', iterfield = ['flo_file'])
    non_linear_registration.inputs.vel_flag  = True
    non_linear_registration.inputs.lncc_val  = -5
    non_linear_registration.inputs.maxit_val = 150
    non_linear_registration.inputs.be_val    = 0.025    
    non_linear_registration.inputs.output_type = 'NIFTI_GZ'    

    subsgen1 = pe.Node(niu.Function(input_names=['T1s_paths', 'Cpps_paths'],
                                    output_names=['substitutions'],
                                    function=get_subs1),
                       name='subsgen1')

    gif_pre_sink = pe.Node(nio.DataSink(), name='gif_pre_sink')
    gif_pre_sink.inputs._outputs = dict([['cpps','cpps']])

    gif = pe.Node(interface=niftyseg.Gif(), name='gif')

    resampler_parc  = pe.Node(interface = niftyreg.RegResample(), name = 'resampler_parc')
    resampler_parc.inputs.inter_val = 'NN'
    resampler_geo   = pe.Node(interface = niftyreg.RegResample(), name = 'resampler_geo')
    resampler_prior = pe.Node(interface = niftyreg.RegResample(), name = 'resampler_prior')

    sformdeupdate_parc  = pe.Node(interface = niftyreg.RegTransform(), name = 'sformdeupdate_parc')
    sformdeupdate_geo   = pe.Node(interface = niftyreg.RegTransform(), name = 'sformdeupdate_geo')
    sformdeupdate_prior = pe.Node(interface = niftyreg.RegTransform(), name = 'sformdeupdate_prior')

    pathgen = pe.Node(niu.Function(input_names=['paths'],
                                   output_names=['path'],
                                   function=get_basedir),
                      name='pathgen')
    
    subsgen2 = pe.Node(niu.Function(input_names=['inputfile', 'outputfile'],
                                    output_names=['substitutions'],
                                    function=get_subs2),
                       name='subsgen2')

    gif_post_sink = pe.Node(nio.DataSink(), name='gif_post_sink')
    
    workflow.connect(input_node, 'template_T1s_directory', grabber, 'base_directory')
    
    workflow.connect(input_node, 'template_average_image', registration_to_average, 'ref_file')
    workflow.connect(input_node, 'in_file',                registration_to_average, 'flo_file')

    workflow.connect(input_node,              'in_file',  sformupdate, 'upd_s_form_input')
    workflow.connect(registration_to_average, 'aff_file', sformupdate, 'upd_s_form_input2')

    workflow.connect(sformupdate, 'out_file',  input_mask, 'in_file')

    workflow.connect(input_mask, 'mask_file',  dilater, 'in_file')

    workflow.connect(sformupdate, 'out_file', cropper, 'in_file')
    workflow.connect(dilater,     'out_file', cropper, 'mask_file')

    workflow.connect(cropper, 'out_file', resampler_mask,  'ref_file')
    workflow.connect(dilater, 'out_file', resampler_mask,  'flo_file')
    
    workflow.connect(cropper, 'out_file',  non_linear_registration, 'ref_file')
    workflow.connect(grabber, 'T1s_paths', non_linear_registration, 'flo_file')
    
    workflow.connect(grabber, 'T1s_paths',                subsgen1, 'T1s_paths')
    workflow.connect(non_linear_registration, 'cpp_file', subsgen1, 'Cpps_paths')
    
    workflow.connect(input_node,              'out_cpp_directory', gif_pre_sink, 'base_directory')
    workflow.connect(non_linear_registration, 'cpp_file',          gif_pre_sink, '@cpps')
    workflow.connect(subsgen1,                'substitutions',     gif_pre_sink, 'substitutions')
    
    workflow.connect(gif_pre_sink, 'out_file', pathgen, 'paths')
    
    workflow.connect(pathgen,        'path',             gif, 'cpp_dir')
    workflow.connect(cropper,        'out_file',         gif, 'in_file')
    workflow.connect(resampler_mask, 'res_file',         gif, 'mask_file')
    workflow.connect(input_node,     'template_db_file', gif, 'database_file')
    
    workflow.connect(sformdeupdate_parc, 'out_file', subsgen2, 'inputfile')
    workflow.connect(input_node,         'in_file',  subsgen2, 'outputfile')

    workflow.connect(sformupdate, 'out_file',     resampler_parc,  'ref_file')
    workflow.connect(sformupdate, 'out_file',     resampler_geo,   'ref_file')
    workflow.connect(sformupdate, 'out_file',     resampler_prior, 'ref_file')
    workflow.connect(gif,         'parc_file',     resampler_parc,  'flo_file')
    workflow.connect(gif,         'geo_file',      resampler_geo,   'flo_file')
    workflow.connect(gif,         'prior_file',    resampler_prior, 'flo_file')

    workflow.connect(registration_to_average, 'aff_file', invaffer, 'inv_aff_input')

    workflow.connect(resampler_parc,  'res_file', sformdeupdate_parc,  'upd_s_form_input')
    workflow.connect(resampler_geo,   'res_file', sformdeupdate_geo,   'upd_s_form_input')
    workflow.connect(resampler_prior, 'res_file', sformdeupdate_prior, 'upd_s_form_input')
    workflow.connect(invaffer,        'out_file', sformdeupdate_parc,  'upd_s_form_input2')
    workflow.connect(invaffer,        'out_file', sformdeupdate_geo,   'upd_s_form_input2')
    workflow.connect(invaffer,        'out_file', sformdeupdate_prior, 'upd_s_form_input2')

    workflow.connect(input_node,          'out_res_directory', gif_post_sink, 'base_directory')
    workflow.connect(subsgen2,            'substitutions', gif_post_sink, 'substitutions')
    workflow.connect(sformdeupdate_parc,  'out_file',      gif_post_sink, 'labels')
    workflow.connect(sformdeupdate_geo,   'out_file',      gif_post_sink, 'labels_geo')
    workflow.connect(sformdeupdate_prior, 'out_file',      gif_post_sink, 'priors')

#    workflow.connect(gif, 'out_tiv',   gif_post_sink, 'others@tiv')
#    workflow.connect(gif, 'out_seg',   gif_post_sink, 'others@seg')
#    workflow.connect(gif, 'out_brain', gif_post_sink, 'others@brain')
#    workflow.connect(gif, 'out_bias',  gif_post_sink, 'others@bias')


    return workflow
