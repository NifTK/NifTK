#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl

import nipype.interfaces.io             as nio
import inspect

import cropimage as cropimage
import n4biascorrection as biascorrection

import registration as reg
import seg_gif_propagation as gif_propagation

import os


'''
'''

def get_upperdir(path):
    import os
    return os.path.dirname(os.path.normpath(path))

def generate_pair_files(in_files):
    ref_files = []
    flo_files = []
    
    for i in range(len(in_files)):
        for j in range(i+1, len(in_files)):
            ref_files.append(in_files[i])
            flo_files.append(in_files[j])

    return ref_files, flo_files



def create_T1_and_cpp_dirs(basedir):
    import os
    ret1 = os.path.join(basedir, 'T1s')
    ret2 = os.path.join(basedir, 'cpps')
    ret3 = os.path.join(basedir, 'labels')
    if not os.path.exists(ret1):
        os.mkdir(ret1)
    if not os.path.exists(ret2):
        os.mkdir(ret2)
    if not os.path.exists(ret3):
        os.mkdir(ret3)
    return ret1, ret2, ret3

def extract_directories(in_files):
    import os
    dirs = []
    for in_file in in_files:
        dirs.append(os.path.dirname(in_file))
    dirs = list(set(dirs))
    dirs.sort()
    return dirs

def find_items(in_items, in_toseparate):
    import os
    indices_1 = []
    indices_2 = []
    toseparate = []
    for item in in_toseparate:
        pattern = os.path.basename(os.path.abspath(item))
        if pattern.endswith('.nii.gz'):
            pattern = pattern[:-7]
        if pattern.endswith('.nii'):
            pattern = pattern[:-4]
        toseparate.append(pattern)

    for i in range(len(in_items)):
        item = in_items[i]
        pattern = os.path.basename(os.path.abspath(item))
        if pattern.endswith('.nii.gz'):
            pattern = pattern[:-7]
        if pattern.endswith('.nii'):
            pattern = pattern[:-4]
        print 'patern  is ', pattern
        isitin = False
        for tosep in toseparate:
            if tosep in pattern:
                isitin = True
                break
        if isitin:
            indices_1.append(i)
        else:
            indices_2.append(i)

    print 'i1 is ', indices_1
    print 'i2 is ', indices_2

    return indices_1, indices_2

def create_database(inputs, labels, iteration, basedir):

    import os, glob
    import neuromorphometricslabels

    def ensuredir(f):
        if not os.path.exists(f):
            print("making directory " + f)
            os.mkdir(f)
            
    def createlink(target, link):
        if not os.path.exists(link):
            if not os.path.exists(target):
                print("target does not exist " + target)
            else:
                os.symlink(target, link)

    database = os.path.join(basedir, 'db' + str(iteration))

    print("creating database at " + database)
  
    ensuredir(database)
    
    labelxmlfile=os.path.join(database, "labels.xml")
    databasexmlfile=os.path.join(database, "db.xml")
    labelsdir=os.path.join(database, "labels")
    priorsdir=os.path.join(database, "priors")
    tissuesdir=os.path.join(database, "tissues")
    geosdir=os.path.join(database, "labels_geo")
    databasexmlcontent='<?xml version="1.0"?> \n\
    <document>\n\
    <data>\n\
    <fname>T1s</fname>\n\
    <path>' + os.path.basename(inputs) + '</path>\n\
    <descr>T1 MRI Data</descr>\n\
    <sform>1</sform>\n\
    </data>\n\
    \n\
    <info>\n\
    <fname>labels</fname>\n\
    <path>labels</path>\n\
    <gpath>labels_geo</gpath>\n\
    <descr>Segmentation</descr>\n\
    <extra>labels.xml</extra>\n\
    <type>LabelSeg</type>\n\
    </info>\n\
    \n\
    </document>'
  
    labelxmlcontent=neuromorphometricslabels.getlabels()

    if not os.path.exists(databasexmlfile):
        f1=open(databasexmlfile, 'w+')
        f1.write(databasexmlcontent)
        f1.close()
    if not os.path.exists(labelxmlfile):
        f2=open(labelxmlfile, 'w+')
        f2.write(labelxmlcontent)
        f2.close()

    ensuredir(labelsdir)
    ensuredir(priorsdir)
    ensuredir(tissuesdir)
    ensuredir(geosdir)
    createlink(inputs, os.path.join(database,os.path.basename(inputs)))

    labelslist=glob.glob(labels+os.sep+'*.nii.gz')
    labelslinkslist = []
    for file in labelslist:
        filename=os.path.basename(file)
        link=os.path.join(labelsdir,filename)
        createlink(file, link)
        labelslinkslist.append(link)

    futuredatabase = os.path.join(basedir, 'db' + str(iteration+1))
    ensuredir(futuredatabase)

    return databasexmlfile, labelxmlfile, labelslinkslist, futuredatabase


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


def get_subs(in_paths, out_paths, prefix_to_remove):
    import os
    subs = []
    commonpath = os.path.commonprefix(out_paths)
    for i in range(len(out_paths)):
       basename = os.path.basename(in_paths[i])
       _, filepath = out_paths[i].split(commonpath,1)
       subs.append((prefix_to_remove+filepath, basename))
    return subs

def get_subs_cpps(in_paths, cpp_files, invcpp_files, prefix_to_remove):
    import os
    subs = []
    outputsubdirs   = []
    outputfilenames = []
    for in_file in in_paths:
        basename = os.path.basename(in_file)
        outputfilenames.append(basename)
        if basename.endswith('.nii.gz'):
            basename = basename[:-7]
        if basename.endswith('.nii'):
            basename = basename[:-4]
        outputsubdirs.append(basename+os.sep)

    N = len(in_paths)
    size = N * (N-1) / 2

    commonpath  = os.path.commonprefix(cpp_files)
    counter     = 0
    counterlimit = N - 1
    index_2_start = 0

    index_1 = 0
    index_2 = 0
    
    for i in range(len(cpp_files)):
       _, cpp = cpp_files[i].split(commonpath,1)
       _, invcpp = invcpp_files[i].split(commonpath,1)
       
       subs.append((prefix_to_remove + cpp,    outputsubdirs[index_1]   + outputfilenames[index_2+1]))
       subs.append((prefix_to_remove + invcpp, outputsubdirs[index_2+1] + outputfilenames[index_1]))

       index_2 = index_2 + 1
       counter = counter + 1

       if counter >= counterlimit :
           index_1 = index_1 + 1
           counterlimit = counterlimit - 1
           index_2 = index_1
           counter = 0

    return subs


def prepare_inputs(name='prepare_inputs', ref_file = None, ref_mask = None):

    do_post_bias_correction = False

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(fields=['in_files']),
        name='input_node')
    
    biascorrect_pre = pe.MapNode(interface = biascorrection.N4BiasCorrection(),
                                 name = 'biascorrect_pre',
                                 iterfield = ['in_file'])
    biascorrect_pre.inputs.in_downsampling = 2

    initial_ref = False

    if ref_file != None:
        initial_ref = True

    groupwise_coregistration = reg.create_atlas(name = 'groupwise_coregistration', 
                                                itr_rigid = 0,
                                                itr_affine = 1,
                                                itr_non_lin = 0,
                                                initial_ref = initial_ref)
    
    if initial_ref:
        groupwise_coregistration.inputs.input_node.ref_file = ref_file

    sformupdate = pe.MapNode(interface = niftyreg.RegTransform(), 
                             name = 'sformupdate', 
                             iterfield = ['upd_s_form_input', 'upd_s_form_input2'])

    if ref_mask == None:
        average_mask = pe.Node(interface=fsl.BET(), name='average_mask')
        average_mask.inputs.mask = True
    
    average_mask_dil = pe.Node(interface = niftyseg.BinaryMaths(), name = 'average_mask_dil')
    average_mask_dil.inputs.operation = 'dil'
    average_mask_dil.inputs.operand_value = 6
    
    average_mask_res = pe.MapNode(interface = niftyreg.RegResample(), 
                                  name = 'average_mask_res',
                                  iterfield = ['ref_file'])
    average_mask_res.inputs.inter_val = 'NN'

    average_crop = pe.Node(interface = cropimage.CropImage(), name='average_crop')

    crop = pe.MapNode(interface = cropimage.CropImage(), name='crop', 
                      iterfield = ['in_file', 'mask_file'])

    biascorrect_post = pe.MapNode(interface = biascorrection.N4BiasCorrection(),
                                  name = 'biascorrect_post',
                                  iterfield = ['in_file'])
    biascorrect_post.inputs.in_downsampling = 2

    output_node = pe.Node(
        interface = niu.IdentityInterface(fields=['out_files', 'out_average', 'out_mask', 'out_affs', 'out_masks']),
        name='output_node')
    
    workflow.connect(input_node,               'in_files',                  biascorrect_pre,          'in_file')
    workflow.connect(biascorrect_pre,          'out_file',                  groupwise_coregistration, 'input_node.in_files')
    workflow.connect(input_node,               'in_files',                  sformupdate,              'upd_s_form_input')
    workflow.connect(groupwise_coregistration, 'output_node.aff_files',     sformupdate,              'upd_s_form_input2')
    if ref_mask == None:
        workflow.connect(groupwise_coregistration, 'output_node.average_image', average_mask,             'in_file')
        workflow.connect(average_mask,             'mask_file',                 average_mask_dil,         'in_file')
    else:
        average_mask_dil.inputs.in_file = ref_mask
        
    workflow.connect(average_mask_dil,         'out_file',                  average_crop,             'mask_file')

    workflow.connect(groupwise_coregistration, 'output_node.average_image', average_crop,             'in_file')
    workflow.connect(sformupdate,              'out_file',                  average_mask_res,         'ref_file')

    if ref_mask == None:
        workflow.connect(average_mask, 'mask_file', average_mask_res, 'flo_file')
    else:
        average_mask_res.inputs.flo_file = ref_mask
    
    workflow.connect(sformupdate,              'out_file',                  crop,                     'in_file')    
    workflow.connect(average_mask_res,         'res_file',                  crop,                     'mask_file')
    workflow.connect(groupwise_coregistration, 'output_node.average_image', output_node,              'out_average')
    if do_post_bias_correction:
        workflow.connect(crop,                 'out_file',                  biascorrect_post,         'in_file')
    workflow.connect(average_crop,             'out_file',                  output_node,              'out_mask')
    if do_post_bias_correction:
        workflow.connect(biascorrect_post,     'out_file',                  output_node,              'out_files')
    else:
        workflow.connect(crop,                 'out_file',                  output_node,              'out_files')

    workflow.connect(groupwise_coregistration, 'output_node.aff_files',     output_node,              'out_affs')
    workflow.connect(average_mask_res,         'res_file',                  output_node,              'out_masks')

    return workflow

def seg_gif_preproc(name='seg_gif_preproc', ref_file = None, ref_mask = None):

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_entries_directory',
                    'out_T1_directory',
                    'out_cpps_directory']),
        name='input_node')

    grabber = pe.Node(interface = nio.DataGrabber(outfields=['T1s_paths']), name = 'grabber')
    grabber.inputs.template = '*.nii*'
    grabber.inputs.sort_filelist = False

    makesubdir = pe.MapNode(
        niu.Function(input_names=['base_dir', 'input_filename'],
                     output_names=['cpp_path'],
                     function=create_subdir),
        name='makesubdir',
        iterfield = ['input_filename'])

    prep_inputs = prepare_inputs(name='prep_inputs', ref_file = ref_file, ref_mask = ref_mask)

    generate_pair = pe.Node(
        niu.Function(input_names=['in_files'],
                     output_names=['ref_files', 'flo_files'],
                     function=generate_pair_files),
        name='generate_pair')

    non_lin_reg = pe.MapNode(interface=niftyreg.RegF3D(), 
                             name = 'non_lin_reg', 
                             iterfield = ['ref_file', 'flo_file'])
    non_lin_reg.inputs.vel_flag  = True
    non_lin_reg.inputs.lncc_val  = -5
    non_lin_reg.inputs.maxit_val = 150
    non_lin_reg.inputs.be_val    = 0.025    

    workflow.connect(input_node,    'in_entries_directory',    grabber,       'base_directory')
    workflow.connect(input_node,    'out_cpps_directory',      makesubdir,    'base_dir')
    workflow.connect(grabber,       'T1s_paths',               makesubdir,    'input_filename')
    workflow.connect(grabber,       'T1s_paths',               prep_inputs,   'input_node.in_files')
    workflow.connect(prep_inputs,   'output_node.out_files',   generate_pair, 'in_files')
    workflow.connect(generate_pair, 'ref_files',               non_lin_reg,   'ref_file')
    workflow.connect(generate_pair, 'flo_files',               non_lin_reg,   'flo_file')

    subsgen1 = pe.Node(
        niu.Function(input_names=['in_paths', 'out_paths', 'prefix_to_remove'],
                     output_names=['substitutions'],
                     function=get_subs),
        name='subsgen1')
    subsgen1.inputs.prefix_to_remove = '_crop'

    T1_sink = pe.Node(nio.DataSink(), name='T1_sink')

    subsgen2 = pe.Node(
        niu.Function(input_names=['in_paths', 'cpp_files', 'invcpp_files', 'prefix_to_remove'],
                     output_names=['substitutions'],
                     function=get_subs_cpps),
        name='subsgen2')
    subsgen2.inputs.prefix_to_remove = '_non_lin_reg'

    cpp_sink = pe.Node(nio.DataSink(), name='cpp_sink')

    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['out_T1s',
                    'out_average',
                    'out_mask',
                    'out_affs',
                    'out_cpps',
                    'out_masks']),
        name='output_node')

    workflow.connect(grabber,     'T1s_paths',               subsgen1, 'in_paths')
    workflow.connect(prep_inputs, 'output_node.out_files',   subsgen1, 'out_paths')
    workflow.connect(input_node,  'out_T1_directory',        T1_sink, 'base_directory')
    workflow.connect(subsgen1,    'substitutions',           T1_sink, 'substitutions')
    workflow.connect(prep_inputs, 'output_node.out_files',   T1_sink, '@T1s')
    workflow.connect(grabber,     'T1s_paths',               subsgen2, 'in_paths')
    workflow.connect(non_lin_reg, 'cpp_file',                subsgen2, 'cpp_files')
    workflow.connect(non_lin_reg, 'invcpp_file',             subsgen2, 'invcpp_files')
    workflow.connect(input_node,  'out_cpps_directory',      cpp_sink, 'base_directory')
    workflow.connect(subsgen2,    'substitutions',           cpp_sink, 'substitutions')
    workflow.connect(non_lin_reg, 'cpp_file',                cpp_sink, '@cpps')
    workflow.connect(non_lin_reg, 'invcpp_file',             cpp_sink, '@invcpps')
    workflow.connect(T1_sink,     'out_file',                output_node, 'out_T1s')
    workflow.connect(cpp_sink,    'out_file',                output_node, 'out_cpps')
    workflow.connect(prep_inputs, 'output_node.out_affs',    output_node, 'out_affs')
    workflow.connect(prep_inputs, 'output_node.out_average', output_node, 'out_average')
    workflow.connect(prep_inputs, 'output_node.out_mask',    output_node, 'out_mask')
    workflow.connect(prep_inputs, 'output_node.out_masks',   output_node, 'out_masks')
    
    return workflow




def create_seg_gif_create_template_database_workflow(name='gif_create_template_library', number_of_iterations = 1, ref_file = None, ref_mask = None):

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_entries_directory',
                    'in_initial_labels_directory',
                    'out_database_directory']),
        name='input_node')

    # make a node to create all subdirectories, that will include the cropped T1s and cpps
    create_T1_and_cpp = pe.Node(interface = niu.Function(input_names = ['basedir'],
                                                         output_names = ['out_T1_directory', 'out_cpp_directory', 'out_label_directory'],
                                                         function=create_T1_and_cpp_dirs),
                        name = 'create_T1_and_cpp')

    workflow.connect(input_node, 'out_database_directory', create_T1_and_cpp, 'basedir')
    
    preproc = seg_gif_preproc(name = 'preproc', ref_file = ref_file, ref_mask = ref_mask)
    
    workflow.connect (input_node,        'in_entries_directory', preproc, 'input_node.in_entries_directory')
    workflow.connect (create_T1_and_cpp, 'out_T1_directory',     preproc, 'input_node.out_T1_directory')
    workflow.connect (create_T1_and_cpp, 'out_cpp_directory',    preproc, 'input_node.out_cpps_directory')

    extract_cpp_dirs = pe.Node(interface = niu.Function(input_names = ['in_files'],
                                                        output_names = ['out_directories'],
                                                        function=extract_directories),
                               name = 'extract_cpp_dirs')

    workflow.connect(preproc, 'output_node.out_cpps', extract_cpp_dirs, 'in_files')

    grabber = pe.Node(interface = nio.DataGrabber(outfields=['out_files']), name = 'grabber')
    grabber.inputs.template = '*.nii*'
    grabber.inputs.sort_filelist = False

    workflow.connect(input_node, 'in_initial_labels_directory', grabber, 'base_directory')
    
    find_labels_T1s = pe.Node(interface = niu.Function(input_names  = ['in_items', 'in_toseparate'],
                                                       output_names = ['out_labels_indices', 'out_T1s_indices'],
                                                       function=find_items),
                              name = 'find_labels_T1s')
    select_T1s = pe.Node(interface = niu.Select(),
                         name = 'select_T1s')
    select_T1masks = pe.Node(interface = niu.Select(),
                           name = 'select_T1masks')
    select_labelmasks = pe.Node(interface = niu.Select(),
                           name = 'select_labelmasks')
    select_labelsaffs = pe.Node(interface = niu.Select(),
                                name = 'select_labelsaffs')
    select_cpp_dirs = pe.Node(interface = niu.Select(),
                              name = 'select_cpp_dirs')


    workflow.connect(preproc, 'output_node.out_T1s', find_labels_T1s, 'in_items')
    workflow.connect(grabber, 'out_files',           find_labels_T1s, 'in_toseparate')

    workflow.connect(find_labels_T1s, 'out_T1s_indices', select_T1s, 'index')
    workflow.connect(preproc, 'output_node.out_T1s',     select_T1s, 'inlist')

    workflow.connect(find_labels_T1s, 'out_T1s_indices', select_T1masks, 'index')
    workflow.connect(preproc, 'output_node.out_masks',   select_T1masks, 'inlist')

    workflow.connect(find_labels_T1s, 'out_labels_indices', select_labelmasks, 'index')
    workflow.connect(preproc, 'output_node.out_masks',      select_labelmasks, 'inlist')

    workflow.connect(find_labels_T1s, 'out_labels_indices', select_labelsaffs, 'index')
    workflow.connect(preproc, 'output_node.out_affs',       select_labelsaffs, 'inlist')

    workflow.connect(extract_cpp_dirs, 'out_directories', select_cpp_dirs, 'inlist')
    workflow.connect(find_labels_T1s, 'out_T1s_indices',  select_cpp_dirs, 'index')

    sformupdate_labels = pe.MapNode(interface = niftyreg.RegTransform(), 
                             name = 'sformupdate_labels', 
                             iterfield = ['upd_s_form_input', 'upd_s_form_input2'])
    crop_labels = pe.MapNode(interface = cropimage.CropImage(), 
                             name='crop_labels', 
                             iterfield = ['in_file', 'mask_file'])
    sink_labels = pe.Node(nio.DataSink(), name='sink_labels')
    subs = []
    subs.append (('_crop_labels[0-9]+'+os.sep, ''))
    subs.append (('_reg_transform_crop_image', ''))
    sink_labels.inputs.regexp_substitutions = subs

    workflow.connect(grabber, 'out_files',     sformupdate_labels, 'upd_s_form_input')
    workflow.connect(select_labelsaffs, 'out', sformupdate_labels, 'upd_s_form_input2')
    workflow.connect(sformupdate_labels, 'out_file', crop_labels, 'in_file')
    workflow.connect(select_labelmasks, 'out', crop_labels, 'mask_file')
    workflow.connect(create_T1_and_cpp, 'out_label_directory', sink_labels, 'base_directory')
    workflow.connect(crop_labels, 'out_file', sink_labels, '@labels')

    extract_label_dir = pe.Node(interface = niu.Function(input_names = ['paths'],
                                                         output_names = ['out_dir'],
                                                         function = gif_propagation.get_basedir),
                                name = 'extract_label_dir')
    workflow.connect(sink_labels, 'out_file', extract_label_dir, 'paths')

    resampler_mask  = pe.MapNode(interface = niftyreg.RegResample(), 
                                 name = 'resampler_mask',
                                 iterfield = ['ref_file', 'flo_file'])
    resampler_mask.inputs.inter_val = 'NN'
    workflow.connect (select_T1s,     'out', resampler_mask, 'ref_file')
    workflow.connect (select_T1masks, 'out', resampler_mask, 'flo_file')
                                
    list_of_gifs = []

    for i in range(1, number_of_iterations+1):
        create_db = pe.Node(interface = niu.Function(input_names = ['inputs', 'labels', 'iteration', 'basedir'],
                                                     output_names = ['db_file', 'labels_file', 'labels_files', 'next_db_directory'],
                                                     function=create_database),
                            name = 'create_db_'+str(i))
        create_db.inputs.iteration = i

        workflow.connect (create_T1_and_cpp, 'out_T1_directory',            create_db, 'inputs')
        workflow.connect (extract_label_dir, 'out_dir',                     create_db, 'labels')

        if (i == 1):
            workflow.connect (input_node, 'out_database_directory', create_db, 'basedir')
        else:
            extract_basedir = pe.Node(interface = niu.Function(input_names  = ['path'],
                                                               output_names = ['out_dir'],
                                                               function=get_upperdir),
                                      name = 'extract_basedir_'+str(i))
            workflow.connect (list_of_gifs[len(list_of_gifs)-1], 'output_node.out_directory', extract_basedir, 'path')
            workflow.connect (extract_basedir, 'out_dir', create_db, 'basedir')
            

        gif = gif_propagation.create_niftyseg_gif_propagation_pipeline_simple(name = 'gif_propagation_'+str(i))
        
        workflow.connect (select_T1s,       'out',                  gif, 'input_node.in_file')
        workflow.connect (select_cpp_dirs,  'out',                  gif, 'input_node.cpp_directory')
        workflow.connect (resampler_mask,   'res_file',             gif, 'input_node.mask_file')
        workflow.connect (create_db,        'db_file',              gif, 'input_node.template_db_file')
        workflow.connect (create_db,        'next_db_directory',    gif, 'input_node.out_directory')    

        list_of_gifs.append(gif)
    
    return workflow
