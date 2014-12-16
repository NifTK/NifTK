#! /usr/bin/env python

import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.io             as nio
import inspect
import os

import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import seg_gif_propagation              as gif
import niftk                            as niftk
import registration                     as reg
import cropimage                        as cropimage

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')



'''
A Function that takes a list of objects as input in_files and returns 
a list of N * (N - 1) pairs of distinct objects
'''
def generate_unique_pairs(in_files):
    files_1 = []
    files_2 = []
    
    for i in range(len(in_files)):
        for j in range(i+1, len(in_files)):
            files_1.append(in_files[i])
            files_2.append(in_files[j])

    return files_1, files_2


def find_item_in_list(in_item, in_tosearch, in_tochoose):
    import os
    basename = os.path.basename(in_item)
    basename = basename.replace('.nii.gz', '')
    basename = basename.replace('.nii', '')
    for index in range(len(in_tosearch)):        
        if basename in in_tosearch[index]:
            return in_tochoose[index]

    return None


'''
The function returns a workflow that prepares the inputs.
it does:
1. bias correction of all inputs
2. Cropping inputs according to 10 voxels surrounding the skull
3. Crop the labels accordingly
4. Group wise registration between cropped inputs
5. Generating N * (N - 1) pairs of images to be aligned
6. Non linear registration of all pairs

Outputs are:

1. the output files cropped and bias corrected
2. the corresponding masks
3. the labels cropped
4. the affine matrices to the average,
5. the N * (N - 1) non linear cpp files
6. the N * (N - 1) non linear inv_cpp files
'''
def preprocessing_inputs_pipeline(name='preprocessing_inputs_pipeline', number_of_affine_iterations = 7, ref_file = mni_template, ref_mask = mni_template_mask):

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(fields=['in_files', 'in_labels']),
        name='input_node')

    '''
    *****************************************************************************
    First step: Cropping inputs according to 10 voxels surrounding the skull
    *****************************************************************************
    '''
    register_mni_to_image = pe.MapNode(interface = niftyreg.RegAladin(),
                                       name = 'register_mni_to_image',
                                       iterfield = ['ref_file'])
    register_mni_to_image.inputs.flo_file = mni_template
    resample_mni_mask_to_image = pe.MapNode(interface = niftyreg.RegResample(),
                                            name = 'resample_mni_mask_to_image',
                                            iterfield = ['ref_file', 'aff_file'])
    resample_mni_mask_to_image.inputs.inter_val = 'NN'
    resample_mni_mask_to_image.inputs.flo_file = mni_template_mask

    dilate_image_mask = pe.MapNode(interface = niftyseg.BinaryMaths(),
                                   name = 'dilate_image_mask',
                                   iterfield = ['in_file'])
    dilate_image_mask.inputs.operation = 'dil'
    dilate_image_mask.inputs.operand_value = 10
    
    crop_image_with_mask = pe.MapNode(interface = cropimage.CropImage(), 
                                      name='crop_image_with_mask', 
                                      iterfield = ['in_file', 'mask_file'])

    resample_image_mask_to_cropped_image = pe.MapNode(interface = niftyreg.RegResample(),
                                            name = 'resample_image_mask_to_cropped_image',
                                                      iterfield = ['ref_file', 'aff_file'])
    resample_image_mask_to_cropped_image.inputs.inter_val = 'NN'
    resample_image_mask_to_cropped_image.inputs.flo_file = mni_template_mask

    bias_correction = pe.MapNode(interface = niftk.N4BiasCorrection(),
                                       name = 'bias_correction',
                                       iterfield = ['in_file', 'mask_file'])
    bias_correction.inputs.in_downsampling=2
    

    '''
    *****************************************************************************
    Aux step: Crop the labels accordingly
    *****************************************************************************
    '''
    find_labels = pe.MapNode(interface = niu.Function(
        input_names = ['in_item', 'in_tosearch', 'in_tochoose'],
        output_names = ['out_item'],
        function = find_item_in_list),
                             name = 'find_labels', 
                             iterfield = ['in_item'])

    resample_label_to_cropped_image = pe.MapNode(interface = niftyreg.RegResample(),
                                            name = 'resample_label_to_cropped_image',
                                                 iterfield = ['ref_file', 'flo_file'])
    resample_label_to_cropped_image.inputs.inter_val = 'NN'



    '''
    *****************************************************************************
    Second step: Group wise registration between cropped inputs
    *****************************************************************************
    '''

    groupwise_registration = reg.create_atlas(name = 'groupwise_registration',
                                              initial_ref = True,
                                              itr_rigid = 0,
                                              itr_affine = number_of_affine_iterations,
                                              itr_non_lin = 0)
    groupwise_registration.get_node('input_node').inputs.ref_file = ref_file


    '''
    *****************************************************************************
    Third step: Generating N * (N - 1) pairs of images to be aligned
    *****************************************************************************
    '''

    generate_image_pairs = pe.Node(niu.Function(
        input_names=['in_files'],
        output_names=['files_1', 'files_2'],
        function=generate_unique_pairs),
                                            name='generate_image_pairs')
    '''
    *****************************************************************************
    I noticed the resulting displacement fields were smoother and better without providing the masks...
    *****************************************************************************
    generate_mask_pairs = pe.Node(niu.Function(
        input_names=['in_files'],
        output_names=['files_1', 'files_2'],
        function=generate_unique_pairs),
                                            name='generate_mask_pairs')
    '''
    generate_aff_pairs_1 = pe.Node(niu.Function(
        input_names=['in_files'],
        output_names=['files_1', 'files_2'],
        function=generate_unique_pairs),
                                            name='generate_aff_pairs_1')
    generate_aff_pairs_2 = pe.Node(niu.Function(
        input_names=['in_files'],
        output_names=['files_1', 'files_2'],
        function=generate_unique_pairs),
                                            name='generate_aff_pairs_2')

    '''
    *****************************************************************************
    Fourth step: Calculate the cumulated input affine transformations
    *****************************************************************************
    '''

    invert_affine_transformations = pe.MapNode(niftyreg.RegTransform(),
                                               name = 'invert_affine_transformations',
                                               iterfield = ['inv_aff_input'])
    compose_affine_transformations = pe.MapNode(niftyreg.RegTransform(),
                                                name = 'compose_affine_transformations',
                                                iterfield = ['comp_input', 'comp_input2'])


    '''
    *****************************************************************************
    Fith step: Non linear registration of all pairs
    *****************************************************************************
    '''
    nonlinear_registration = pe.MapNode(interface = niftyreg.RegF3D(), 
                                        name='nonlinear_registration', 
                                        iterfield=['ref_file', 'flo_file', 'aff_file'])
    nonlinear_registration.inputs.vel_flag  = True
    nonlinear_registration.inputs.lncc_val  = -5
    nonlinear_registration.inputs.maxit_val = 150
    nonlinear_registration.inputs.be_val    = 0.025    



    
    '''
    *****************************************************************************
    First step: Cropping inputs according to 10 voxels surrounding the skull
    *****************************************************************************
    '''
    workflow.connect(input_node, 'in_files', register_mni_to_image, 'ref_file')
    workflow.connect(input_node, 'in_files', resample_mni_mask_to_image, 'ref_file')
    workflow.connect(register_mni_to_image, 'aff_file', resample_mni_mask_to_image, 'aff_file')
    workflow.connect(resample_mni_mask_to_image, 'res_file', dilate_image_mask, 'in_file')
    workflow.connect(input_node, 'in_files', crop_image_with_mask, 'in_file')
    workflow.connect(dilate_image_mask, 'out_file', crop_image_with_mask, 'mask_file')
    workflow.connect(crop_image_with_mask, 'out_file', resample_image_mask_to_cropped_image, 'ref_file')
    workflow.connect(register_mni_to_image, 'aff_file', resample_image_mask_to_cropped_image, 'aff_file')
    workflow.connect(crop_image_with_mask, 'out_file', bias_correction, 'in_file')
    workflow.connect(resample_image_mask_to_cropped_image, 'res_file', bias_correction, 'mask_file')

    '''
    *****************************************************************************
    Aux step: Cropping the labels
    *****************************************************************************
    '''
    
    workflow.connect(input_node, 'in_labels', find_labels, 'in_item')
    workflow.connect(input_node, 'in_files', find_labels, 'in_tosearch')
    workflow.connect(crop_image_with_mask, 'out_file', find_labels, 'in_tochoose')
    workflow.connect(find_labels, 'out_item', resample_label_to_cropped_image, 'ref_file')
    workflow.connect(input_node, 'in_labels', resample_label_to_cropped_image, 'flo_file')
    
 
    '''
    *****************************************************************************
    Second step: Group wise registration between cropped inputs
    *****************************************************************************
    '''
    workflow.connect(bias_correction, 'out_file', groupwise_registration, 'input_node.in_files')
    #workflow.connect(resample_image_mask_to_cropped_image, 'res_file', groupwise_registration, 'input_node.rmask_files')
    
    
    '''
    *****************************************************************************
    Third step: Generating N * (N - 1) pairs of images to be aligned
    *****************************************************************************
    '''
    workflow.connect(bias_correction, 'out_file', generate_image_pairs, 'in_files')
    '''
    *****************************************************************************
    I noticed the resulting displacement fields were smoother and better without providing the masks...
    *****************************************************************************
    workflow.connect(resample_image_mask_to_cropped_image, 'res_file', generate_mask_pairs, 'in_files')
    '''

    '''
    *****************************************************************************
    Fourth step: Calculate the cumulated input affine transformations
    *****************************************************************************
    '''
    workflow.connect(groupwise_registration, 'output_node.trans_files', invert_affine_transformations, 'inv_aff_input')
    workflow.connect(invert_affine_transformations, 'out_file', generate_aff_pairs_1, 'in_files')
    workflow.connect(groupwise_registration, 'output_node.trans_files', generate_aff_pairs_2, 'in_files')
    workflow.connect(generate_aff_pairs_1, 'files_1', compose_affine_transformations, 'comp_input')
    workflow.connect(generate_aff_pairs_2, 'files_2', compose_affine_transformations, 'comp_input2')
    

    '''
    *****************************************************************************
    Fith step: Non linear registration of all pairs
    *****************************************************************************
    '''

    workflow.connect(generate_image_pairs, 'files_1', nonlinear_registration, 'ref_file')
    workflow.connect(generate_image_pairs, 'files_2', nonlinear_registration, 'flo_file')
    '''
    *****************************************************************************
    I noticed the resulting displacement fields were smoother and better without providing the masks...
    *****************************************************************************
    workflow.connect(generate_mask_pairs, 'files_1', nonlinear_registration, 'rmask_file')
    workflow.connect(generate_mask_pairs, 'files_2', nonlinear_registration, 'fmask_file')
    '''
    workflow.connect(compose_affine_transformations, 'out_file', nonlinear_registration, 'aff_file')


    '''
    *****************************************************************************
    Connect the outputs
    *****************************************************************************
    '''
    output_node = pe.Node(interface = niu.IdentityInterface(
        fields=['out_files',
                'out_masks',
                'out_labels',
                'out_affs',
                'out_cpps',
                'out_invcpps']),
                          name='output_node')
    workflow.connect(bias_correction, 'out_file', output_node, 'out_files')
    workflow.connect(resample_image_mask_to_cropped_image, 'res_file', output_node, 'out_masks')
    workflow.connect(resample_label_to_cropped_image, 'res_file', output_node, 'out_labels')
    workflow.connect(compose_affine_transformations, 'out_file', output_node, 'out_affs')
    workflow.connect(nonlinear_registration, 'cpp_file', output_node, 'out_cpps')
    workflow.connect(nonlinear_registration, 'invcpp_file', output_node, 'out_invcpps')

    return workflow




'''
Generates the tree structure necessary for the preprocessing steps 
of GIF.
it creates 
t1s_dir: The resulting T1 directory
labels_dir: The resulting labels directory
cpps_dir: The directory containing directories of cpps
cpps_directories: The actual cpp directories
matrices_files: all N * N matrix files (identity)
'''
def generate_gif_preprocessing_output_tree(in_T1s,
                                           in_labels,
                                           out_directory):

    import os, glob

    cpps_directories = []
    matrices_files = []

    '''
    method to create a directory safely
    '''
    def ensuredir(f):
        if not os.path.exists(f):
            print("making directory " + f)
            os.mkdir(f)
            
    ensuredir(out_directory)

    cpps_dir = os.path.join(out_directory, 'cpps')
    labels_dir = os.path.join(out_directory, 'labels')
    t1s_dir = os.path.join(out_directory, 'T1s')
    masks_dir = os.path.join(out_directory, 'masks')

    ensuredir(cpps_dir)
    ensuredir(labels_dir)
    ensuredir(t1s_dir)
    ensuredir(masks_dir)

    id_matrix_content = '1 0 0 0 \n' + \
                        '0 1 0 0 \n' + \
                        '0 0 1 0 \n' + \
                        '0 0 0 1'
    
    for t1 in in_T1s:
        basename = os.path.basename(t1)
        basename = basename.replace('.nii.gz', '')
        basename = basename.replace('.nii', '')
        cpp_dir = os.path.join(cpps_dir, basename)
        ensuredir(cpp_dir)
        cpps_directories.append(cpp_dir)

        for t1bis in in_T1s:
            basenamebis = os.path.basename(t1bis)
            basenamebis = basenamebis.replace('.nii.gz', '')
            basenamebis = basenamebis.replace('.nii', '')
            matrix_file = os.path.join(cpp_dir, basenamebis + '.txt')
            f=open(matrix_file, 'w+')
            f.write(id_matrix_content)
            f.close()
            matrices_files.append(matrix_file)


    return \
        t1s_dir, \
        masks_dir, \
        labels_dir, \
        cpps_dir, \
        cpps_directories, \
        matrices_files



'''
Convenient function that generates the valuable substitutions 
to perform from the cpp files towards the cpp sink
based on (N * N-1) generated pairs. Assumes same order (alphanumerical order) 
between files 
'''
def find_cpp_substitutions_function(in_files, in_cpps, in_invcpps):
    import os
    
    '''
    method that generates N * (N-1) pairs of indices
    necessary for the substitutions
    '''
    def generate_indices(number_of_items):
        indices_1 = []
        indices_2 = []
        
        for i in range(number_of_items):
            for j in range(i+1, number_of_items):
                indices_1.append(i)
                indices_2.append(j)
            
        return indices_1, indices_2

    indices_1, indices_2 = generate_indices(len(in_files))

    '''
    Generates the list of basenames
    '''
    basenames = []
    for in_file in in_files:
        basename = os.path.basename(in_file)
        basenames.append(basename.replace('.nii.gz', ''))

    '''
    The replacement starts from after the mapflow directory structure
    '''
    start_index = in_cpps[0].rfind('mapflow')+8
    subs = []
    for i in range(len(in_cpps)):
        cpp = in_cpps[i]
        invcpp = in_invcpps[i]
        subs.append(( cpp[start_index:],    os.path.join(basenames[indices_1[i]], basenames[indices_2[i]] + '.nii.gz') ))
        subs.append(( invcpp[start_index:], os.path.join(basenames[indices_2[i]], basenames[indices_1[i]] + '.nii.gz') ))
    
    return subs




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

'''
Create the META-workflow that will create the cpps and cropped T1s and labels
The only step after that workflow is the GIF propagation
'''
def create_seg_gif_template_database_workflow_1(name='gif_create_template_library_1', 
                                                ref_file = mni_template, 
                                                ref_mask = mni_template_mask,
                                                number_of_affine_iterations = 7):

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_T1s_directory',
                    'in_labels_directory',
                    'out_directory']),
        name='input_node')


    '''
    *****************************************************************************
    First step: Grab the T1s and labels images
    *****************************************************************************
    '''

    grabber_images = pe.Node(interface = nio.DataGrabber(outfields=['images']), 
                      name = 'grabber_images')
    grabber_images.inputs.template = '*.nii*'
    grabber_images.inputs.sort_filelist = True
    
    grabber_labels = pe.Node(interface = nio.DataGrabber(outfields=['labels']), 
                      name = 'grabber_labels')
    grabber_labels.inputs.template = '*.nii*'
    grabber_labels.inputs.sort_filelist = True

    workflow.connect(input_node, 'in_T1s_directory', grabber_images, 'base_directory')
    workflow.connect(input_node, 'in_labels_directory', grabber_labels, 'base_directory')

    '''
    *****************************************************************************
    Second step: generates the output cpp directories and fill them with 
    fake id matices
    *****************************************************************************
    '''

    generate_input_tree = pe.Node(niu.Function(
        input_names = ['in_T1s',
                       'in_labels',
                       'out_directory'],
        output_names = ['t1s_dir', 
                        'masks_dir', 
                        'labels_dir',
                        'cpps_dir',
                        'cpps_directories',
                        'matrices_files'],
        function = generate_gif_preprocessing_output_tree),
                                          name = 'generate_input_tree')

    workflow.connect(grabber_images, 'images', generate_input_tree, 'in_T1s')
    workflow.connect(grabber_labels, 'labels', generate_input_tree, 'in_labels')
    workflow.connect(input_node, 'out_directory', generate_input_tree, 'out_directory')

    '''
    *****************************************************************************
    Third step: Bias Correct / Crop / register input T1s and perform 
    pairwise non-linear registrations
    *****************************************************************************
    '''
    
    preprocessing = preprocessing_inputs_pipeline(name = 'preprocessing', 
                                                  number_of_affine_iterations = number_of_affine_iterations,
                                                  ref_file = ref_file,
                                                  ref_mask = ref_mask)
    workflow.connect(grabber_images, 'images', preprocessing, 'input_node.in_files')
    workflow.connect(grabber_labels, 'labels', preprocessing, 'input_node.in_labels')
    
    '''
    *****************************************************************************
    Fourth step: Sink the results in the output directories
    *****************************************************************************
    '''

    t1_sink = pe.Node(nio.DataSink(parameterization=True),
                      name = 't1_sink')
    workflow.connect(generate_input_tree, 't1s_dir', t1_sink, 'base_directory') 
    workflow.connect(preprocessing, 'output_node.out_files', t1_sink, '@t1s')
    find_substitutions_files = pe.Node(niu.Function(
        input_names = ['in_files', 'out_files'],
        output_names = ['substitutions'],
        function = find_preprocessing_substitutions),
                                     name = 'find_substitutions_files')
    workflow.connect(grabber_images, 'images', find_substitutions_files, 'in_files')
    workflow.connect(preprocessing, 'output_node.out_files', find_substitutions_files, 'out_files')
    workflow.connect(find_substitutions_files, 'substitutions', t1_sink, 'substitutions')

    masks_sink = pe.Node(nio.DataSink(parameterization=True),
                      name = 'masks_sink')
    workflow.connect(generate_input_tree, 'masks_dir', masks_sink, 'base_directory') 
    workflow.connect(preprocessing, 'output_node.out_masks', masks_sink, '@masks') 
    find_substitutions_masks = pe.Node(niu.Function(
        input_names = ['in_files', 'out_files'],
        output_names = ['substitutions'],
        function = find_preprocessing_substitutions),
                                     name = 'find_substitutions_masks')
    workflow.connect(grabber_images, 'images', find_substitutions_masks, 'in_files')
    workflow.connect(preprocessing, 'output_node.out_masks', find_substitutions_masks, 'out_files')
    workflow.connect(find_substitutions_masks, 'substitutions', masks_sink, 'substitutions')
    
    label_sink = pe.Node(nio.DataSink(parameterization=True),
                         name = 'label_sink')
    workflow.connect(generate_input_tree, 'labels_dir', label_sink, 'base_directory') 
    workflow.connect(preprocessing, 'output_node.out_labels', label_sink, '@labels') 
    find_substitutions_labels = pe.Node(niu.Function(
        input_names = ['in_files', 'out_files'],
        output_names = ['substitutions'],
        function = find_preprocessing_substitutions),
                                     name = 'find_substitutions_labels')
    workflow.connect(grabber_labels, 'labels', find_substitutions_labels, 'in_files')
    workflow.connect(preprocessing, 'output_node.out_labels', find_substitutions_labels, 'out_files')
    workflow.connect(find_substitutions_labels, 'substitutions', label_sink, 'substitutions')

    find_cpp_substitutions = pe.Node(niu.Function(
        input_names = ['in_files', 'in_cpps', 'in_invcpps'],
        output_names = ['substitutions'],
        function = find_cpp_substitutions_function),
                                     name = 'find_cpp_substitutions')
    
    cpp_sink = pe.Node(nio.DataSink(parameterization=True),
                       name = 'cpp_sink')
    workflow.connect(generate_input_tree, 'cpps_dir', cpp_sink, 'base_directory') 
    workflow.connect(preprocessing, 'output_node.out_cpps', cpp_sink, '@cpps') 
    
    workflow.connect(grabber_images, 'images', find_cpp_substitutions, 'in_files') 
    workflow.connect(preprocessing, 'output_node.out_cpps', find_cpp_substitutions, 'in_cpps') 
    workflow.connect(preprocessing, 'output_node.out_invcpps', find_cpp_substitutions, 'in_invcpps') 
    workflow.connect(find_cpp_substitutions, 'substitutions', cpp_sink, 'substitutions') 

    return workflow








'''
Generates the tree structure necessary for the postprocessing steps 
of GIF.
it creates each db folder and in each one:
db_xml_file
db_labels_xml_file
db_labels_link_file
'''
def generate_gif_postprocessing_output_tree(in_t1s_directory,
                                            in_labels,
                                            out_directory,
                                            number_of_iterations):

    import os, glob
    import neuromorphometricslabels

    db_directories = []
    db_xml_files = []
    db_labels_xml_files = []
    db_labels_link_files = []

    def ensuredir(f):
        if not os.path.exists(f):
            print("making directory " + f)
            os.mkdir(f)
            
    def createlink(target, link):
        if not os.path.exists(link):
            if not os.path.exists(target):
                print("target does not exist " + target)
            else:
                if not os.path.exists(link):
                    print("making link " + link)
                    os.symlink(target, link)

    ensuredir(out_directory)
    
    for iteration in range(number_of_iterations):
        
        database = os.path.join(out_directory, 'db' + str(iteration))
        
        print("creating database at " + database)
        
        ensuredir(database)
        db_directories.append(database)

        labelxmlfile=os.path.join(database, "labels.xml")
        databasexmlfile=os.path.join(database, "db.xml")
        labelsdir=os.path.join(database, "labels")
        priorsdir=os.path.join(database, "priors")
        tissuesdir=os.path.join(database, "tissues")
        geosdir=os.path.join(database, "labels_geo")
        databasexmlcontent='<?xml version="1.0"?> \n' + \
            '<document>\n' + \
            '<data>\n' + \
            '<fname>T1s</fname>\n' + \
            '<path>' + os.path.basename(in_t1s_directory) + '</path>\n' + \
            '<descr>T1 MRI Data</descr>\n' + \
            '<sform>1</sform>\n' + \
            '</data>\n' + \
            '\n' + \
            '<info>\n' + \
            '<fname>labels</fname>\n' + \
            '<path>labels</path>\n' + \
            '<gpath>labels_geo</gpath>\n' + \
            '<descr>Segmentation</descr>\n' + \
            '<extra>labels.xml</extra>\n' + \
            '<type>LabelSeg</type>\n' + \
            '</info>\n' + \
            '\n' + \
            '</document>'
        
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
        createlink(in_t1s_directory, os.path.join(database,os.path.basename(in_t1s_directory)))
        
        labelslinkslist = []
        for label in in_labels:
            filename=os.path.basename(label)
            link=os.path.join(labelsdir,filename)
            createlink(label, link)
            labelslinkslist.append(link)
            
        if (iteration == 0):
            first_db_file = databasexmlfile
        db_xml_files.append(databasexmlfile)
        db_labels_xml_files.append(labelxmlfile)
        db_labels_link_files.append(labelslinkslist)



    return \
        first_db_file, \
        db_directories, \
        db_xml_files, \
        db_labels_xml_files, \
        db_labels_link_files



'''
Create the META-workflow that will propagate the labels iteratively
'''
def create_seg_gif_template_database_workflow_2(name='gif_create_template_library_2', number_of_iterations = 1):

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_T1s_directory',
                    'in_masks_directory',
                    'in_labels_directory',
                    'in_cpps_directory',
                    'out_databases_directory']),
        name='input_node')


    '''
    *****************************************************************************
    First step: grab the images / labels / masks / cpps
    *****************************************************************************
    '''

    grabber_images = pe.Node(interface = nio.DataGrabber(outfields=['images']), 
                      name = 'grabber_images')
    grabber_images.inputs.template = '*.nii*'
    grabber_images.inputs.sort_filelist = True
    
    grabber_masks = pe.Node(interface = nio.DataGrabber(outfields=['masks']), 
                      name = 'grabber_masks')
    grabber_masks.inputs.template = '*.nii*'
    grabber_masks.inputs.sort_filelist = True
    
    grabber_labels = pe.Node(interface = nio.DataGrabber(outfields=['labels']), 
                      name = 'grabber_labels')
    grabber_labels.inputs.template = '*.nii*'
    grabber_labels.inputs.sort_filelist = True

    grabber_cpps = pe.Node(interface = nio.DataGrabber(outfields=['cpps']), 
                      name = 'grabber_cpps')
    grabber_cpps.inputs.template = '*'
    grabber_cpps.inputs.sort_filelist = True

    workflow.connect(input_node, 'in_T1s_directory', grabber_images, 'base_directory')
    workflow.connect(input_node, 'in_masks_directory', grabber_masks, 'base_directory')
    workflow.connect(input_node, 'in_labels_directory', grabber_labels, 'base_directory')
    workflow.connect(input_node, 'in_cpps_directory', grabber_cpps, 'base_directory')

    '''
    *****************************************************************************
    Second step: Create the output tree architecture holding the databases
    *****************************************************************************
    '''

    generate_output_tree = pe.Node(niu.Function(
        input_names = ['in_t1s_directory',
                       'in_labels',
                       'out_directory',
                       'number_of_iterations'],
        output_names = ['first_db_file',
                        'db_directories',
                        'db_xml_files', 
                        'db_labels_xml_files', 
                        'db_labels_link_files'],
        function = generate_gif_postprocessing_output_tree),
                                          name = 'generate_output_tree')

    workflow.connect(input_node, 'in_T1s_directory', generate_output_tree, 'in_t1s_directory')
    workflow.connect(grabber_labels, 'labels', generate_output_tree, 'in_labels')
    workflow.connect(input_node, 'out_databases_directory', generate_output_tree, 'out_directory')
    generate_output_tree.inputs.number_of_iterations = number_of_iterations


    '''
    *****************************************************************************
    Third step: Creating the list of GIF propagation steps and link them to another
    *****************************************************************************
    '''

    list_of_gifs = []
    for i in range(1, number_of_iterations):
        gif_propagation = gif.create_niftyseg_gif_propagation_pipeline_simple(name = 'gif_propagation_'+str(i))        
        workflow.connect (grabber_images, 'images', gif_propagation, 'input_node.in_file')
        workflow.connect (grabber_cpps, 'cpps', gif_propagation, 'input_node.cpp_directory')
        workflow.connect (grabber_masks, 'masks', gif_propagation, 'input_node.mask_file')
        #workflow.connect (generate_output_tree, 'out', 'input_node.out_directory')
        if i == 1:
            workflow.connect (generate_output_tree, 'first_db_file', gif_propagation, 'input_node.template_db_file')
        else:
            workflow.connect (list_of_gifs[i-1], 'output_node.out_db', gif_propagation, 'input_node.template_db_file')        
        list_of_gifs.append(gif_propagation)

    return workflow


