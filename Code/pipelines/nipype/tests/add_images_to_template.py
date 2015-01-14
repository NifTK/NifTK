#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import argparse
import os
from distutils import spawn
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyseg       as niftyseg

import niftk
import seg_gif_create_template_library  as seggif



mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')


'''
Convenient function that generates the substitutions 
between files 
'''
def find_preprocessing_substitutions(out_file, basename, extension):
    import os
    subs = []
    subs.append(( os.path.basename(out_file), basename+extension ))
    return subs



'''
The function returns a workflow that prepares the input.
it does:
1. bias correction of all inputs
2. Cropping input according to 10 voxels surrounding the skull
3. Generating N pairs of images to be aligned
4. Non linear registration of all pairs

Outputs are:

1. the output file cropped and bias corrected
2. the corresponding mask
3. the affine matrice to the average,
4. the N non linear cpp files
5. the N non linear inv_cpp files
'''

def preprocessing_input_pipeline(name='preprocessing_inputs_pipeline', number_of_affine_iterations = 7, ref_file = mni_template, ref_mask = mni_template_mask):

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    input_node = pe.Node(
        interface = niu.IdentityInterface(fields=['in_file', 'in_images', 'in_affines']),
        name='input_node')

    '''
    *****************************************************************************
    First step: Cropping inputs according to 10 voxels surrounding the skull
    *****************************************************************************
    '''
    register_mni_to_image = pe.Node(interface = niftyreg.RegAladin(),
                                    name = 'register_mni_to_image')
    register_mni_to_image.inputs.flo_file = mni_template
    resample_mni_mask_to_image = pe.Node(interface = niftyreg.RegResample(),
                                         name = 'resample_mni_mask_to_image')
    resample_mni_mask_to_image.inputs.inter_val = 'NN'
    resample_mni_mask_to_image.inputs.flo_file = mni_template_mask

    dilate_image_mask = pe.Node(interface = niftyseg.BinaryMaths(),
                                name = 'dilate_image_mask')
    dilate_image_mask.inputs.operation = 'dil'
    dilate_image_mask.inputs.operand_value = 10
    
    crop_image_with_mask = pe.Node(interface = niftk.CropImage(), 
                                   name='crop_image_with_mask')

    resample_image_mask_to_cropped_image = pe.Node(interface = niftyreg.RegResample(),
                                                   name = 'resample_image_mask_to_cropped_image')
    resample_image_mask_to_cropped_image.inputs.inter_val = 'NN'
    resample_image_mask_to_cropped_image.inputs.flo_file = mni_template_mask

    bias_correction = pe.Node(interface = niftk.N4BiasCorrection(),
                                       name = 'bias_correction')
    bias_correction.inputs.in_downsampling=2
    bias_correction.inputs.in_maxiter=1000
    bias_correction.inputs.in_convergence=0.000100
    bias_correction.inputs.in_fwhm=0.050000

    '''
    *****************************************************************************
    Second step: Calculate the cumulated input affine transformations
    *****************************************************************************
    '''
    register_mni_to_cropped_image = pe.Node(interface = niftyreg.RegAladin(),
                                    name = 'register_mni_to_cropped_image')
    register_mni_to_cropped_image.inputs.ref_file = mni_template
    
    invert_affine_transformations = pe.Node(niftyreg.RegTransform(),
                                            name = 'invert_affine_transformations',
                                            iterfield = ['inv_aff_input'])
    compose_affine_transformations = pe.MapNode(niftyreg.RegTransform(),
                                                name = 'compose_affine_transformations',
                                                iterfield = ['comp_input2'])


    '''
    *****************************************************************************
    Third step: Non linear registration of all pairs
    *****************************************************************************
    '''
    nonlinear_registration = pe.MapNode(interface = niftyreg.RegF3D(), 
                                        name='nonlinear_registration', 
                                        iterfield=['flo_file', 'aff_file'])
    nonlinear_registration.inputs.vel_flag  = True
    nonlinear_registration.inputs.lncc_val  = -5
    nonlinear_registration.inputs.maxit_val = 150
    nonlinear_registration.inputs.be_val    = 0.025

    
    '''
    *****************************************************************************
    First step: Cropping inputs according to 10 voxels surrounding the skull
    *****************************************************************************
    '''
    workflow.connect(input_node, 'in_file', register_mni_to_image, 'ref_file')
    workflow.connect(input_node, 'in_file', resample_mni_mask_to_image, 'ref_file')
    workflow.connect(register_mni_to_image, 'aff_file', resample_mni_mask_to_image, 'aff_file')
    workflow.connect(resample_mni_mask_to_image, 'res_file', dilate_image_mask, 'in_file')
    workflow.connect(input_node, 'in_file', crop_image_with_mask, 'in_file')
    workflow.connect(dilate_image_mask, 'out_file', crop_image_with_mask, 'mask_file')
    workflow.connect(crop_image_with_mask, 'out_file', resample_image_mask_to_cropped_image, 'ref_file')
    workflow.connect(register_mni_to_image, 'aff_file', resample_image_mask_to_cropped_image, 'aff_file')
    workflow.connect(crop_image_with_mask, 'out_file', bias_correction, 'in_file')
    workflow.connect(resample_image_mask_to_cropped_image, 'res_file', bias_correction, 'mask_file')
    

    '''
    *****************************************************************************
    Fourth step: Calculate the cumulated input affine transformations
    *****************************************************************************
    '''
    workflow.connect(bias_correction, 'out_file', register_mni_to_cropped_image, 'flo_file')
    workflow.connect(register_mni_to_cropped_image, 'aff_file', invert_affine_transformations, 'inv_aff_input')
    workflow.connect(invert_affine_transformations, 'out_file', compose_affine_transformations, 'comp_input')
    workflow.connect(input_node, 'in_affines', compose_affine_transformations, 'comp_input2')
    

    '''
    *****************************************************************************
    Fith step: Non linear registration of all pairs
    *****************************************************************************
    '''

    workflow.connect(bias_correction, 'out_file', nonlinear_registration, 'ref_file')
    workflow.connect(input_node, 'in_images', nonlinear_registration, 'flo_file')
    workflow.connect(compose_affine_transformations, 'out_file', nonlinear_registration, 'aff_file')
    

    '''
    *****************************************************************************
    Connect the outputs
    *****************************************************************************
    '''
    output_node = pe.Node(interface = niu.IdentityInterface(
        fields=['out_file',
                'out_mask',
                'out_aff',
                'out_cpps',
                'out_invcpps']),
                          name='output_node')
    workflow.connect(bias_correction, 'out_file', output_node, 'out_file')
    workflow.connect(resample_image_mask_to_cropped_image, 'res_file', output_node, 'out_mask')
    workflow.connect(register_mni_to_cropped_image, 'aff_file', output_node, 'out_aff')
    workflow.connect(nonlinear_registration, 'cpp_file', output_node, 'out_cpps')
    workflow.connect(nonlinear_registration, 'invcpp_file', output_node, 'out_invcpps')

    return workflow


parser = argparse.ArgumentParser(description='GIF Template Creation')
parser.add_argument('-i', '--input',
                    dest='input',
                    metavar='input',
                    help='Input image',
                    required=True)
parser.add_argument('--t1s',
                    dest='t1s',
                    metavar='t1s',
                    help='Existing T1s',
                    nargs='+',
                    required=True)
parser.add_argument('--affs',
                    dest='affs',
                    metavar='affs',
                    help='Existing Affines',
                    nargs='+',
                    required=True)
parser.add_argument('--t1_dir',
                    dest='t1_dir',
                    metavar='t1_dir',
                    help='Output T1 directory',
                    required=True)
parser.add_argument('--aff_dir',
                    dest='aff_dir',
                    metavar='aff_dir',
                    help='Output Affines directory',
                    required=True)
parser.add_argument('--mask_dir',
                    dest='mask_dir',
                    metavar='mask_dir',
                    help='Output Masks directory',
                    required=True)
parser.add_argument('--cpp_dir',
                    dest='cpp_dir',
                    metavar='cpp_dir',
                    help='Output Cpps directory',
                    required=True)


args = parser.parse_args()

basename = os.path.basename(args.input)
basename = basename.replace('.nii.gz', '')
basename = basename.replace('.nii', '')

t1s = [ os.path.abspath(f) for f in args.t1s ]
affs = [ os.path.abspath(f) for f in args.affs ]

input_cpp_dir = os.path.join(os.path.abspath(args.cpp_dir), basename)
if not os.path.exists(input_cpp_dir):
    os.mkdir(input_cpp_dir)

basedir = os.getcwd()

r = preprocessing_input_pipeline (name = 'add_image_to_template', 
                                  ref_file = mni_template, 
                                  ref_mask = mni_template_mask)

r.base_dir = basedir

r.inputs.input_node.in_file = os.path.abspath(args.input)
r.inputs.input_node.in_images = t1s
r.inputs.input_node.in_affines = affs

t1_sink = pe.Node(nio.DataSink(parameterization=False),
                  name = 't1_sink')
t1_sink.inputs.base_directory = os.path.abspath(args.t1_dir)

find_substitutions_file = pe.Node(niu.Function(
    input_names = ['out_file', 'basename', 'extension'],
    output_names = ['substitutions'],
    function = find_preprocessing_substitutions),
                                   name = 'find_substitutions_file')
find_substitutions_file.inputs.basename = basename
find_substitutions_file.inputs.extension = '.nii.gz'
r.connect(r.get_node('output_node'), 'out_file', t1_sink, '@t1')
r.connect(r.get_node('output_node'), 'out_file', find_substitutions_file, 'out_file')
r.connect(find_substitutions_file, 'substitutions', t1_sink, 'substitutions')

aff_sink = pe.Node(nio.DataSink(parameterization=False),
                  name = 'aff_sink')
aff_sink.inputs.base_directory = os.path.abspath(args.aff_dir)

find_substitutions_aff = pe.Node(niu.Function(
    input_names = ['out_file', 'basename', 'extension'],
    output_names = ['substitutions'],
    function = find_preprocessing_substitutions),
                                   name = 'find_substitutions_aff')
find_substitutions_aff.inputs.basename = basename
find_substitutions_aff.inputs.extension = '.txt'
r.connect(r.get_node('output_node'), 'out_aff', aff_sink, '@aff')
r.connect(r.get_node('output_node'), 'out_aff', find_substitutions_aff, 'out_file')
r.connect(find_substitutions_aff, 'substitutions', aff_sink, 'substitutions')

mask_sink = pe.Node(nio.DataSink(parameterization=False),
                  name = 'mask_sink')
mask_sink.inputs.base_directory = os.path.abspath(args.mask_dir)

find_substitutions_mask = pe.Node(niu.Function(
    input_names = ['out_file', 'basename', 'extension'],
    output_names = ['substitutions'],
    function = find_preprocessing_substitutions),
                                   name = 'find_substitutions_mask')
find_substitutions_mask.inputs.basename = basename
find_substitutions_mask.inputs.extension = '.nii.gz'
r.connect(r.get_node('output_node'), 'out_mask', mask_sink, '@mask')
r.connect(r.get_node('output_node'), 'out_mask', find_substitutions_mask, 'out_file')
r.connect(find_substitutions_mask, 'substitutions', mask_sink, 'substitutions')


cpp_sink = pe.Node(nio.DataSink(parameterization=False),
                  name = 'cpp_sink')
cpp_sink.inputs.base_directory = os.path.abspath(input_cpp_dir)

r.connect(r.get_node('output_node'), 'out_cpps', cpp_sink, '@cpp')
r.connect(r.get_node('output_node'), 'out_invcpps', cpp_sink, '@invcpp')

dot_exec=spawn.find_executable('dot')   
if not dot_exec == None:
    r.write_graph(graph2use='hierarchical')

qsub_exec=spawn.find_executable('qsub')

# Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
try:    
    qsubargs=os.environ['QSUB_OPTIONS']
except KeyError:                
    print 'The environtment variable QSUB_OPTIONS is not set up, we cannot queue properly the process. Using the default script options.'
    qsubargs='-l h_rt=05:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    print qsubargs

# We can use qsub or not depending of this environment variable, by default we use it.
try:    
    run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
except KeyError:                
    run_qsub=True

if not qsub_exec == None and run_qsub:
    r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    r.run(plugin='MultiProc')
