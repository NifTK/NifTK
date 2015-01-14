#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import argparse
import os
from distutils import spawn
import nipype.interfaces.niftyreg       as niftyreg

import seg_gif_create_template_library  as seggif

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')

parser = argparse.ArgumentParser(description='GIF Template Creation')
parser.add_argument('-i', '--inputs',
                    dest='inputs',
                    metavar='inputs',
                    help='Input images',
                    required=True)
parser.add_argument('-l','--labels',
                    dest='labels',
                    metavar='labels',
                    help='Input initial labels',
                    required=True)
parser.add_argument('-p','--propagate',
                    dest='propagate',
                    metavar='propagate',
                    help='Propagate the labels (default:0, only do the pre-steps)',
                    required=False,
                    type=int,
                    default = 0)
parser.add_argument('-o','--output',
                    dest='output',
                    metavar='output',
                    help='output directory to which the template library is stored',
                    required=True)

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

basedir = os.getcwd()

r = seggif.create_seg_gif_template_database_workflow_1(name = 'gif_create_template_1', 
                                                       ref_file = mni_template, 
                                                       ref_mask = mni_template_mask,
                                                       number_of_affine_iterations = 1)

r.base_dir = basedir
r.inputs.input_node.in_T1s_directory = os.path.abspath(args.inputs)
r.inputs.input_node.in_labels_directory = os.path.abspath(args.labels)
r.inputs.input_node.out_directory = result_dir

# Run the overall workflow
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




if args.propagate >= 1:

    t1_dir = os.path.join(result_dir, 'T1s')
    labels_dir = os.path.join(result_dir, 'labels')
    masks_dir = os.path.join(result_dir, 'masks')
    cpps_dir = os.path.join(result_dir, 'cpps')
    
    r2 = seggif.create_seg_gif_template_database_workflow_2(name = 'gif_create_template_2', 
                                                            number_of_iterations = 2)
    
    r2.base_dir = basedir
    r2.inputs.input_node.in_T1s_directory = t1_dir
    r2.inputs.input_node.in_masks_directory = masks_dir
    r2.inputs.input_node.in_labels_directory = labels_dir
    r2.inputs.input_node.in_cpps_directory = cpps_dir
    r2.inputs.input_node.out_databases_directory = result_dir
    
    
    # Run the overall workflow
    dot_exec=spawn.find_executable('dot')   
    if not dot_exec == None:
        r2.write_graph(graph2use='hierarchical')
    
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
        r2.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
    else:
        r2.run(plugin='MultiProc')


