#! /usr/bin/env python

import seg_gif_create_template_library as seggif
import glob, os

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import seg_gif_propagation as gif
import argparse
import os
import nipype.interfaces.niftyreg as niftyreg

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

r = seggif.create_seg_gif_create_template_database_workflow(name = 'gif_create_template', 
                                                            number_of_iterations = 0, 
                                                            ref_file = mni_template, 
                                                            ref_mask = mni_template_mask)

r.base_dir = basedir
r.inputs.input_node.in_entries_directory = os.path.abspath(args.inputs)
r.inputs.input_node.in_initial_labels_directory = os.path.abspath(args.labels)
r.inputs.input_node.out_database_directory = result_dir

r.write_graph(graph2use='hierarchical')

qsubargs='-l h_rt=01:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
r.run(plugin='SGE', plugin_args={'qsub_args': qsubargs})
