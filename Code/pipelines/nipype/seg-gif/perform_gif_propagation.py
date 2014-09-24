#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import argparse
import os
from distutils import spawn

import nipype.interfaces.niftyreg as niftyreg
import seg_gif_propagation as gif

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil1.nii.gz')

parser = argparse.ArgumentParser(description='GIF Propagation')

parser.add_argument('-i', '--inputfile',
                    dest='inputfile',
                    metavar='inputfile',
                    nargs='+',
                    help='Input target image to propagate labels in',
                    required=True)

parser.add_argument('-m','--mask',
                    dest='mask',
                    metavar='mask',
                    help='Mask image corresponding to inputfile',
                    required=True)

parser.add_argument('-c','--cpp',
                    dest='cpp',
                    metavar='cpp',
                    help='cpp directory to store/read cpp files related to inputfile',
                    required=True)

parser.add_argument('-d','--database',
                    dest='database',
                    metavar='database',
                    help='gif-based database xml file describing the inputs',
                    required=True)

parser.add_argument('-o','--output',
                    dest='output',
                    metavar='output',
                    help='output directory to which the gif outputs are stored',
                    required=True)

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

basedir = os.getcwd()

inputfiles = [os.path.abspath(f) for f in args.inputfile]
masks = [os.path.abspath(f) for f in args.mask]
cpps = [os.path.abspath(f) for f in args.cpp]

infosource = pe.Node(niu.IdentityInterface(fields = ['inputfile']),
                     name = 'infosource',
                     synchronize=True)
infosource.iterables = [ ('inputfile', inputfiles), 
                         ('mask', masks), 
                         ('cpp', cpps) ]
                        

r = gif.create_niftyseg_gif_propagation_pipeline_simple(name='gif_propagation_workflow_s')
r.base_dir = basedir
r.connect(infosource, 'inputfile', r.get_node('input_node'), 'in_file')
r.connect(infosource, 'mask', r.get_node('input_node'), 'in_mask_file')
r.connect(infosource, 'cpp', r.get_node('input_node'), 'in_cpp_dir')
r.inputs.input_node.in_db_file = os.path.abspath(args.database)
r.inputs.input_node.out_dir = result_dir

r.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')    
if not qsub_exec == None:
    qsubargs='-l h_rt=05:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    r.run(plugin='MultiProc')
    
    
