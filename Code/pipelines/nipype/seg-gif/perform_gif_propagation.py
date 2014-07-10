#! /usr/bin/env python

use_simple = False

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import seg_gif_propagation as gif
import argparse
import os
import nipype.interfaces.niftyreg as niftyreg

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil1.nii.gz')

parser = argparse.ArgumentParser(description='GIF Propagation')
parser.add_argument('-i', '--inputfile',
                    dest='inputfile',
                    metavar='inputfile',
                    help='Input target image to propagate labels in',
                    required=True)
parser.add_argument('-t','--t1s',
                    dest='t1s',
                    metavar='t1s',
                    help='T1 directory of the template database',
                    required=True)
parser.add_argument('-d','--database',
                    dest='database',
                    metavar='database',
                    help='gif-based database xml file describing the inputs',
                    required=True)
parser.add_argument('-s','--simple',
                    dest='simple',
                    metavar='simple',
                    help='Use the simple version of the workflow (default: 1)',
                    required=False,
                    type=int,
                    default=0)
parser.add_argument('-o','--output',
                    dest='output',
                    metavar='output',
                    help='output directory to which the gif outputs are stored',
                    required=True)

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

cpp_dir = os.path.join(result_dir, 'cpps')
if not os.path.exists(cpp_dir):
    os.mkdir(cpp_dir)

basedir = os.getcwd()

if args.simple == 1:

    r = gif.create_niftyseg_gif_propagation_pipeline_simple(name='gif_propagation_workflow')
    r.base_dir = basedir
    r.inputs.input_node.in_file = os.path.abspath(args.inputfile)
    r.inputs.input_node.template_db_file = os.path.abspath(args.database)
    r.inputs.input_node.out_directory = result_dir
    r.inputs.input_node.cpp_directory = cpp_dir
    r.write_graph(graph2use='colored')
    r.run('MultiProc')

else:

    mni_to_input = pe.Node(interface=niftyreg.RegAladin(), name='mni_to_input')
    mni_to_input.inputs.ref_file = os.path.abspath(args.inputfile)
    mni_to_input.inputs.flo_file = mni_template
    
    mask_resample  = pe.Node(interface = niftyreg.RegResample(), name = 'mask_resample')
    mask_resample.inputs.inter_val = 'NN'
    mask_resample.inputs.ref_file = os.path.abspath(args.inputfile)
    mask_resample.inputs.flo_file = mni_template_mask
    
    r = gif.create_niftyseg_gif_propagation_pipeline(name='gif_propagation_workflow')
    r.base_dir = basedir
    r.inputs.input_node.in_file = os.path.abspath(args.inputfile)
    r.inputs.input_node.template_db_file = os.path.abspath(args.database)
    r.inputs.input_node.out_res_directory = result_dir
    r.inputs.input_node.out_cpp_directory = cpp_dir
    r.inputs.input_node.template_T1s_directory = os.path.abspath(args.t1s)

    r.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
    r.connect(mask_resample, 'res_file', r.get_node('input_node'), 'in_mask')
    r.write_graph(graph2use='colored')
    r.run('Linear')

