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

inputfiles = [os.path.abspath(f) for f in args.inputfile]

infosource = pe.Node(niu.IdentityInterface(fields = ['inputfile']),
                     name = 'infosource')
infosource.iterables = ('inputfile', inputfiles)

print inputfiles[0]

if args.simple == 1:

    r = gif.create_niftyseg_gif_propagation_pipeline_simple(name='gif_propagation_workflow_s')
    r.base_dir = basedir
    r.connect(infosource, 'inputfile', r.get_node('input_node'), 'in_file')
    r.inputs.input_node.in_db_file = os.path.abspath(args.database)
    r.inputs.input_node.out_dir = result_dir
    r.inputs.input_node.in_cpp_dir = cpp_dir
    r.write_graph(graph2use='colored')
    r.run('Linear')

else:

    r = gif.create_niftyseg_gif_propagation_pipeline(name='gif_propagation_workflow')
    r.base_dir = basedir
    
    mni_to_input = pe.Node(interface=niftyreg.RegAladin(), name='mni_to_input')
    r.connect(infosource, 'inputfile', mni_to_input, 'ref_file')
    mni_to_input.inputs.flo_file = mni_template
    mask_resample  = pe.Node(interface = niftyreg.RegResample(), name = 'mask_resample')
    mask_resample.inputs.inter_val = 'NN'
    r.connect(infosource, 'inputfile', mask_resample, 'ref_file')
    mask_resample.inputs.flo_file = mni_template_mask
    r.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
    
    r.connect(infosource, 'inputfile', r.get_node('input_node'), 'in_file')
    r.connect(mask_resample, 'res_file', r.get_node('input_node'), 'in_mask')
    r.inputs.input_node.in_db_file = os.path.abspath(args.database)
    r.inputs.input_node.out_cpp_dir = cpp_dir
    r.inputs.input_node.in_t1s_dir = os.path.abspath(args.t1s)

    data_sink = pe.Node(nio.DataSink(parameterization=False),
                         name = 'data_sink')
    data_sink.inputs.base_directory = result_dir
    r.connect(r.get_node('output_node'), 'out_parc_file', data_sink, 'label') 
    r.connect(r.get_node('output_node'), 'out_geo_file', data_sink, 'geo') 
    r.connect(r.get_node('output_node'), 'out_prior_file', data_sink, 'prior')
    
    find_gif_substitutions = pe.Node(interface = niu.Function(
        input_names = ['in_db_file'],
        output_names = ['substitutions'],
        function=gif.find_gif_substitutions_function),
                                     name = 'find_gif_substitutions')
    find_gif_substitutions.inputs.in_db_file = os.path.abspath(args.database)
    r.connect(find_gif_substitutions, 'substitutions', data_sink, 'regexp_substitutions')


    r.write_graph(graph2use='colored')
    
    qsub_exec=spawn.find_executable('qsub')    
    if not qsub_exec == None:
        qsubargs='-l h_rt=05:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
        r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
    else:
        r.run(plugin='MultiProc')

