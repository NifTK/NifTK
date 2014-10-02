#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import argparse
import os
from distutils import spawn

import nipype.interfaces.niftyreg as niftyreg
import seg_gif_propagation as gif

def find_data_directory_function(in_db_file):
    def find_xml_data_path(in_file):
        import xml.etree.ElementTree as ET
        tree = ET.parse(in_file)
        root = tree.getroot()
        return root.findall('data')[0].findall('path')[0].text
    import os
    database_directory = os.path.dirname(in_db_file)
    data_directory = find_xml_data_path(in_db_file)
    ret = os.path.abspath(data_directory)
    if os.path.exists(ret):
        return ret
    ret = os.path.join(data_directory, data_directory)
    if os.path.exists(ret):
        return ret
    return None

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
                    nargs='+',
                    help='Mask image corresponding to inputfile',
                    required=False)

parser.add_argument('-c','--cpp',
                    dest='cpp',
                    metavar='cpp',
                    nargs='+',
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

infosource = pe.Node(niu.IdentityInterface(fields = ['inputfile', 'mask', 'cpp']),
                     name = 'infosource',
                     synchronize=True)

inputfiles = [os.path.abspath(f) for f in args.inputfile]
cpps = [os.path.abspath(f) for f in args.cpp]

if args.mask is None:
    infosource.iterables = [ ('inputfile', inputfiles), 
                             ('cpp', cpps) ]
else:
    masks = [os.path.abspath(f) for f in args.mask]
    infosource.iterables = [ ('inputfile', inputfiles), 
                             ('mask', masks), 
                             ('cpp', cpps) ]

r = gif.create_niftyseg_gif_propagation_pipeline_simple(name='gif_propagation_workflow_s')
r.base_dir = basedir

r.connect(infosource, 'inputfile', r.get_node('input_node'), 'in_file')
r.connect(infosource, 'cpp', r.get_node('input_node'), 'in_cpp_dir')

if args.mask is None:
    mni_to_input = pe.Node(interface=niftyreg.RegAladin(), 
                           name='mni_to_input')
    mni_to_input.inputs.flo_file = mni_template
    mask_resample  = pe.Node(interface = niftyreg.RegResample(), 
                             name = 'mask_resample')
    mask_resample.inputs.inter_val = 'NN'
    mask_resample.inputs.flo_file = mni_template_mask

    r.connect(infosource, 'inputfile', mni_to_input, 'ref_file')
    r.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
    r.connect(mask_resample, 'res_file', r.get_node('input_node'), 'in_mask')
    
else:
    
    r.connect(infosource, 'mask', r.get_node('input_node'), 'in_mask')
    
    
find_data_directory = pe.Node(niu.Function(
    input_names = ['in_db_file'],
    output_names = ['directory'],
    function = find_data_directory_function),
                              name = 'find_data_directory')
find_data_directory.inputs.in_db_file = os.path.abspath(args.database)
r.connect(find_data_directory, 'directory', r.get_node('input_node'), in_t1s_dir)

r.inputs.input_node.in_db_file = os.path.abspath(args.database)

r.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')    
if not qsub_exec == None:
    qsubargs='-l h_rt=05:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    r.run(plugin='MultiProc')
    
    
