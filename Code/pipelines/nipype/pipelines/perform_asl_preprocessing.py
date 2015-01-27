#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import nipype.interfaces.dcm2nii        as mricron
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyseg       as niftyseg
import argparse
import os
from distutils                          import spawn

import niftk

help_message = \
'Perform Arterial Spin Labelling Fitting with pre-processing steps. \n\n' + \
'Mandatory Input is the 4D nifti image ASL sequence. \n' + \
'as well as a T1 image for reference space. \n\n' + \
'Additionally to the ASL information, the pipeline outputs \n' + \
'transformations from ASL to M0 space and transformation from M0 to T1 space. \n\n' 

parser = argparse.ArgumentParser(description=help_message)

parser.add_argument('-s', '--source',
                    dest='source',
                    required=True,
                    help='ASL sequence 4D source file')
parser.add_argument('-t', '--t1',
                    dest='t1',
                    required=False,
                    help='T1 structural image file')
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='Result directory where the output data is to be stored',
                    required=False,
                    default='results')

args = parser.parse_args()

result_dir = os.path.abspath(args.output)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

r = niftk.asl.create_asl_processing_workflow(name = 'asl_workflow')
r.base_dir = os.getcwd()

r.get_node('input_node').inputs.in_source = os.path.abspath(args.source)
if args.t1:
    r.get_node('input_node').inputs.in_t1 = os.path.abspath(args.t1)
    

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = result_dir
ds.inputs.parameterization = False

r.connect(r.get_node('output_node'), 'cbf_file', ds, '@cbf_file')
r.connect(r.get_node('output_node'), 'syn_file', ds, '@syn_file')
r.connect(r.get_node('output_node'), 'error_file', ds, '@error_file')
r.connect(r.get_node('output_node'), 'asl_to_m0_transformations', ds, 'asl_to_m0_transformations')
if args.t1:
    r.connect(r.get_node('output_node'), 'm0_to_t1_transformation', ds, 'm0_to_t1_transformation')


dot_exec=spawn.find_executable('dot')
if not dot_exec == None:
    r.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')

# Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
try:    
    qsubargs=os.environ['QSUB_OPTIONS']
except KeyError:
    qsubargs='-l h_rt=03:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

# We can use qsub or not depending of this environment variable, by default we use it.
try:    
    run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
except KeyError:
    run_qsub=True

if not qsub_exec == None and run_qsub:
    r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    r.run(plugin='MultiProc')

