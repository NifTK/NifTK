#!/usr/bin/env python

#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
from nipype                             import config, logging
from distutils                          import spawn
import sys, os
import argparse, textwrap
from extract_roi_statistics import ExtractRoiStatistics

def print_array_function(in_array, subject_id):
    import os, numpy as np
    array_file = subject_id + '.csv'
    array_format = '%u'
    for f in range(in_array.shape[1] - 1):
        array_format = array_format + ' %5.2f'    
    np.savetxt(array_file, in_array, array_format)
    return os.path.abspath(array_file)


pipelineDescription=textwrap.dedent('''\
Pipeline to perform a simple ROI statistics on an image
or a list of input images.
''')

parser = argparse.ArgumentParser(description=pipelineDescription)

parser.add_argument('-i', '--input_file',
                    dest='input_file',
                    metavar='input_file',
                    help='Input image file to calculate statistics from',
                    nargs='+',
                    required=True)

parser.add_argument('-p', '--par',
                    dest='input_par', 
                    type=str, 
                    nargs='+',
                    metavar='input_par', 
                    help='Parcelation image or list of parcelation images',
                    required=True)

parser.add_argument('-l', '--label',
                    dest='input_label', 
                    type=int,
                    nargs='+',
                    metavar='input_label', 
                    help='Specify Label(s) to extract',
                    required=False)

parser.add_argument('-o', '--output',
                    dest='output', 
                    type=str, 
                    metavar='directory', 
                    help='Output directory containing the statistics results',
                    default=os.getcwd(), 
                    required=False)

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

workflow = pe.Workflow('extract_roi_statistics')
workflow.base_dir = os.getcwd()

# extracting basename of the input file (list)
input_files = [os.path.abspath(f) for f in args.input_file]
par_files = [os.path.abspath(f) for f in args.input_par]
input_filenames = [os.path.basename(f) for f in input_files]

# extracting the 'subject list simply for iterable purposes
subject_list = [f.replace('.nii.gz','') for f in input_filenames]

extract_roi_stats = pe.MapNode(interface=ExtractRoiStatistics(),
                               name='extract_roi_stats',
                               iterfield = ['in_file', 'roi_file'])

print_array = pe.MapNode(interface = niu.Function(
    input_names = ['in_array', 'subject_id'],
    output_names = ['out_file'],
    function=print_array_function),
                         name='print_array',
                         iterfield = ['in_array', 'subject_id']
)

extract_roi_stats.inputs.in_file = input_files
extract_roi_stats.inputs.roi_file = par_files
if args.input_label:
    extract_roi_stats.inputs.in_label = args.input_label
workflow.connect(extract_roi_stats, 'out_array', print_array, 'in_array')
print_array.inputs.subject_id = subject_list

# Create a data sink    
ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
ds.inputs.base_directory = result_dir
workflow.connect(print_array, 'out_file', ds, '@statistics')

workflow.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')

# We can provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
try:
    qsubargs=os.environ['QSUB_OPTIONS']
except KeyError:
    print 'The environtment variable QSUB_OPTIONS is not set up, we cannot queue properly the process. Using the default script options.'
    qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    print qsubargs

# We can use qsub or not depending of this environment variable, by default we use it.
try:
    run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
except KeyError:
    run_qsub=True

if not qsub_exec == None and run_qsub:
    workflow.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    workflow.run(plugin='MultiProc')
