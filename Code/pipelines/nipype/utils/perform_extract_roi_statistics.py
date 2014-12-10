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

def print_array_function(in_array):
    import numpy as np
    import os
    print(in_array)
    array_file = 'statistics.csv'
    np.savetxt(array_file, in_array, '%u %5.2f %5.2f %5.2f')
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
                    type=str, nargs='+',
                    metavar='input_par', 
                    help='Parcelation image or list of parcelation images',
                    required=True)

parser.add_argument('-o', '--output_dir',
                    dest='output_dir', 
                    type=str, \
                    metavar='directory', 
                    help='Output directory containing the statistics results',
                    default=os.getcwd(), 
                    required=False)

args = parser.parse_args()

workflow = pe.Workflow('workflow')
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
    input_names = ['in_array'],
    output_names = ['out_file'],
    function=print_array_function),
                         name='print_array',
                         iterfield = ['in_array']
)

extract_roi_stats.inputs.in_file = input_files
extract_roi_stats.inputs.roi_file = par_files
workflow.connect(extract_roi_stats, 'out_array', print_array, 'in_array')

workflow.write_graph(graph2use='colored')
workflow.run(plugin='MultiProc')



