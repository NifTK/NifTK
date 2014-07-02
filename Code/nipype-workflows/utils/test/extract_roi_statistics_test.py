#!/usr/bin/env python

from    extract_roi_statistics import ExtractRoiStatistics
import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import os

def print_array_function(in_array):
    print(in_array)
    return in_array

parser = argparse.ArgumentParser(description='ROI statistics computation test case')
parser.add_argument('-i', '--img',dest='input_img', type=str, nargs='+', \
    metavar='input_img', help='Image file or list of input images', required=True)
parser.add_argument('--par',dest='input_par', type=str, nargs='+', \
    metavar='input_seg', help='Parcelation image or list of parcelation images', \
    required=True)
args = parser.parse_args()

pipeline = pe.Workflow('workflow')
pipeline.base_dir = os.getcwd()
input_node =  pe.Node(niu.IdentityInterface(
            fields=['in_file', 'roi_file']),
                        name='input_node')
input_node.inputs.in_file = args.input_img[0]
input_node.inputs.roi_file = args.input_par[0]



extract_roi_stats = pe.Node(interface=ExtractRoiStatistics(),
                            name='extract_roi_stats')

print_array = pe.Node(interface = 
                                niu.Function(input_names = ['in_array'],
                                             output_names = [],
                                             function=print_array_function),
                                name='print_array')

                        
pipeline.connect(input_node, 'in_file', extract_roi_stats, 'in_file')
pipeline.connect(input_node, 'roi_file', extract_roi_stats, 'roi_file')

pipeline.connect(extract_roi_stats, 'out_array', print_array, 'in_array')

pipeline.write_graph(graph2use='colored')
pipeline.run()



