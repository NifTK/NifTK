#!/usr/bin/env python

from interslice_correlation_plot import InterSliceCorrelationPlot
import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import os

parser = argparse.ArgumentParser(description='Interslice correlation_plot test case')
parser.add_argument('-i', '--img',dest='input_img', type=str, \
                    help='Input image file', required=True)
parser.add_argument('-b', '--bval',dest='bval', type=str, \
                    help='Input bval file', required=True)
parser.add_argument('-o', '--out',dest='output_folder', type=str, \
                    help='Ouptut folder', required=True)
args = parser.parse_args()

pipeline = pe.Workflow('InterSliceCorrelationPlot')
pipeline.base_dir = os.path.abspath(args.output_folder)
input_node =  pe.Node(niu.IdentityInterface(fields=['in_file','bval']),
                      name='input_node')
input_node.inputs.in_file = os.path.abspath(args.input_img)
input_node.inputs.bval = os.path.abspath(args.bval)

interslice_correlation_plot = pe.Node(interface=InterSliceCorrelationPlot(),
                                      name='interslice_correlation_plot')
                                      
pipeline.connect(input_node, 'in_file', interslice_correlation_plot, 'in_file')
pipeline.connect(input_node, 'bval', interslice_correlation_plot, 'bval_file')

ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
ds.inputs.base_directory = os.path.abspath(os.path.abspath(args.output_folder))
pipeline.connect(interslice_correlation_plot, 'out_file', ds, '@out')

pipeline.write_graph(graph2use='colored')
pipeline.run()



