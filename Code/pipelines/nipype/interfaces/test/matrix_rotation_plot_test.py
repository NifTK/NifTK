#!/usr/bin/env python

from matrix_rotation_plot import MatrixRotationPlot
import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import os

def generate_matrices(number):
    import numpy as np
    import os
    list_matrices = []
    for i in range(0,number):
        matrix=np.random.rand(4,4)/5 + np.eye(4,4)
        matrix_name=os.getcwd()+'/matrix_'+str(i)+'.txt'
        np.savetxt(matrix_name, matrix)
        list_matrices.append(matrix_name)
    return list_matrices

parser = argparse.ArgumentParser(description='Interslice correlation_plot test case')
parser.add_argument('-n', '--num',dest='matrix_number', type=int, \
                    help='Number of input matrices to generate', required=True)
parser.add_argument('-o', '--out',dest='output_folder', type=str, \
                    help='Ouptut folder', required=True)
args = parser.parse_args()

pipeline = pe.Workflow('MatrixRotationPlot')
pipeline.base_dir = os.path.abspath(args.output_folder)

generate_list_matrices = pe.Node(interface =
                             niu.Function(input_names = ['number'],
                                          output_names = ['out_list'],
                                          function=generate_matrices),
                             name='generate_list_matrices')
generate_list_matrices.inputs.number=args.matrix_number

matrix_rotation_plot = pe.Node(interface=MatrixRotationPlot(),
                                      name='matrix_rotation_plot')
pipeline.connect(generate_list_matrices, 'out_list', matrix_rotation_plot, 'in_files')

ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
ds.inputs.base_directory = os.path.abspath(os.path.abspath(args.output_folder))
pipeline.connect(matrix_rotation_plot, 'out_file', ds, '@out')

pipeline.write_graph(graph2use='colored')
pipeline.run()



