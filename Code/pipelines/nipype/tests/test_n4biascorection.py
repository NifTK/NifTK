import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import os
import argparse

import n4biascorrection

name = 'test_n4biascorrection'

parser = argparse.ArgumentParser(description=name)

parser.add_argument('-i', '--input',
                    dest='input',
                    metavar='input',
                    help='Input Image',
                    required=True)

args = parser.parse_args()

workflow = pe.Workflow(name=name)
workflow.base_output_dir=name
workflow.base_dir=name

directory = os.getcwd()

node = pe.Node(interface = n4biascorrection.N4BiasCorrection(), name = 'n4')
output_node = pe.Node(interface = niu.IdentityInterface(fields = ['out_file', 'out_biasfield_file']), name = 'output_node')
workflow.connect(node, 'out_file',  output_node, 'out_file')
workflow.connect(node, 'out_biasfield_file',   output_node, 'out_biasfield_file')

node.inputs.in_file = os.path.absbath(args.input)

workflow.run()
