import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl
import os

import argparse

name = 'test_RegF3d'

parser = argparse.ArgumentParser(description=name)

parser.add_argument('-r', '--reference',
                    dest='ref',
                    metavar='ref',
                    help='Reference Image',
                    required=True)
parser.add_argument('-f', '--floating',
                    dest='flo',
                    metavar='flo',
                    help='Floating Image',
                    required=True)

args = parser.parse_args()

workflow = pe.Workflow(name=name)
workflow.base_output_dir=name
workflow.base_dir=name

directory = os.getcwd()

node = pe.Node(interface = niftyreg.RegF3D(), name = 'regf3d')
output_node = pe.Node(interface = niu.IdentityInterface(fields = ['res_file', 'cpp_file', 'invcpp_file']), name = 'output_node')
workflow.connect(node, 'res_file', output_node, 'res_file')
workflow.connect(node, 'cpp_file', output_node, 'cpp_file')
workflow.connect(node, 'invcpp_file', output_node, 'invcpp_file')

node.inputs.ref_file = os.path.absbath(args.ref)
node.inputs.flo_file = os.path.absbath(args.flo)

workflow.run()

