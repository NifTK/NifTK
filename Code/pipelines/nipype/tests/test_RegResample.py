import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl
import os
import argparse

name = 'test_RegResample'

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

node = pe.Node(interface = niftyreg.RegResample(), name = 'resample')
output_node = pe.Node(interface = niu.IdentityInterface(fields = ['res_file', 'blank_file']), name = 'output_node')
workflow.connect(node, 'res_file', output_node, 'res_file')
workflow.connect(node, 'blank_file', output_node, 'res_file')

node.inputs.ref_file = os.path.absbath(args.ref)
node.inputs.flo_file = os.path.absbath(args.flo)

workflow.run()
