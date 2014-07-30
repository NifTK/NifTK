# Unit test for the susceptibility tools, requires FSL to run BET
from nipype.testing import assert_equal
from add_noise import NoiseAdder
import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import os
    

parser = argparse.ArgumentParser(description='Noise adder test case')
parser.add_argument('-i','--input', help='Input image', required=True)
parser.add_argument('-m', '--mask', help='Mask image', required=False)
parser.add_argument('-n', '--noise_type', help='Noise type (gaussian/rician)', required=True)
parser.add_argument('-s', '--sigma', help='sigma of noise', required=True, type=float)
args = parser.parse_args()

pipeline = pe.Workflow('workflow')
pipeline.base_dir = os.getcwd()
input_node =  pe.Node(niu.IdentityInterface(
            fields=['in_file', 'mask_file', 'noise_type', 'sigma']),
                        name='input_node')
input_node.inputs.in_file = args.input
if args.mask:
    input_node.inputs.mask_file = args.mask
input_node.inputs.noise_type = args.noise_type 
input_node.inputs.sigma = args.sigma



noise_adder = pe.Node(interface=NoiseAdder(), name='noise_adder')


output_node = pe.Node(niu.IdentityInterface(
            fields=['out_file']),
                        name='output_node')
                        
pipeline.connect(input_node, 'in_file',noise_adder, 'in_file')
pipeline.connect(input_node, 'mask_file',noise_adder, 'mask_file')
pipeline.connect(input_node, 'noise_type',noise_adder, 'noise_type')
pipeline.connect(input_node, 'sigma',noise_adder, 'sigma_val')


pipeline.connect(noise_adder, 'out_file', output_node, 'out_file')

pipeline.write_graph(graph2use='exec')
pipeline.run()



