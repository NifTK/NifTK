#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import dti_likelihood_study             as dmri
import argparse
import math
import os

parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-i','--tensors',
                    help='Tensor map used for likelihood simulations',
                    required=True)
parser.add_argument('-l','--bval',
                    help='bval file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-c','--bvec',
                    help='bvec file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-b','--b0',
                    help='b0 file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-t','--t1',
                    help='T1 file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-m','--mask',
                    help='mask file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-p','--parcellation',
                    help='parcellation file to be associated associated with the tensor map',
                    required=True)

args = parser.parse_args()

r = dmri.create_dti_reproducibility_study_workflow(name='dti_likelihood_study')
r.base_dir = os.getcwd()

r.inputs.input_node.in_tensors_file = os.path.abspath(args.tensors)
r.inputs.input_node.in_bvec_file = os.path.abspath(args.bvec)
r.inputs.input_node.in_bval_file = os.path.abspath(args.bval)
r.inputs.input_node.in_b0_file = os.path.abspath(args.b0)
r.inputs.input_node.in_t1_file = os.path.abspath(args.t1)
r.inputs.input_node.in_mask_file = os.path.abspath(args.mask)
r.inputs.input_node.in_labels_file = os.path.abspath(args.parcellation)

r.inputs.input_node.in_stddev_translation = 0.75
r.inputs.input_node.in_stddev_rotation = 0.5*math.pi/180
r.inputs.input_node.in_stddev_shear = 0.04
r.inputs.input_node.in_noise_sigma = 10.0
r.inputs.input_node.in_interpolation_scheme = 'Spline'
r.inputs.input_node.in_log_space = False

r.write_graph(graph2use = 'colored')

qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

#r.run()
#r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
#r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
r.run(plugin='MultiProc')

