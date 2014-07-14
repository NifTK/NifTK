#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import dti_likelihood_study             as dmri
import argparse
import math
import os

parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-i', '--tensors',
                    dest='tensors',
                    metavar='tensors',
                    help='Tensor map used for likelihood simulations',
                    required=True)
parser.add_argument('-l','--bvals',
                    dest='bvals',
                    metavar='bvals',
                    help='bval file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-c','--bvecs',
                    dest='bvecs',
                    metavar='bvecs',
                    help='bvec file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-b','--b0',
                    dest='b0',
                    metavar='b0',
                    help='b0 file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-t','--t1',
                    dest='t1',
                    metavar='t1',
                    help='T1 file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-m','--mask',
                    dest='mask',
                    metavar='mask',
                    help='mask file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-p','--parcellation',
                    dest='parcellation',
                    metavar='parcellation',
                    help='parcellation file to be associated associated with the tensor map',
                    required=True)

args = parser.parse_args()

result_dir = os.getcwd()+'/results/'
#os.mkdir(result_dir)

inter_types = ['LIN', 'CUB']
log_data_values = [False]
number_of_repeats = 1
for i in range(number_of_repeats):
    for log in log_data_values:
        for inter in inter_types:
            pipeline_name = 'dti_likelihood_study_'
            if log == True:
                pipeline_name = pipeline_name+'log_'
            pipeline_name = pipeline_name + inter +'_'+str(i)
            r = dmri.create_dti_likelihood_study_workflow(name=pipeline_name, log_data = log, dwi_interp_type = inter, result_dir=result_dir)
            r.base_dir = os.getcwd()

            r.inputs.input_node.in_tensors_file = os.path.abspath(args.tensors)
            r.inputs.input_node.in_bvec_file = os.path.abspath(args.bvecs)
            r.inputs.input_node.in_bval_file = os.path.abspath(args.bvals)
            r.inputs.input_node.in_b0_file = os.path.abspath(args.b0)
            r.inputs.input_node.in_t1_file = os.path.abspath(args.t1)
            r.inputs.input_node.in_mask_file = os.path.abspath(args.mask)
            r.inputs.input_node.in_labels_file = os.path.abspath(args.parcellation)
            
            r.inputs.input_node.in_stddev_translation = 0.75
            r.inputs.input_node.in_stddev_rotation = 0.5*math.pi/180
            r.inputs.input_node.in_stddev_shear = 0.04
            # SNR of 15, based on the mean b0 in the JHU parcellation region
            r.inputs.input_node.in_noise_sigma = 53.0

            r.write_graph(graph2use = 'colored')

            qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

            #r.run()
            #r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
            #r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
            r.run(plugin='MultiProc', plugin_args={'n_procs' : 3})

