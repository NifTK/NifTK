#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import dti_likelihood_study             as dmri
import argparse
import math
import os

base_dir_atlas = '/cluster/project0/atrophySimulation/temp_Ivor/atlas/chicago-template'
parc_file = base_dir_atlas+'-JHU-labels.nii.gz'
mask_file = base_dir_atlas+'-mask.nii.gz'
t1_file = base_dir_atlas+'-T1.nii.gz'
tensor_file = base_dir_atlas+'-tensors.nii.gz'
bval_file = base_dir_atlas+'.bval'
bvec_file = base_dir_atlas+'.bvec'
b0_file = base_dir_atlas+'-b0.nii.gz'

result_dir = '/cluster/project0/atrophySimulation/temp_Ivor/dti_likelihood_results/'
temp_dir = '/cluster/project0/atrophySimulation/temp_Ivor/dti_likelihood_temp/'
#os.mkdir(result_dir)

inter_types = ['LIN', 'CUB']
log_data_values = [True,False]
number_of_repeats = 1
for i in range(number_of_repeats):
    for log in log_data_values:
        for inter in inter_types:
            pipeline_name = 'dti_'
            if log == True:
                pipeline_name = pipeline_name+'log_'
            
            pipeline_name = pipeline_name + inter +'_'+str(i)
            r = dmri.create_dti_likelihood_study_workflow(name=pipeline_name, log_data = log, dwi_interp_type = inter, result_dir=result_dir)
            r.base_dir = temp_dir

            r.inputs.input_node.in_tensors_file = tensor_file
            r.inputs.input_node.in_bvec_file = bvec_file
            r.inputs.input_node.in_bval_file = bval_file
            r.inputs.input_node.in_b0_file = b0_file
            r.inputs.input_node.in_t1_file = t1_file
            r.inputs.input_node.in_mask_file = mask_file
            r.inputs.input_node.in_labels_file = parc_file
            
            r.inputs.input_node.in_stddev_translation = 0.75
            r.inputs.input_node.in_stddev_rotation = 0.5*math.pi/180
            r.inputs.input_node.in_stddev_shear = 0.04
            r.inputs.input_node.in_noise_sigma = 10.0

            #r.write_graph(graph2use = 'colored')

            qsubargs='-l h_rt=00:30:00 -l tmem=1.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

            #r.run()
            #r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
            qname = 'job_'+pipeline_name
            r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs, 'job_name' : str(qname) })
            #r.run(plugin='MultiProc')

