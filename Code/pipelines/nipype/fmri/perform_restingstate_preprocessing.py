#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
from distutils                          import spawn

import restingstatefmri as fmri
import os
import argparse



def create_restingstatefmri_preprocessing_pipeline(name):


    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_fmri',
                    'in_t1',
                    'in_tissue_segmentation',
                    'in_parcellation']),
        name='input_node')

    resting_state_preproc = pe.Node(interface = fmri.RestingStatefMRIPreprocess(),
                                    name = 'resting_state_preproc')

    workflow.connect(input_node, 'in_fmri', resting_state_preproc, 'in_fmri')
    workflow.connect(input_node, 'in_t1', resting_state_preproc, 'in_t1')
    workflow.connect(input_node, 'in_tissue_segmentation', resting_state_preproc, 'in_tissue_segmentation')
    workflow.connect(input_node, 'in_parcellation', resting_state_preproc, 'in_parcellation')

    # Output node
    output_node = pe.Node( interface=niu.IdentityInterface(
        fields=['out_corrected_fmri', 
                'out_fmri_to_t1_transformation']),
                           name="output_node" )
    workflow.connect(resting_state_preproc, 'out_corrected_fmri', 
              output_node, 'out_corrected_fmri')
    workflow.connect(resting_state_preproc, 'out_fmri_to_t1_transformation', 
              output_node, 'out_fmri_to_t1_transformation')

    return workflow




parser = argparse.ArgumentParser(description='Resting State fMRI preprocessing')
parser.add_argument('-i', '--fmri',
                    dest='fmri',
                    metavar='fmri',
                    help='Input fMRI 4D image',
                    required=True)
parser.add_argument('-t', '--t1',
                    dest='t1',
                    metavar='t1',
                    help='Input T1 image',
                    required=True)
parser.add_argument('-s', '--segmentation',
                    dest='segmentation',
                    metavar='segmentation',
                    help='Input Tissue Segmentation image (from GIF pipeline)',
                    required=True)
parser.add_argument('-p', '--parcellation',
                    dest='parcellation',
                    metavar='parcellation',
                    help='Input Parcellation image (from GIF pipeline)',
                    required=True)
parser.add_argument('-o','--output_dir',
                    dest='output_dir',
                    metavar='output_dir',
                    help='output directory to which the average and the are stored',
                    default=os.path.abspath('results'), 
                    required=False)


args = parser.parse_args()

result_dir = os.path.abspath(args.output_dir)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

basedir = os.getcwd()

r = create_restingstatefmri_preprocessing_pipeline(name='restingstatefmri_preprocessing')
r.base_dir = basedir

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = os.path.abspath(result_dir)
ds.inputs.parameterization = False

r.inputs.input_node.in_t1 = os.path.abspath(args.t1)
r.inputs.input_node.in_fmri = os.path.abspath(args.fmri)
r.inputs.input_node.in_tissue_segmentation = os.path.abspath(args.segmentation)
r.inputs.input_node.in_parcellation = os.path.abspath(args.parcellation)

r.connect(r.get_node('output_node'), 'out_corrected_fmri', ds, '@corrected_fmri')
r.connect(r.get_node('output_node'), 'out_fmri_to_t1_transformation', ds, '@fmri_to_t1_transformation')


# Run the overall workflow

r.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')
if not qsub_exec == None:
	qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
	r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
	r.run(plugin='MultiProc')

