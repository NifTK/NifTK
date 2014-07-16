#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import registration as reg
import argparse
import os
import nipype.interfaces.niftyreg as niftyreg

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')

parser = argparse.ArgumentParser(description='Groupwise registration')
parser.add_argument('-i', '--inputs',
                    dest='inputs',
                    metavar='inputs',
                    help='Input directory where images to be registered are stored',
                    required=True)
parser.add_argument('-o','--output',
                    dest='output',
                    metavar='output',
                    help='output directory to which the average and the are stored',
                    required=True)

parser.add_argument('-r','--rigiditerations',
                    dest='rigiditerations',
                    metavar='rigiditerations',
                    help='Number of rigid iterations',
                    required=False,
                    type=int, 
                    default=4)
parser.add_argument('-a','--affineiterations',
                    dest='affineiterations',
                    metavar='affineiterations',
                    help='Number of affine iterations',
                    required=False,
                    type=int, 
                    default=4)
parser.add_argument('-n','--nonlineariterations',
                    dest='nonlineariterations',
                    metavar='nonlineariterations',
                    help='Number of nonlinear iterations',
                    required=False,
                    type=int, 
                    default=4)

args = parser.parse_args()

basedir = os.getcwd()

dg = pe.Node(interface=nio.DataGrabber(outfields=['in_files']), 
             name='dg')
dg.inputs.base_directory = os.path.abspath(args.inputs)
dg.inputs.sort_filelist = False
dg.inputs.template = '*'
dg.inputs.field_template = dict(in_files = '*.nii*')

r = reg.create_atlas(name="atlas_creation", 
                         itr_rigid = args.rigiditerations,
                         itr_affine = args.affineiterations,
                         itr_non_lin = args.nonlineariterations,
                         initial_ref = True)
r.base_dir = basedir

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = os.path.abspath(args.output)
ds.inputs.parameterization = False

r.inputs.input_node.ref_file = mni_template
r.connect(dg, 'in_files', r.get_node('input_node'), 'in_files')
r.connect(r.get_node('output_node'), 'average_image', ds, '@average_image')
r.connect(r.get_node('output_node'), 'trans_files', ds, 'trans_files')

r.write_graph(graph2use = 'colored')

qsubargs='-l h_rt=00:59:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=1.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
#r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
r.run(plugin='MultiProc')

