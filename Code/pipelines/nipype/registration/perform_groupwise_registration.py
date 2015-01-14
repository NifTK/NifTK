#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import argparse
import os
import nipype.interfaces.niftyreg as niftyreg
from distutils import spawn

import registration as reg

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


# Run the overall workflow
dot_exec=spawn.find_executable('dot')   
if not dot_exec == None:
    r.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')

# Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
try:    
    qsubargs=os.environ['QSUB_OPTIONS']
except KeyError:                
    print 'The environtment variable QSUB_OPTIONS is not set up, we cannot queue properly the process. Using the default script options.'
    qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    print qsubargs

# We can use qsub or not depending of this environment variable, by default we use it.
try:    
    run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
except KeyError:                
    run_qsub=True

if not qsub_exec == None and run_qsub:
	r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
	r.run(plugin='MultiProc')

