#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import argparse
import os
import glob
from distutils import spawn

import nipype.interfaces.niftyreg as niftyreg

import niftk

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil1.nii.gz')

parser = argparse.ArgumentParser(description='GIF Propagation')

parser.add_argument('-i', '--input_dir',
                    dest='input_dir',
                    metavar='input_dir',
                    help='Input directory where to find db.xml, T1s, masks, cpps, labels',
                    required=True)

parser.add_argument('-o','--output_dir',
                    dest='output_dir',
                    metavar='output_dir',
                    help='Output directory where to put the labels',
                    required=True)

args = parser.parse_args()

basedir = os.getcwd()

# extracting the directory where the input file is (are)
input_directory = os.path.abspath(args.input_dir)
# extracting basename of the input file (list)
input_files = [os.path.basename(f) for f in glob.glob(os.path.join(input_directory, 'T1s', '*.nii*'))]
# extracting the 'subject list' simply for iterable purposes
subject_list = [f.replace('.nii.gz','') for f in input_files]

print 'input directory: ', input_directory
print 'input files: ', input_files
print 'subject list: ', subject_list

# extracting the database file:
database_file = os.path.join(input_directory, 'db.xml')

# the dictionary to be used for iteration
info = dict(target=[['subject_id']],
            mask=[['subject_id']],
            cpp_dir=[['subject_id']])

# the iterable node
infosource = pe.Node(niu.IdentityInterface(fields = ['subject_id']),
                     name = 'infosource')
infosource.iterables = ('subject_id', subject_list)

# a data grabber to get the actual image file
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')
# the template is a simple string 'subject_id.nii.gz'
datasource.inputs.template = '%s'
datasource.inputs.base_directory = input_directory
# with these two lines the grabber should grab the file indicated by the iterable node
datasource.inputs.field_template = dict(target='T1s/%s.nii.gz',
                                        mask='masks/%s.nii.gz',
                                        cpp_dir='cpps/%s')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

# The processing pipeline itself is instantiated
r = niftk.gif.create_niftyseg_gif_propagation_pipeline_simple(name='gif_iteration')
r.base_dir = basedir

# Connect all the nodes together
r.connect(infosource, 'subject_id', datasource, 'subject_id')
r.connect(datasource, 'target', r.get_node('input_node'), 'in_file')
r.connect(datasource, 'mask', r.get_node('input_node'), 'in_mask_file')
r.connect(datasource, 'cpp_dir', r.get_node('input_node'), 'in_cpp_dir')
r.inputs.input_node.in_db_file = database_file
r.inputs.input_node.out_dir = os.path.abspath(args.output_dir)

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
    qsubargs='-l h_rt=05:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
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
    
    
