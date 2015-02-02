#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import argparse
import os
from distutils import spawn

import nipype.interfaces.niftyreg as niftyreg

import niftk


def find_data_directory_function(in_db_file):
    def find_xml_data_path(in_file):
        import xml.etree.ElementTree as ET
        tree = ET.parse(in_file)
        root = tree.getroot()
        return root.findall('data')[0].findall('path')[0].text
    import os
    database_directory = os.path.dirname(in_db_file)
    data_directory = find_xml_data_path(in_db_file)
    print 'data_directory', data_directory
    ret = os.path.abspath(data_directory)
    if os.path.exists(ret):
        return ret
    ret = os.path.join(database_directory, data_directory)
    print 'ret', ret
    if os.path.exists(ret):
        return ret
    return None

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')

parser = argparse.ArgumentParser(description='GIF Propagation')

parser.add_argument('-i', '--input_file',
                    dest='input_file',
                    metavar='input_file',
                    help='Input image file to propagate labels in',
                    nargs='+',
                    required=True)

parser.add_argument('-d','--database',
                    dest='database',
                    metavar='database',
                    help='gif-based database xml file describing the inputs',
                    required=True)

parser.add_argument('-o','--output',
                    dest='output',
                    metavar='output',
                    help='output directory to which the gif outputs are stored',
                    required=True)

parser.add_argument('-n','--n_procs',
                    dest='n_procs',
                    metavar='n_procs',
                    help='maximum number of CPUs to be used when using the MultiProc plugin',
                    required=False,
                    default = 10)

parser.add_argument('-u','--username',
                    dest='username',
                    metavar='username',
                    help='Username to use to submit jobs on the cluster',
                    required=False)

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

basedir = os.getcwd()

# extracting basename of the input file (list)
input_files = [os.path.basename(f) for f in args.input_file]
# extracting the directory where the input file is (are)
input_directory = os.path.abspath(os.path.dirname(args.input_file[0]))
# extracting the 'subject list simply for iterable purposes
subject_list = [f.replace('.nii.gz','') for f in input_files]

# the dictionary to be used for iteration
info = dict(target=[['subject_id']])

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
datasource.inputs.field_template = dict(target='%s.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True



# The processing pipeline itself is instantiated
workflow_name='gif_propagation_workflow'
if len(subject_list) == 1:
    workflow_name=workflow_name + '_' + subject_list[0]

r = niftk.gif.create_niftyseg_gif_propagation_pipeline(name=workflow_name)

r.base_dir = basedir

# the input image is registered to the MNI for masking purpose
mni_to_input = pe.Node(interface=niftyreg.RegAladin(), 
                       name='mni_to_input')
mni_to_input.inputs.flo_file = mni_template
mask_resample  = pe.Node(interface = niftyreg.RegResample(), 
                         name = 'mask_resample')
mask_resample.inputs.inter_val = 'NN'
mask_resample.inputs.flo_file = mni_template_mask
    
# The directory where the other images are is found in the database xml file
find_data_directory = pe.Node(niu.Function(
    input_names = ['in_db_file'],
    output_names = ['directory'],
    function = find_data_directory_function),
                              name = 'find_data_directory')
find_data_directory.inputs.in_db_file = os.path.abspath(args.database)

# a data grabber to get template images
templatessource = pe.Node(interface=nio.DataGrabber(outfields=['images']),
                          name = 'templatessource')
templatessource.inputs.template = '*.nii*'
templatessource.inputs.sort_filelist = True

# Connect all the nodes together
r.connect(infosource, 'subject_id', datasource, 'subject_id')
r.connect(datasource, 'target', mni_to_input, 'ref_file')
r.connect(datasource, 'target', mask_resample, 'ref_file')
r.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
r.connect(find_data_directory, 'directory', templatessource, 'base_directory')
r.connect(datasource, 'target', r.get_node('input_node'), 'in_file')
r.connect(templatessource, 'images', r.get_node('input_node'), 'in_templates')
r.connect(mask_resample, 'res_file', r.get_node('input_node'), 'in_mask_file')
r.inputs.input_node.in_db_file = os.path.abspath(args.database)
r.inputs.input_node.out_dir = result_dir

# Run the overall workflow
dot_exec=spawn.find_executable('dot')   
if not dot_exec == None:
    r.write_graph(graph2use='colored')

qsub_exec=spawn.find_executable('qsub')

# Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
try:
    qsubargs=os.environ['QSUB_OPTIONS']
except KeyError:
    qsubargs='-l h_rt=05:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

# We can use qsub or not depending on this environment variable, by default we use it.
try:
    run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
except KeyError:
    run_qsub=True

if not qsub_exec == None and run_qsub:
    if args.username:
        r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs, 'username' : args.username})
    else:
        r.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
else:
    r.run(plugin='MultiProc', plugin_args={'n_procs' : args.n_procs})
    
    
