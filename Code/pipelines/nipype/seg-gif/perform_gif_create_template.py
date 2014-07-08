#! /usr/bin/env python

import seg_gif_create_template_library as seggif
import glob, os

basedir = '/Users/nicolastoussaint/data/nipype/gif/'

T1s    = basedir + 'template-database/T1s/'
labels = basedir + 'template-database/labels/'
out    = basedir + 'output-database/'
T1s_files = glob.glob(T1s + '*.nii.gz')

ref = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
ref_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil1.nii.gz')

T1 = T1s + '1000_3.nii.gz'

r = seggif.create_seg_gif_create_template_database_workflow(name = 'gif_create_template', number_of_iterations = 3, ref_file = ref, ref_mask = ref_mask)

r.base_dir = basedir
r.inputs.input_node.in_entries_directory = T1s
r.inputs.input_node.in_initial_labels_directory = labels
r.inputs.input_node.out_database_directory = out

#r = seggif.prepare_inputs('gif-prepare-inputs')
#r.base_dir = basedir
#r.inputs.input_node.in_files = T1s_files

#r = seggif.seg_gif_preproc('gif-preproc')
#r.base_dir = basedir
#r.inputs.input_node.in_entries_directory = T1s
#r.inputs.input_node.out_T1_directory = out + 'T1s'
#r.inputs.input_node.out_cpps_directory = out + 'cpps'

r.write_graph(graph2use='hierarchical')
r.run('MultiProc')
exit
