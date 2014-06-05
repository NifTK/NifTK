#! /usr/bin/env python
import nipype.pipeline.engine as pe          # pypeline engine
import registration as reg
import nipype.interfaces.io as nio
import nipype.interfaces.niftyreg as niftyreg
from nipype.interfaces.base import isdefined

output_dir = '/Users/isimpson/Software/nipype/test_output/'
ref_file = '/Users/isimpson/Software/nipype/test_data/scaled_pm.nii.gz'
second_round = False

dg = pe.Node(interface=nio.DataGrabber(outfields=['data']), name='dg')

dg.inputs.base_directory = '/Users/isimpson/Software/nipype/test_data/'
dg.inputs.sort_filelist = False
dg.inputs.template = 'vol*.nii.gz' 

# Add options here
linear_hash = {}

pipeline = pe.Workflow('workflow')
#r = reg.create_linear_coregistration_workflow('rigid_workflow', linear_options_hash = linear_hash)
# As we're passing an initial reference image, initial_ref=True
r = reg.create_atlas('atlas_creation', linear_options_hash = linear_hash, initial_ref=False, itr_rigid = 3, itr_affine = 0, itr_non_lin=0)
# If we don't have a reference image file defined, make one by averaging the input images
if not isdefined(ref_file):
    ave_ims = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims")
    pipeline.connect(dg, 'data', ave_ims, 'in_files')
    pipeline.connect(ave_ims, 'out_file', r, 'input_node.ref_file')
#else:
   # r.inputs.input_node.ref_file = ref_file

# Connect up the data inputs
pipeline.connect(dg, 'data', r, 'input_node.in_files')

ds = pe.Node(interface=nio.DataSink(parameterization=True), name='sink')

ds.inputs.base_directory = output_dir


# If we want a second round of affine registrations
if second_round == True:
    r2 = reg.create_linear_coregistration_workflow('affine_workflow')
    pipeline.connect(r, 'output_node.average_image', r2, 'input_node.ref_file')
    pipeline.connect(dg, 'data', r2, 'input_node.in_files')
    # Connect up the outputs to the sink
    pipeline.connect(r2, 'output_node.aff_files', ds, 'aff')
    pipeline.connect(r2, 'output_node.average_image', ds, 'ave_im')
else:
    pipeline.connect(r, 'output_node.aff_files', ds, 'aff')
    pipeline.connect(r, 'output_node.average_image', ds, 'ave_im')


pipeline.write_graph(graph2use='exec')
