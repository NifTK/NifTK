#! /usr/bin/env python
import nipype.pipeline.engine as pe          # pypeline engine
import registration as reg
import nipype.interfaces.io as nio

dg = pe.Node(interface=nio.DataGrabber(outfields=['data']), name='dg')

dg.inputs.base_directory = '/Users/isimpson/Software/nipype/test_data/'
dg.inputs.sort_filelist = False
dg.inputs.template = 'vol*.nii.gz' 



ref_file = '/Users/isimpson/Software/nipype/test_data/scaled_pm.nii.gz'
pipeline = pe.Workflow('workflow')
r = reg.create_linear_coregistration_workflow('rigid_workflow', rig_only = True)
r.inputs.input_node.ref_file = ref_file
# Connect up the data inputs'
pipeline.connect(dg, 'data', r, 'input_node.in_files')

r2 = reg.create_linear_coregistration_workflow('affine_workflow', rig_only= False)


# Now connect an affine registration workflow
pipeline.connect(r, 'output_node.average_image', r2, 'input_node.ref_file')
pipeline.connect(dg, 'data', r2, 'input_node.in_files')

pipeline.write_graph(graph2use='exec')

''' Need to investigate how to do data sink
ds = pe.Node(interface=nio.DataSink(infields=['container'], parameterization=True), name='sink')
ds.inputs.base_directory = os.path.abspath('/Users/isimpson/Software/nipype/test_output/')
ds.inputs.container = "subject"
pipeline.connect([(dg, ds, [('data','files')])])
pipeline.connect([(r, ds, [('inputnode.in_files','container')])])
pipeline.connect([(r, ds, [('lin_reg.aff_file','aff')])])
pipeline.connect([(r, ds, [('ave_ims.out_file','ave')])]) '''


