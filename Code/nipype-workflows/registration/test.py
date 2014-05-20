#! /usr/bin/env python

import nipype.interfaces.utility as niu     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg
import registration as reg
from nipype.interfaces.base import traits
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import os

dg = pe.Node(interface=nio.DataGrabber(outfields=['data']), name='dg')

dg.inputs.base_directory = '/Users/isimpson/Software/nipype/test_data/'
dg.inputs.sort_filelist = False
dg.inputs.template = 'vol*.nii.gz' 

ds = pe.Node(interface=nio.DataSink(infields=['container'], parameterization=True), name='sink')
ds.inputs.base_directory = os.path.abspath('/Users/isimpson/Software/nipype/test_output/')
ds.inputs.container = "subject"

ref_file = '/Users/isimpson/Software/nipype/test_data/scaled_pm.nii.gz'
flo_file_list =  ['/Users/isimpson/Software/nipype/test_data/vol0000.nii.gz', '/Users/isimpson/Software/nipype/test_data/vol0000.nii.gz']
avg_im = 'average_im.nii.gz'

'''lin_reg = pe.MapNode(interface=niftyreg.RegAladin(), name="lin_reg", iterfield=['flo_file'])
lin_reg.iterables = ('flo_file',flo_file_list)
lin_reg.inputs.ref_file = ref_file

# Average the images, this needs to be a join node as the lin-reg node is iterable
#ave_ims = pe.JoinNode(interface=niftyreg.RegAverage(), name="ave_ims", joinfield='demean_files', joinsource='lin_reg')
#ave_ims.inputs.demean1_ref_file = ref_file
#ave_ims.inputs.out_file = avg_im

output_node = pe.JoinNode(
        niu.IdentityInterface(
            fields=['aff_files']),
                        name='outputnode', joinfield='aff_files', joinsource='lin_reg')

pipeline = pe.Workflow(name='reg_ave_test')

# Connect the output from reg_aladin that's suitable for reg_average into demean_files
#pipeline.connect([(lin_reg, ave_ims,[('avg_output','demean_files')]),
pipeline.connect([                (lin_reg, output_node,[('aff_file', 'aff_files')])
    ])

'''


pipeline = pe.Workflow('workflow')
r = reg.create_linear_coregistration_workflow('rigid_workflow')
#r.inputs.inputnode.in_files = traits.List(traits.File,dg.run().outputs)
r.inputs.inputnode.ref_file = ref_file
pipeline.connect([(dg, r, [('data','inputnode.in_files')])])
#pipeline.connect([(dg, ds, [('data','files')])])
#pipeline.connect([(r, ds, [('inputnode.in_files','container')])])
#pipeline.connect([(r, ds, [('lin_reg.aff_file','aff')])])
#pipeline.connect([(r, ds, [('ave_ims.out_file','ave')])])

pipeline.write_graph(graph2use='exec')
