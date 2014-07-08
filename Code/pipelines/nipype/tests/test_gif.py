import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl
import os

name = 'test'
workflow = pe.Workflow(name=name)
workflow.base_output_dir=name
workflow.base_dir=name

directory = os.getcwd()

gif = pe.Node(interface = niftyseg.Gif(), name = 'gif')

output_node = pe.Node(interface = niu.IdentityInterface(fields = ['parc_file', 'geo_file', 'out_dir', 'prior_file']), name = 'output_node')

workflow.connect(gif, 'parc_file',  output_node, 'parc_file')
workflow.connect(gif, 'geo_file',   output_node, 'geo_file')
workflow.connect(gif, 'prior_file', output_node, 'prior_file')
workflow.connect(gif, 'out_dir',    output_node, 'out_dir')

gif.inputs.in_file = os.path.join(directory, '../output-database/T1s/1002_3.nii.gz')
gif.inputs.database_file = os.path.join(directory, '../output-database/db1/db.xml')
gif.inputs.cpp_dir = os.path.join(directory, '../output-database/cpps/1002_3/')
gif.inputs.out_dir = os.path.join(directory, '../output-database/')

workflow.run()

