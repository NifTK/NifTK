import nipype.interfaces.utility        as niu     # utility
import nipype.pipeline.engine           as pe          # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.fsl            as fsl
import os

import n4biascorrection

name = 'test'
workflow = pe.Workflow(name=name)
workflow.base_output_dir=name
workflow.base_dir=name

directory = os.getcwd()

biascorrection = pe.Node(interface = n4biascorrection.N4BiasCorrection(), name = 'biascorrection')

output_node = pe.Node(interface = niu.IdentityInterface(fields = ['out_file', 'out_biasfield_file']), name = 'output_node')

workflow.connect(biascorrection, 'out_file',  output_node, 'out_file')
workflow.connect(biascorrection, 'out_biasfield_file',   output_node, 'out_biasfield_file')

biascorrection.inputs.in_file = os.path.join(directory, '1000_3.nii.gz')
biascorrection.inputs.
workflow.run()

