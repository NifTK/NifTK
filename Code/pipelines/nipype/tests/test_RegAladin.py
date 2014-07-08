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

aladin = pe.Node(interface = niftyreg.RegAladin(), name = 'aladin')

output_node = pe.Node(interface = niu.IdentityInterface(fields = ['res_file', 'aff_file']), name = 'output_node')

workflow.connect(aladin, 'aff_file', output_node, 'aff_file')

aladin.inputs.ref_file = os.path.join(directory, '1000_3.nii.gz')
aladin.inputs.flo_file = os.path.join(directory, '1001_3.nii.gz')

workflow.run()

