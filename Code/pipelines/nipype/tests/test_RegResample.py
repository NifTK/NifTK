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

resampler = pe.Node(interface = niftyreg.RegResample(), name = 'resampler')

output_node = pe.Node(interface = niu.IdentityInterface(fields = ['res_file', 'blank_file']), name = 'output_node')

workflow.connect(resampler, 'res_file', output_node, 'res_file')
workflow.connect(resampler, 'blank_file', output_node, 'blank_file')

resampler.inputs.ref_file = os.path.join(directory, '1000_3.nii.gz')
resampler.inputs.flo_file = os.path.join(directory, '1001_3.nii.gz')

workflow.run()

