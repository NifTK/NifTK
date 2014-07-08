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

regf3d = pe.Node(interface = niftyreg.RegF3D(), name = 'regf3d')
regf3d.inputs.vel_flag = True

output_node = pe.Node(interface = niu.IdentityInterface(fields = ['res_file', 'cpp_file', 'invcpp_file']), name = 'output_node')

workflow.connect(regf3d, 'res_file', output_node, 'res_file')
workflow.connect(regf3d, 'cpp_file', output_node, 'cpp_file')
workflow.connect(regf3d, 'invcpp_file', output_node, 'invcpp_file')

regf3d.inputs.ref_file = os.path.join(directory, '1000_3.nii.gz')
regf3d.inputs.flo_file = os.path.join(directory, '1001_3.nii.gz')

workflow.run()

