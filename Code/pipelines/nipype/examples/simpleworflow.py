#! /usr/bin/env python
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl

better  = pe.Node(interface=dcm2nii.DCM2NII(), name='dcm2nii')
better.inputs.in_file = '/home/ntoussai/data/sge_tests/1001_3.nii.gz'
better.outputs.out_file = '/home/ntoussai/data/sge_tests/1001_3-bet.nii.gz'

better2 = pe.Node(interface=fsl.BET(), name='bet2')
#better2.inputs.in_file = '/home/ntoussai/data/sge_tests/1001_3-copy.nii.gz'
better2.outputs.out_file = '/home/ntoussai/data/sge_tests/1001_3-copy-bet.nii.gz'

workflow = pe.Workflow(name='preproc')
workflow.base_dir = '/home/ntoussai/data/sge_tests/'

workflow.add_nodes([better, better2])

workflow.connect(better, 'out_file', better2, 'in_file')

qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

workflow.run(plugin='SGEGraph',plugin_args=dict(qsub_args=qsubargs))

#workflow.run()
