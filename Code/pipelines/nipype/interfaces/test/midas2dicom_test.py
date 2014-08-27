# Unit test for the susceptibility tools, requires FSL to run BET
from nipype.testing import assert_equal
from midas2dicom import Midas2Dicom
import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import os
database_paths = ['/var/lib/midas/data/fidelity/images/ims-study/',
									'/var/lib/midas/data/ppadti/images/ims-study/']

parser = argparse.ArgumentParser(description='MIDAS to DICOM test case')
parser.add_argument('-m','--midas_code', help='MIDAS code input', required=True)
args = parser.parse_args()

pipeline = pe.Workflow('workflow')
pipeline.base_dir = os.getcwd()
input_node =  pe.Node(niu.IdentityInterface(
            fields=['midas_code']),
                        name='input_node')
input_node.inputs.midas_code = args.midas_code

dicomfinder = pe.Node(interface=Midas2Dicom(), name='dicomfinder')
dicomfinder.inputs.midas_dirs = database_paths

output_node = pe.Node(niu.IdentityInterface(
            fields=['dicom_directory']),
                        name='output_node')
                        
pipeline.connect(input_node, 'midas_code',dicomfinder, 'midas_code')

pipeline.connect(dicomfinder, 'dicom_dir', output_node, 'dicom_directory')

pipeline.run()

print pipeline.get_node('output_node').outputs


