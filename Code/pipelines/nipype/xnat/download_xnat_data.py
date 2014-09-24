#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import nipype.interfaces.dcm2nii        as mricron
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyseg       as niftyseg
import diffusion_mri_processing         as dmri
import argparse
import os

parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-i', '--server',
                    dest='server',
                    metavar='server',
                    help='XNAT server from where the data is taken',
                    required=True)
parser.add_argument('-u', '--username',
                    dest='username',
                    metavar='username',
                    help='xnat server username',
                    required=True)
parser.add_argument('-q', '--password',
                    dest='password',
                    metavar='password',
                    help='xnat server password',
                    required=True)
parser.add_argument('-p', '--project',
                    dest='project',
                    metavar='project',
                    help='xnat server project',
                    required=True)
parser.add_argument('-s', '--subject',
                    dest='subject',
                    metavar='subject',
                    help='xnat server subject',
                    required=True
                )
parser.add_argument('-e', '--experiment',
                    dest='experiment',
                    metavar='experiment',
                    help='xnat server experiment',
                    required=True
                )
parser.add_argument('-a', '--scan',
                    dest='scan',
                    metavar='scan',
                    help='xnat server scan',
                    required=True
                )
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='output',
                    default = 'results',
                    required=False)

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

r = pe.Workflow(name='xnat_grabber')
r.base_output_dir='xnat_grabber'
r.base_dir = os.getcwd()

dg = pe.Node(interface = nio.XNATSource(infields=['project','subject', 'experiment', 'scan'],
                                        outfields = ['output']),
             name = 'dg')
dg.inputs.query_template = '/projects/%s/subjects/%s/experiments/%s/scans/%s/resources/NIFTI'
dg.inputs.query_template_args['output'] = [['project','subject','experiment', 'scan']]
dg.inputs.user = args.username
dg.inputs.pwd = args.password
dg.inputs.server = args.server.strip('/')
dg.inputs.project = args.project
dg.inputs.subject = args.subject
dg.inputs.experiment = args.experiment
dg.inputs.scan = args.scan

dcm2nii = pe.Node(interface = mricron.Dcm2nii(), 
                  name = 'dcm2nii')
dcm2nii.inputs.args = '-d n'
dcm2nii.inputs.gzip_output = True
dcm2nii.inputs.anonymize = False
dcm2nii.inputs.reorient = True
dcm2nii.inputs.reorient_and_crop = False

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = result_dir
ds.inputs.parameterization = False

r.connect(dg, 'output', dcm2nii, 'source_names')
r.connect(dcm2nii, 'converted_files', ds, 'nifti')
r.connect(dg, 'output', ds, 'dicom')
r.run(plugin='MultiProc')

