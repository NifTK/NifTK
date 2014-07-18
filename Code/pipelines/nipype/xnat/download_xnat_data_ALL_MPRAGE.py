#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import nipype.interfaces.dcm2nii        as mricron
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyseg       as niftyseg
import diffusion_mri_processing         as dmri
import argparse
import os, sys
import pyxnat


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
parser.add_argument('-s', '--subjects',
                    dest='subjects',
                    metavar='subjects',
                    help='xnat server subjects',
                    required=True,
                    nargs='+')
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='output',
                    default = 'results',
                    required=False)

args = parser.parse_args()

current_dir = os.getcwd()

result_dir = os.path.join(current_dir, args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

xnat = pyxnat.Interface(args.server, args.username, args.password, '/tmp/')

i = 0
for subject in args.subjects:

    experiments = xnat.select('/project/' + args.project + '/subjects/' + subject + '/experiments/*').get('obj')
    first_mr_experiment = None
    for exp in experiments:
        if exp.label().find('MR00') > -1:
            first_mr_experiment = exp
            break

    if first_mr_experiment != None:
        for scan in first_mr_experiment.scans():
            if scan.label().find('MPRAGE') > -1:
                first_mprage_scan = scan
                break
            if scan.attrs.get('type').find('MPRAGE') > -1:
                first_mprage_scan = scan
                break
        
    if first_mprage_scan == None:
        print 'No MP-RAGE found for subject ', subject, ' continuing...'
        continue

    i = i+1
    r = pe.Workflow(name='xnat_grabber_'+str(i))
    r.base_output_dir='xnat_grabber'
    r.base_dir = current_dir
    
    dg = pe.Node(interface = nio.XNATSource(infields=['project','subject'],
                                            outfields = ['struct']),
                 name = 'dg')
    dg.inputs.query_template = '/projects/%s/subjects/%s/experiments/%s/scans/%s/resources/DICOM'
    dg.inputs.query_template_args['struct'] = [['project','subject',first_mr_experiment.label(), first_mprage_scan.label()]]
    dg.inputs.user = args.username
    dg.inputs.pwd = args.password
    dg.inputs.server = args.server
    dg.inputs.project = args.project
    dg.inputs.subject = subject

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
    subs = []
    subs.append (('.*', result_dir + os.sep + subject + '.nii.gz'))
    ds.inputs.regexp_substitutions = subs
    
    r.connect(dg, 'struct', dcm2nii, 'source_names')
    r.connect(dcm2nii, 'converted_files', ds, '@xnat_data')
    r.run(plugin='Linear')

