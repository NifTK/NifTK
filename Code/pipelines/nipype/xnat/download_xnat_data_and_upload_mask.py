#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import nipype.interfaces.dcm2nii        as mricron
import nipype.interfaces.fsl            as fsl
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
        if exp.label().find('MR') > -1:
            first_mr_experiment = exp
            break

    first_mprage_scan = None
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
    dg.inputs.query_template = '/projects/%s/subjects/%s/experiments/%s/scans/%s/resources/NIFTI'
    dg.inputs.query_template_args['struct'] = [['project','subject',first_mr_experiment.label(), first_mprage_scan.id()]]
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
#'/project/ADNI/subjects/0002/experiments/*/assessors/BET_MASK/resources/NIFTI

    bet = pe.MapNode(interface = fsl.BET(mask=True), name = 'bet', iterfield = ['in_file'])

    dsx = pe.Node(interface = nio.XNATSink(),
                 name = 'dsx')
    dsx.inputs.user = args.username
    dsx.inputs.pwd = args.password
    dsx.inputs.server = args.server
    dsx.inputs.project_id = args.project
    dsx.inputs.subject_id = subject
    dsx.inputs.experiment_id = first_mr_experiment.label()
    dsx.inputs.assessor_id = 'BET_MASK_2'

    assessor = first_mr_experiment.assessor('BET_MASK_2')
    if not assessor.exists():
        assessor.create()

    r.connect(dg, 'struct', dcm2nii, 'source_names')
    r.connect(dcm2nii, 'converted_files', bet, 'in_file')
    r.connect(bet, 'mask_file', dsx, '@xnat_data')

    r.run(plugin='Linear')

