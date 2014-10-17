#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import argparse
import os, sys
import getpass
import pyxnat
import nipype.interfaces.dcm2nii        as mricron

def get_sink_container_function(directory, project, subject, experiment, scan):
    import os    
    return os.path.join(directory,
                        project,
                        subject,
                        experiment,
                        scan)
    
description = ' XNAT Downloader:: ' + \
              'Downloads all scans that match a specific regular expression provided, browsing all subjects and experiments' + \
              ' The script will store the data in the provided result directory.' + \
              ' Make sure all the XNAT resources exist.' \
              ' CAUTION: if you desire the DICOM to be downloaded, please' + \
              ' add the -d option. ' + \
              ' Provide a config file for convenience. it is a simple text file with: ' + \
              '{"server":"https://myserver","user":"myusername","password":"mypassword","cachedir":"/tmp/"}' + \
              '.'


parser = argparse.ArgumentParser(description=description)

parser.add_argument('-i', '--server',
                    dest='server',
                    metavar='server',
                    help='XNAT server from where the data is taken')
parser.add_argument('-u', '--username',
                    dest='username',
                    metavar='username',
                    help='xnat server username')
parser.add_argument('-c', '--config',
                    dest='config',
                    metavar='config',
                    help='xnat configuration file: \n{"server":"https://myserver","user":"myusername","password":"mypassword","cachedir":"/tmp/"}')

parser.add_argument('-p', '--project',
                    dest='project',
                    metavar='project',
                    help='xnat server project',
                    required=True)
parser.add_argument('-r', '--regexp',
                    dest='regexp',
                    metavar='regexp',
                    help='regular expression to match the scan name / description',
                    required=True)

parser.add_argument('-d', '--dicom',
                    dest='dicom',
                    help='Download the DICOM (default is NIFTI ONLY) and convert to nifti',
                    required=False,
                    action='store_true')

parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='Output directory to store the data. The data is stored as project/subject/experiment/scan/[dicom/nifti]',
                    default = 'results',
                    required=False)

args = parser.parse_args()              

if args.config == None:
  if (args.server == None) or (args.username == None):
    print 'ERROR: Please provide either a config file or a server and username'
    sys.exit()
  pwd = getpass.getpass()
  server = pyxnat.Interface(server=args.server, user=args.username, password=pwd)

else:
  if (args.server != None) or (args.username != None):
    print 'ERROR: Please provide either a config file or a server and username'
    sys.exit()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


experiments_list = server.select('/project/'+args.project+'/subjects/*/experiments/*').get('obj')

print 'experiment list is of length ', len(experiments_list)

scans = []
projects = []
subjects = []
experiments = []
scans = []
regexp = args.scan

for experiment in experiments_list:
    for scan in experiment.scans().get('obj'):
        description = scan.attrs.get('type')
        if ( description.find(regexp) > -1 ):
            
            print 'subj: ', scan.parent().parent().label(), ' -- matching scan found: ', description, ' -- URI: ', scan.attrs.get('URI')
            scans.append(scan.label())
            experiments.append(scan.parent().label())
            subjects.append(scan.parent().parent().label())
            projects.append(args.project)
            


    
infosource = pe.Node(niu.IdentityInterface(fields = ['projects', 'subjects', 'experiments', 'scans']),
                     name = 'infosource', synchronize=True)
infosource.iterables = [('projects', projects),
                        ('subjects', subjects),
                        ('experiments', experiments),
                        ('scans', scans)]

r = pe.Workflow(name='xnat_downloader')
r.base_output_dir='xnat_grabber'
r.base_dir = os.getcwd()

dg = pe.Node(interface = nio.XNATSource(infields=['project','subject', 'experiment', 'scan'],
                                        outfields = ['output']),
             name = 'dg')

if args.dicom == True:
    resource = 'DICOM'
else:
    resource = 'NIFTI'

dg.inputs.query_template = '/projects/%s/subjects/%s/experiments/%s/scans/%s/resources/'+resource
dg.inputs.query_template_args['output'] = [['project','subject','experiment', 'scan']]

if args.config == None:
  dg.inputs.user = args.username
  dg.inputs.pwd = pwd
  dg.inputs.server = args.server.strip('/')
else:
  dg.inputs.config = os.path.abspath(args.config)

r.connect(infosource, 'projects', dg, 'project')
r.connect(infosource, 'subjects', dg, 'subject')
r.connect(infosource, 'experiments', dg, 'experiment')
r.connect(infosource, 'scans', dg, 'scan')

dg.inputs.project = projects
dg.inputs.subject = subjects
dg.inputs.experiment = experiments
dg.inputs.scan = scans

dcm2nii = pe.Node(interface = mricron.Dcm2nii(), 
                  name = 'dcm2nii')
dcm2nii.inputs.args = '-d n'
dcm2nii.inputs.gzip_output = True
dcm2nii.inputs.anonymize = True
dcm2nii.inputs.reorient = True
dcm2nii.inputs.reorient_and_crop = True

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.parameterization = False

get_sink_container = pe.Node(interface = niu.Function(input_names = ['directory', 'project', 'subject', 'experiment', 'scan'],
                                                       output_names = ['container'],
                                                       function = get_sink_container_function),
                              name = 'get_sink_container')
get_sink_container.inputs.directory = result_dir

r.connect(infosource, 'projects', get_sink_container, 'project')
r.connect(infosource, 'subjects', get_sink_container, 'subject')
r.connect(infosource, 'experiments', get_sink_container, 'experiment')
r.connect(infosource, 'scans', get_sink_container, 'scan')
r.connect(get_sink_container, 'container', ds, 'base_directory')

if args.dicom == True:
    r.connect(dg, 'output', dcm2nii, 'source_names')
    r.connect(dcm2nii, 'converted_files', ds, 'nifti')
    r.connect(dg, 'output', ds, 'dicom')
else:
    r.connect(dg, 'output', ds, 'nifti')

r.run(plugin='MultiProc')

