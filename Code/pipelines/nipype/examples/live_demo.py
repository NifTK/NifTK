#! /usr/bin/env python

import os
import nipype.interfaces.utility  as niu            
import nipype.interfaces.io       as nio     
import nipype.pipeline.engine     as pe          

import nipype.interfaces.niftyreg as niftyreg

def create_workflow(name = 'simple_workflow'):
    input_node  = pe.Node    (interface=niu.IdentityInterface(fields=['ref_file', 'flo_file']),
                              name='input_node')
    aladin      = pe.MapNode (interface=niftyreg.RegAladin(), 
                              name='aladin',
                              iterfield=['flo_file'])
    resample    = pe.MapNode (interface=niftyreg.RegResample(), 
                              name='resample',
                              iterfield=['flo_file', 'aff_file'])
    output_node = pe.Node    (interface=niu.IdentityInterface(fields=['res_file', 'aff_file']),
                              name='output_node')
    w = pe.Workflow(name=name)
    w.base_output_dir=name
    w.connect(input_node, 'ref_file', aladin,      'ref_file')
    w.connect(input_node, 'flo_file', aladin,      'flo_file')
    w.connect(aladin,     'aff_file', resample,    'aff_file')
    w.connect(input_node, 'ref_file', resample,    'ref_file')
    w.connect(input_node, 'flo_file', resample,    'flo_file')
    w.connect(resample,   'res_file', output_node, 'res_file')
    w.connect(aladin,     'aff_file', output_node, 'aff_file')
    return w

dg = pe.Node(interface=nio.DataGrabber(outfields=['ref', 'flo']), 
             name='dg')
dg.inputs.base_directory = os.getcwd()
dg.inputs.sort_filelist = False
dg.inputs.template = '*'
dg.inputs.field_template = dict(ref = 'reference.nii.gz',
                                flo = 'floating_*.nii.gz')

w = create_workflow('myworkflow')
w.base_dir = os.getcwd()
w.connect(dg, 'ref', w.get_node('input_node'), 'ref_file')
w.connect(dg, 'flo', w.get_node('input_node'), 'flo_file')

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = os.getcwd()
ds.inputs.parameterization = False
w.connect(w.get_node('output_node'), 'res_file', ds, 'resampled')

w.write_graph()

w.run()
