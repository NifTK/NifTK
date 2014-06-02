#! /usr/bin/env python

use_simple = False

import seg_gif_propagation as gif

basedir = '/Users/nicolastoussaint/data/nipype/gif/'

infile = basedir + '072_S_4391_ADNI2_M_74.61_3.0T_HC_277672.nii.gz'
T1s    = basedir + 'template-database/T1s/'
db     = basedir + 'template-database/db.xml'
avg    = basedir + 'template-database/average.nii.gz'
outdir = basedir + 'output-database/'
cppdir = basedir + 'output-database/cpps'


if use_simple:

    r = gif.create_niftyseg_gif_propagation_pipeline_simple('gif-workflow')
    r.base_dir = basedir
    r.inputs.input_node.in_file = infile
    r.inputs.input_node.template_db_file = db
    r.inputs.input_node.out_res_directory = outdir
    r.inputs.input_node.out_cpp_directory = cppdir
    r.write_graph(graph2use='orig')
    r.run('Linear')
    exit

else:

    r = gif.create_niftyseg_gif_propagation_pipeline('gif-workflow')
    r.base_dir = basedir
    
    r.inputs.input_node.in_file = infile
    r.inputs.input_node.template_T1s_directory = T1s
    r.inputs.input_node.template_db_file = db
    r.inputs.input_node.template_average_image = avg
    r.inputs.input_node.out_res_directory = outdir
    r.inputs.input_node.out_cpp_directory = cppdir
    
    r.write_graph(graph2use='orig')
    
    r.run('Linear')

