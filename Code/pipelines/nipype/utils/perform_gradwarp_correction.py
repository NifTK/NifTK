#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
from nipype                             import config, logging
from distutils                          import spawn

import os
import textwrap
import argparse
import nipype.interfaces.niftyreg as niftyreg


def gen_substitutions(in_file, prefix, suffix):    
    from nipype.utils.filemanip import split_filename
    _, in_bn, _ = split_filename(in_file)
    subs = []    
    subs.append(('_res', ''))
    subs.append((in_bn, prefix+in_bn+'_unwarped'+suffix))

    return subs

"""
Main
"""
def main():
    # Create the parser
    pipelineDescription=textwrap.dedent('''\
    Pipeline to perform a gradwarp correction on an input image
    or a list of input images.
    ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
        description=pipelineDescription)
    """ Input images """
    parser.add_argument('-i', '--img',dest='input_img', \
                        type=str, \
                        metavar='input_img', \
                        help='Input image file to correct', \
                        required=True)
    """ Input coefficient file """
    parser.add_argument('-c', '--coeff',dest='input_coeff', 
                        type=str, metavar='coeff_file', 
                        help='File containing the spherical harmonic coefficient',
                        required=True)
    """ Interpolation order input """
    parser.add_argument('--inter', metavar='inter', type=str, \
            choices=['NN', 'LIN', 'CUB', 'SINC'], default='CUB', \
            help='Interpolation order to resample to unwarped image. '+ \
                'Choices are NN, LIN, CUB, SINC. [CUB]')
    """ Table offset values input """
    parser.add_argument('--offset_x', metavar='offset_x', type=float, \
            default=0, \
            help='Scanner table offset in x direction in mm. [0]', \
            required=False)
    parser.add_argument('--offset_y', metavar='offset_y', type=float, \
            default=0, \
            help='Scanner table offset in x direction in mm. [0]', \
            required=False)
    parser.add_argument('--offset_z', metavar='offset_z', type=float, \
            default=0, \
            help='Scanner table offset in x direction in mm. [0]', \
            required=False)
    """ Scanner type input input """
    parser.add_argument('--scanner', metavar='scanner', type=str, \
            choices=['ge','siemens'], default='siemens', \
            help='Scanner type. Choices are ge and siemens. [siemens]', \
            required=False)
    """ Gradwarp radius input """
    parser.add_argument('--radius', metavar='radius', type=float, \
            default=0.225, \
            help='Gradwarp radius in meter. [0.225]', \
            required=False)
    """ Output argument """
    parser.add_argument('--output_dir',dest='output_dir', 
                        type=str, \
                        metavar='directory', \
                        help='Output directory containing the unwarped result\n' + \
                        'Default is the current directory', \
                        default=os.path.abspath('.'), \
                        required=False)
    parser.add_argument('--output_pre',
                        dest='output_pre', \
                        type=str, \
                        metavar='prefix', \
                        help='Output result prefix', \
                        default='', \
                        required=False)
    parser.add_argument('--output_suf', \
                        dest='output_suf', \
                        type=str, \
                        metavar='suffix', \
                        help='Output result suffix', \
                        default='', \
                        required=False)
    
    	# Parse the arguments
    args=parser.parse_args()

    # Create a workflow to process the images
    workflow = pe.Workflow(name='gradwarp_correction')
    workflow.base_dir = args.output_dir
    workflow.base_output_dir='gradwarp_correction'
    
    # Specify how and where to save the log files
    config.update_config({'logging': {'log_directory': os.path.abspath(args.output_dir),
                                      'log_to_file': True}})
    logging.update_logging(config)
    config.enable_debug_mode()
    
    # Define the input and output node
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_file',
                    'input_coeff']),
            name='input_node')    
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['out_file']),
            name='output_node')
    input_node.inputs.in_file=args.input_img
    input_node.inputs.input_coeff=args.input_coeff
    
    # The gradwarp field is computed.
    gradwarp=pe.Node(interface = niftk.GradwarpCorrection(),
                     name='gradwarp')
    gradwarp.inputs.offset_x=-1*args.offset_x
    gradwarp.inputs.offset_y=-1*args.offset_y
    gradwarp.inputs.offset_z=-1*args.offset_z
    gradwarp.inputs.radius=args.radius
    gradwarp.inputs.scanner_type=args.scanner
    workflow.connect(input_node, 'in_file', gradwarp, 'in_file')
    workflow.connect(input_node, 'input_coeff', gradwarp, 'coeff_file')
    
    # The obtained deformation field is used the resample the input image
    resampling=pe.Node(interface = niftyreg.RegResample(),
                     name='resampling')
    resampling.inputs.inter_val=args.inter
    workflow.connect(input_node, 'in_file', resampling, 'ref_file')
    workflow.connect(input_node, 'in_file', resampling, 'flo_file')
    workflow.connect(gradwarp, 'out_file', resampling, 'trans_file')

    # Create a node to add the suffix and prefix if required
    subsgen = pe.Node(interface = niu.Function(input_names = ['in_file', \
        'prefix','suffix'], output_names = ['substitutions'], \
        function = gen_substitutions), name = 'subsgen')
    workflow.connect(input_node, 'in_file', subsgen, 'in_file')
    subsgen.inputs.prefix=args.output_pre
    subsgen.inputs.suffix=args.output_suf
    
    workflow.connect(resampling, 'res_file', output_node, 'out_file')
    
    # Create a data sink    
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = os.path.abspath(os.path.abspath(args.output_dir))
    workflow.connect(subsgen, 'substitutions', ds, 'substitutions')
    workflow.connect(output_node, 'out_file', ds, '@img')
    
    # Run the overall workflow    
    dot_exec=spawn.find_executable('dot')   
    if not dot_exec == None:
	workflow.write_graph(graph2use='colored')

    qsub_exec=spawn.find_executable('qsub')

    # Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
    try:    
        qsubargs=os.environ['QSUB_OPTIONS']
    except KeyError:                
        print 'The environtment variable QSUB_OPTIONS is not set up, we cannot queue properly the process. Using the default script options.'
      	qsubargs='-l h_rt=01:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
        print qsubargs

    # We can use qsub or not depending on this environment variable, by default we use it.
    try:    
        run_qsub=os.environ['RUN_QSUB'] in ['true', '1', 't', 'y', 'yes', 'TRUE', 'YES', 'T', 'Y']
    except KeyError:                
        run_qsub=True

    if not qsub_exec == None and run_qsub:
       workflow.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
    else:
        workflow.run(plugin='MultiProc')


if __name__ == "__main__":
    main()
