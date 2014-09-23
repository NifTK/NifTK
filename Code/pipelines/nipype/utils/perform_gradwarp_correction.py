#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
from nipype                             import config, logging
from distutils                          import spawn

#from gradwarp_correction import GradwarpCorrection
import os
import textwrap
import argparse
#import nipype.interfaces.niftyreg as niftyreg


def gen_substitutions(in_files, prefix, suffix):    
    from nipype.utils.filemanip import split_filename
    subs = []
    for i in range(0,len(in_files)):
        in_file=in_files[i]
        _, in_bn, _ = split_filename(in_file)
        subs.append((in_bn, \
            prefix+in_bn+'_unwarped'+suffix))

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
    parser.add_argument('-i', '--img',dest='input_img', 
                        type=str, nargs='+', \
                        metavar='input_img', 
                        help='Image file or list of input images',
                        required=True) 
    """ Input coefficient file """
    parser.add_argument('-c', '--coeff',dest='input_coeff', 
                        type=str, metavar='coeff_file', 
                        help='File containing the spherical harmonic coefficient',
                        required=True)
    """ Interpolation order input """
    parser.add_argument('--inter', metavar='order', nargs=1, type=str, \
            choices=[0, 1, 3, 4], default=3, \
            help='Interpolation order to resample to unwarped image. Default is 3 (cubic)')
    """ Output argument """
    parser.add_argument('--output_dir',dest='output_dir', 
                        type=str, \
                        metavar='directory', 
                        help='Output directory containing the unwarped result\n' + \
                        'Default is the current directory', \
                        default=os.path.abspath('.'), 
                        required=False)
    parser.add_argument('--output_pre',
                        dest='output_pre', 
                        type=str, \
                        metavar='prefix', 
                        help='Output result prefix', \
                        default='', 
                        required=False)
    parser.add_argument('--output_suf',
                        dest='output_suf',
                        type=str, \
                        metavar='suffix', 
                        help='Output result suffix', \
                        default='', 
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
            fields=['in_files',
                    'input_coeff']),
            name='input_node')
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['out_files']),
            name='output_node')
    input_node.inputs.in_files=args.input_img
    input_node.inputs.input_coeff=args.input_coeff
    
    # The gradwarp still does not work for the PET MRI yet.
    # This pipeline is thus empty at the moment
    workflow.connect(input_node, 'in_files', output_node, 'out_files')

    # Create a node to add the suffix and prefix if required
    subsgen = pe.Node(interface = niu.Function(input_names = ['in_files', \
        'prefix','suffix'], output_names = ['substitutions'], \
        function = gen_substitutions), name = 'subsgen')
    workflow.connect(input_node, 'in_files', subsgen, 'in_files')
    subsgen.inputs.prefix=args.output_pre
    subsgen.inputs.suffix=args.output_suf
    
    # Create a data sink    
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = os.path.abspath(os.path.abspath(args.output_dir))
    workflow.connect(subsgen, 'substitutions', ds, 'substitutions')
    workflow.connect(output_node, 'out_files', ds, '@img')
    
    # Run the overall workflow    
    # workflow.write_graph(graph2use='colored')
    qsub_exec=spawn.find_executable('qsub')
    if not qsub_exec == None:
        qsubargs='-l h_rt=01:00:00 -l tmem=2.8G -l h_vmem=2.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
        workflow.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
    else:
        workflow.run(plugin='MultiProc')


if __name__ == "__main__":
    main()
