#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
from nipype                             import config, logging
from distutils                          import spawn

from n4biascorrection import N4BiasCorrection
import sys
import os
import textwrap
import argparse
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')


def gen_substitutions(in_files, prefix, suffix):    
    from nipype.utils.filemanip import split_filename
    subs = []
    for i in range(0,len(in_files)):
        in_file=in_files[i]
        _, in_bn, _ = split_filename(in_file)
        subs.append((in_bn+'_corrected', \
            prefix+in_bn+'_corrected'+suffix))
        subs.append((in_bn+'_biasfield', \
            prefix+in_bn+'_biasfield'+suffix))
    return subs

"""
Main
"""
def main():
    # Create the parser
    pipelineDescription=textwrap.dedent('''\
    Pipeline to perform a bias field correction on an input image
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
    parser.add_argument('-m', '--mask',dest='input_mask', 
                        type=str, nargs='+', \
                        metavar='input_mask', 
                        help='Mask image or list of mask images (optional)', 
                        required=False)
    """ Output argument """
    parser.add_argument('--output_dir',dest='output_dir', 
                        type=str, \
                        metavar='directory', 
                        help='Output directory containing the registration result\n' + \
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

    	# Check the parsed arguments
    if not args.input_mask==None:
        if not len(args.input_img)==len(args.input_mask):
            print('The number of input and mask images are expected to be the same.')
            print(str(len(args.input_img))+' image(s) versus '+len(args.input_mask)+' mask(s). Exit.')
            sys.exit(1)
    
    # Create a workflow to process the images
    workflow = pe.Workflow(name='bias_correction')
    workflow.base_dir = args.output_dir
    workflow.base_output_dir='bias_correction'
    
    # Specify how and where to save the log files
    config.update_config({'logging': {'log_directory': os.path.abspath(args.output_dir),
                                      'log_to_file': True}})
    logging.update_logging(config)
    config.enable_debug_mode()
    
    # Define the input and output node
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['in_files',
                    'mask_files']),
            name='input_node')
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['out_img_files',
                    'out_bias_files']),
            name='output_node')
    input_node.inputs.in_files=args.input_img
    input_node.inputs.mask_files=args.input_mask
    
    # Fnding masks to use for bias correction:
    bias_correction=pe.MapNode(interface = N4BiasCorrection(),
                               name='bias_correction', 
                               iterfield=['in_file', 'mask_file'])
    bias_correction.inputs.in_downsampling=2

    if args.input_mask is None:
	mni_to_input = pe.MapNode(interface=niftyreg.RegAladin(), 
                                  name='mni_to_input',
                                  iterfield = ['ref_file'])
	mni_to_input.inputs.flo_file = mni_template
	mask_resample  = pe.MapNode(interface = niftyreg.RegResample(), 
                                    name = 'mask_resample',
                                    iterfield = ['ref_file', 'aff_file'])
	mask_resample.inputs.inter_val = 'NN'
	mask_resample.inputs.flo_file = mni_template_mask
	mask_eroder = pe.MapNode(interface = niftyseg.BinaryMaths(), 
                                 name = 'mask_eroder',
                                 iterfield = ['in_file'])
	mask_eroder.inputs.operation = 'ero'
	mask_eroder.inputs.operand_value = 3
	workflow.connect(input_node, 'in_files', mni_to_input, 'ref_file')
	workflow.connect(input_node, 'in_files', mask_resample, 'ref_file')
	workflow.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
	workflow.connect(mask_resample, 'res_file', mask_eroder, 'in_file')
        workflow.connect(mask_eroder, 'out_file', bias_correction, 'mask_file')
    else:
        workflow.connect(input_node, 'mask_files', bias_correction, 'mask_file')
    
    workflow.connect(input_node, 'in_files', bias_correction, 'in_file')
    	
    # Gather the processed images
    workflow.connect(bias_correction, 'out_file', output_node, 'out_img_files')
    workflow.connect(bias_correction, 'out_biasfield_file', output_node, 'out_bias_files')
    
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
    workflow.connect(output_node, 'out_img_files', ds, '@img')
    workflow.connect(output_node, 'out_bias_files', ds, '@field')
    
    # Run the overall workflow    
    # workflow.write_graph(graph2use='colored')
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
