#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
import nipype.interfaces.niftyreg       as niftyreg     # NiftyReg
from nipype                             import config, logging

import sys
import os
import textwrap
import argparse

def gen_substitutions(ref_files, flo_files, prefix, suffix):    
    from nipype.utils.filemanip import split_filename
    subs = []
    for i in range(0,len(ref_files)):
        ref_file=ref_files[i]
        flo_file=flo_files[i]
        _, ref_bn, _ = split_filename(ref_file)
        _, flo_bn, _ = split_filename(flo_file)
        subs.append((flo_bn+'_aff', \
                     prefix+'ref_'+ref_bn+'_flo_'+flo_bn+'_aff'+suffix))
        subs.append((flo_bn+'_res', \
                     prefix+'ref_'+ref_bn+'_flo_'+flo_bn+'_res'+suffix))
        subs.append(('_reg_transform', ''))
    return subs
    
"""
Main
"""
def main():
    # Create the parser
    transformation_types=['rigid','aff_to_rig', 'affine']
    pipelineDescription=textwrap.dedent('''\
            Pipeline to perform a global registration given one (or several)
            reference image(s) and one (or several) floating image(s). By
            default, the registrationsare perform using a symmetric framework
            but the user can request an asymmetric approach. The available
            transformation model are rigid, affine and 12to6 (6to3 in 2D).
        ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
        description=pipelineDescription)
    """ Input images """
    parser.add_argument('-r','--ref',dest='input_ref', type=str, nargs='+', \
        metavar='image', help='Reference image or list of reference images', \
        required=True)
    parser.add_argument('-f','--flo',dest='input_flo', type=str, nargs='+', \
        metavar='image', help='Floating image or list of floating images', \
        required=True)
    parser.add_argument('--rmask',dest='input_rmask', type=str, nargs='+', \
        metavar='image', help='Reference mask image or list of reference mask images', \
        required=False, default='')
    parser.add_argument('--fmask',dest='input_fmask', type=str, nargs='+', \
        metavar='image', help='Floating mask image or list of floating mask images', \
        required=False, default='')
    """ Input argument """
    parser.add_argument('-t','--trans', dest='trans_type', nargs=1, type=str, \
        choices=transformation_types, default=[transformation_types[0]], \
        help='Type of transformation parametrisation to use. ' + \
        'Choices are: '+str(transformation_types)+' without quotes\n' + \
        'The default value is ' + str(transformation_types[0]));
    parser.add_argument('--no_sym',dest='asymmetric_reg', action='store_const', \
        default=False, const=True, help='The registration is symmetric by default, ' + \
        'use this flag to force asymmetric registration', \
        required=False)        
    """ Output argument """
    parser.add_argument('--output_dir',dest='output_dir', type=str, \
        metavar='directory', help='Output directory containing the registration result\n' + \
        'Default is the current directory', \
        default=os.path.abspath('.'), required=False)
    parser.add_argument('--output_pre',dest='output_pre', type=str, \
        metavar='prefix', help='Output result prefix', \
        default='', required=False)
    parser.add_argument('--output_suf',dest='output_suf', type=str, \
        metavar='suffix', help='Output result suffix', \
        default='', required=False)
        
    # Parse the arguments
    args=parser.parse_args()
    
    # Check that the number of reference and floating is equal
    if len(args.input_ref) != len(args.input_flo):
        print('The number of reference and floating images are expected to be the same.')
        print(str(len(args.input_ref))+' reference versus '+len(args.input_flo)+' floating. Exit.')
        sys.exit(1)
    # Check that the number of specified mask is equal to the number of image
    if len(args.input_rmask)>0 and len(args.input_ref) != len(args.input_rmask):
        print('The number of reference and reference mask images are expected to be the same.')
        print(str(len(args.input_ref))+' reference versus '+len(args.input_rmask)+' mask. Exit.')
        sys.exit(1)
    if len(args.input_fmask)>0 and len(args.input_flo) != len(args.input_fmask):
        print('The number of floating and floating mask images are expected to be the same.')
        print(str(len(args.input_flo))+' floating versus '+len(args.input_fmask)+' mask. Exit.')
        sys.exit(1)
        
    # Create a workflow to process the images
    workflow = pe.Workflow(name='global_reg')
    workflow.base_dir = args.output_dir
    workflow.base_output_dir='global_reg'
    
    # Specify how and where to save the log files
    config.update_config({'logging': {'log_directory': os.path.abspath(args.output_dir),
                                      'log_to_file': True}})
    logging.update_logging(config)
    config.enable_debug_mode()
    
    # Define the input and output node
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['ref_files',
                    'flo_files',
                    'rmask_files',
                    'fmask_files']),
            name='input_node')
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['res_files',
                    'aff_files']),
            name='output_node')
    input_node.inputs.ref_files=args.input_ref;
    input_node.inputs.flo_files=args.input_flo;
    input_node.inputs.rmask_files=args.input_rmask;
    input_node.inputs.fmask_files=args.input_fmask;

    
    # MapNode to perform the global registration
    if len(args.input_rmask)>0 and len(args.input_fmask):
        aladin=pe.MapNode(interface = niftyreg.RegAladin(), name='aladin', \
            iterfield=['ref_file', 'flo_file', 'rmask_file', 'fmask_file'])
        workflow.connect(input_node, 'rmask_files', aladin, 'rmask_file')
        workflow.connect(input_node, 'fmask_files', aladin, 'fmask_file')
    elif len(args.input_rmask)>0:
        aladin=pe.MapNode(interface = niftyreg.RegAladin(), name='aladin', \
            iterfield=['ref_file', 'flo_file', 'rmask_file'])
        workflow.connect(input_node, 'rmask_files', aladin, 'rmask_file')
    elif len(args.input_fmask)>0:
        aladin=pe.MapNode(interface = niftyreg.RegAladin(), name='aladin', \
            iterfield=['ref_file', 'flo_file', 'fmask_file'])
        workflow.connect(input_node, 'fmask_files', aladin, 'fmask_file')
    else:
        aladin=pe.MapNode(interface = niftyreg.RegAladin(), name='aladin', \
            iterfield=['ref_file', 'flo_file'])
    workflow.connect(input_node, 'ref_files', aladin, 'ref_file')
    workflow.connect(input_node, 'flo_files', aladin, 'flo_file')
    aladin.inputs.nosym_flag=args.asymmetric_reg
    if args.trans_type[0]=='rigid':
        aladin.inputs.rig_only_flag=True

    # Extract the rigid from the affine if required
    if args.trans_type[0]=='aff_to_rig':
        transform=pe.MapNode(interface = niftyreg.RegTransform(), name='transform', \
            iterfield=['aff_2_rig_input'])
        workflow.connect(aladin, 'aff_file', transform, 'aff_2_rig_input')
        resample=pe.MapNode(interface = niftyreg.RegResample(), name='resample', \
            iterfield=['ref_file','flo_file','trans_file'])
        workflow.connect(input_node, 'ref_files', resample, 'ref_file')
        workflow.connect(input_node, 'flo_files', resample, 'flo_file')
        workflow.connect(transform, 'out_file', resample, 'trans_file')
        workflow.connect(transform, 'out_file', output_node, 'aff_files')
        workflow.connect(resample, 'res_file', output_node, 'res_files')
    else:
        workflow.connect(aladin, 'aff_file', output_node, 'aff_files')
        workflow.connect(aladin, 'res_file', output_node, 'res_files')
        
            
    # Create a node to add the suffix and prefix if required
    subsgen = pe.Node(interface = niu.Function(input_names = ['ref_files', \
        'flo_files', 'prefix','suffix'], output_names = ['substitutions'], \
        function = gen_substitutions), name = 'subsgen')
    workflow.connect(input_node, 'ref_files', subsgen, 'ref_files')
    workflow.connect(input_node, 'flo_files', subsgen, 'flo_files')
    subsgen.inputs.prefix=args.output_pre
    subsgen.inputs.suffix=args.output_suf
    
    # Create a data sink    
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = os.path.abspath(os.path.abspath(args.output_dir))
    workflow.connect(subsgen, 'substitutions', ds, 'substitutions')
    workflow.connect(output_node, 'aff_files', ds, '@aff')
    workflow.connect(output_node, 'res_files', ds, '@res')

    # Run the overall workflow
#     workflow.write_graph(graph2use='colored')
    qsub_exec=spawn.find_executable('qsub')
	if not qsub_exec == None:
		qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
		workflow.run(plugin='SGE',plugin_args={'qsub_args': qsubargs})
	else:
		workflow.run(plugin='MultiProc')

    
if __name__ == "__main__":
    main()