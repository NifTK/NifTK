#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
from regional_average                   import create_reg_avg_value_pipeline
from nipype                             import config, logging
from distutils                          import spawn

import sys
import glob
import os
import textwrap
import argparse

def get_all_images_in_directory(path):
    list_of_images=[]
    list_of_images=list_of_images+glob.glob(path+'/*.nii')
    list_of_images=list_of_images+glob.glob(path+'/*.nii.gz')
    list_of_images=list_of_images+glob.glob(path+'/*.hdr')
    list_of_images.sort()
    return list_of_images;

class reg_avg_value_variables():
    def __init__(self):
        self.roi_choices=['cereb','gm_cereb','pons','none']
        self.parser=None
        self.reg_avg_value_create_parser()
    def reg_avg_value_create_parser(self):
        pipelineDescription=textwrap.dedent('''\
            Regional noramlised average value (SUVR) computation.
            Based on MRI scan (--mri) and its associated parcelation (--par)
            and tissus segmentation (--seg), a functional scan (--img) intensities are
            normalised (pons, grey matter of the cerebellum, full cerebellum or none)
            and average over multiple pre-defined region of interest. The user can
            also specify to perform the average computation without normalisation.
            This pipeline can be used for example to compute SUVR based on PET image
            or regional average instensity of ASL derived data.
        ''')
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
            description=pipelineDescription)
        """ Input images """
        self.parser.add_argument('--input_img_dir',dest='input_img_dir', type=str, \
            metavar='directory', help='Input directory containing functional images', \
            required=False)
        self.parser.add_argument('--input_mri_dir',dest='input_mri_dir', type=str, \
            metavar='directory', help='Input directory containing MRI images', \
            required=False)
        self.parser.add_argument('--input_seg_dir',dest='input_seg_dir', type=str, \
            metavar='directory', help='Input directory containing segmentation images', \
            required=False)
        self.parser.add_argument('--input_par_dir',dest='input_par_dir', type=str, \
            metavar='directory', help='Input directory containing parcelation images', \
            required=False)
        self.parser.add_argument('--img',dest='input_img', type=str, nargs='+', \
            metavar='image', help='Function image or list of function images', \
            required=False)
        self.parser.add_argument('--mri',dest='input_mri', type=str, nargs='+', \
            metavar='image', help='MRI image or list of MRI images', \
            required=False)
        self.parser.add_argument( '--seg',dest='input_seg', type=str, nargs='+', \
            metavar='image', help='Segmentation image or list of segmentation images', \
            required=False)
        self.parser.add_argument('--par',dest='input_par', type=str, nargs='+', \
            metavar='image', help='Parcelation image or list of parcelation images', \
            required=False)
        """ Other input """
        self.parser.add_argument('--input_trans_dir',dest='input_trans_dir', type=str, \
            metavar='directory', help='Input directory containing input transformations '+ \
            'parametrising a warping where the functional image is used as a floating'+ \
            'image and the MRI image as a reference image. The registration is performed'+ \
            'in the pipeline if no transformations are specified.', \
            required=False)
        self.parser.add_argument('--trans',dest='input_trans', type=str, nargs='+', \
            metavar='file', help='Input transformations or list of transformations'+ \
            'parametrising a warping where the functional image is used as a floating'+ \
            'image and the MRI image as a reference image. The registration is performed'+ \
            'in the pipeline if no transformations are specified.', \
            required=False)
        """ Output argument """
        self.parser.add_argument('--output_dir',dest='output_dir', type=str, \
            metavar='directory', help='Output directory containing the pipeline result', \
            default='.', required=False)
        self.parser.add_argument('--output_pre',dest='output_pre', type=str, \
            metavar='prefix', help='Output result prefix', \
            default='', required=False)
        self.parser.add_argument('--output_suf',dest='output_suf', type=str, \
            metavar='suffix', help='Output result suffix', \
            default='', required=False)
        """ Processing options"""
        self.parser.add_argument('--use_aff',dest='affine_flag', action='store_const', \
            default=False, const=True, help='use an affine registration '+ \
            'between the function image and MRI. Rigid is used by default. This'+ \
            'option is only used when no input transformations are specified', \
            required=False)        
        self.parser.add_argument('--roi', metavar='roi', nargs=1, type=str, \
            choices=self.roi_choices, default=[self.roi_choices[0]], \
            help='ROI to use to perform the function image intensities normalisation. ' + \
            'Choices are: '+str(self.roi_choices)+' without quotes.' + \
            'The default value is ' + str(self.roi_choices[0]));
        self.parser.add_argument('--thr', metavar='float', type=float, \
            nargs=1, default=[0.95], \
            help='threshold to use to define the ROI from a '+ \
            'probabilitic segmentation for the cerebellum grey matter.' + \
            'The default value is set to 0.95')
        self.parser.add_argument('--smoo', metavar='value', type=float, \
            default=0, help='FWHM of the Gaussian kernel to apply to the '+ \
            'input functional images. No Smoothing is applied by default.')
    def reg_avg_value_parse_arguments(self):
        args=self.parser.parse_args()
        # Perform some checks
        if args.input_img==None and args.input_img_dir==None:
            print('No input functional images have been specified. Exit.')
            sys.exit(1)
        if args.input_mri==None and args.input_img_dir==None:
            print('No input MRI images have been specified. Exit.')
            sys.exit(1)
        if args.input_seg==None and args.input_img_seg==None:
            print('No input segmentation images have been specified. Exit.')
            sys.exit(1)
        if args.input_par==None and args.input_img_par==None:
            print('No input parcelation images have been specified. Exit.')
            sys.exit(1)
        self.input_img=[]
        self.input_mri=[]
        self.input_seg=[]
        self.input_par=[]
        self.input_trans=[]
        if args.input_img:
            self.input_img=args.input_img
        elif args.input_img_dir:
            self.input_img=get_all_images_in_directory(args.input_img_dir)
        if args.input_mri:
            self.input_mri=args.input_mri
        elif args.input_mri_dir:
            self.input_mri=get_all_images_in_directory(args.input_mri_dir)
        if args.input_seg:
            self.input_seg=args.input_seg
        elif args.input_seg_dir:
            self.input_seg=get_all_images_in_directory(args.input_seg_dir)
        if args.input_par:
            self.input_par=args.input_par
        elif args.input_par_dir:
            self.input_par=get_all_images_in_directory(args.input_par_dir)
        if args.input_trans:
            self.input_trans=args.input_trans
        elif args.input_trans_dir:
            self.input_trans=glob.glob(args.input_trans_dir).sort()
        self.output_folder=args.output_dir
        self.output_prefix=args.output_pre
        self.output_suffix=args.output_suf
        self.func_mri_use_affine=args.affine_flag        
        self.roi=args.roi[0]
        self.seg_threshold=args.thr[0]
        self.fwhm=args.smoo
        if len(self.input_img)==0:
            print('No input image has been specified')
            sys.exit(1)        
        if not len(self.input_img)==len(self.input_mri):
            print('The number of specified functional and MRI images are expected ' + \
            'to be identical ('+str(len(self.input_img))+' vs '+ \
            str(len(self.input_mri))+')')
            sys.exit(1)
        if not len(self.input_mri)==len(self.input_par):
            print('The number of specified MRI and associated parcelation are expected ' + \
            'to be identical ('+str(len(self.input_mri))+' vs '+ \
            str(len(self.input_par))+')')
            sys.exit(1)
        if not len(self.input_mri)==len(self.input_seg):
            print('The number of segmentation is expected ' + \
            'to be identical to the number of functional/MRI images (' + \
            str(len(self.input_mri))+' vs '+str(len(self.input_seg))+')')
            sys.exit(1)
        if not len(self.input_img)==len(self.input_trans) and len(self.input_trans) > 0:
            print('The number of transformation is expected ' + \
            'to be identical to the number of functional/MRI images (' + \
            str(len(self.input_mri))+' vs '+str(len(self.input_trans))+')')
            sys.exit(1)


def gen_substitutions(functional_files, mri_files, roi, prefix, suffix):    
    from nipype.utils.filemanip import split_filename
    subs = []
    for i in range(0,len(functional_files)):
        functional_file=functional_files[i]
        mri_file=mri_files[i]
        _, func_bn, _ = split_filename(functional_file)
        _, mri_bn, _ = split_filename(mri_file)
        subs.append(('norm_'+roi+'_'+func_bn, \
                     prefix+'norm_'+roi+'_'+func_bn+suffix))
        subs.append(('suvr_'+roi+'_'+func_bn, \
                     prefix+'suvr_'+roi+'_'+func_bn+suffix))
        subs.append((mri_bn+'_aff', \
                     prefix+'ref_'+func_bn+'_flo_'+mri_bn+'_aff'+suffix))
    return subs
    
"""
Main
"""
def main():
    # Initialise the pipeline variables and the argument parsing
    reg_avg_value_var = reg_avg_value_variables()
    # Parse the input arguments
    reg_avg_value_var.reg_avg_value_parse_arguments()

    # Create the workflow
    workflow = pe.Workflow(name='reg_avg_value')
    workflow.base_dir = reg_avg_value_var.output_folder
    workflow.base_output_dir='reg_avg_value'

    # Create the output folder if it does not exists    
    if not os.path.exists(os.path.abspath(reg_avg_value_var.output_folder)):
        os.mkdir(os.path.abspath(reg_avg_value_var.output_folder))

    # Specify how and where to save the log files
    config.update_config({'logging': {'log_directory': os.path.abspath(reg_avg_value_var.output_folder),
                                      'log_to_file': True}})
    logging.update_logging(config)
    config.enable_debug_mode()
    
    # Create the input node interface
    reg_avg_value_pipeline=create_reg_avg_value_pipeline(reg_avg_value_var)
        
    # Create a node to add the suffix and prefix if required
    subsgen = pe.Node(interface = niu.Function(input_names = ['functional_files', \
        'mri_files', 'roi', 'prefix','suffix'], output_names = ['substitutions'], \
        function = gen_substitutions), name = 'subsgen')
    workflow.connect(reg_avg_value_pipeline, 'input_node.functional_files', subsgen, 'functional_files')
    workflow.connect(reg_avg_value_pipeline, 'input_node.mri_files', subsgen, 'mri_files')
    subsgen.inputs.roi=reg_avg_value_var.roi
    subsgen.inputs.prefix=reg_avg_value_var.output_prefix
    subsgen.inputs.suffix=reg_avg_value_var.output_suffix
    
    # Create a data sink    
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = os.path.abspath(os.path.abspath(reg_avg_value_var.output_folder))
    workflow.connect(subsgen, 'substitutions', ds, 'substitutions')
    workflow.connect(reg_avg_value_pipeline, 'output_node.out_files', ds, '@out')
    workflow.connect(reg_avg_value_pipeline, 'output_node.norm_files', ds, '@norm')
    workflow.connect(reg_avg_value_pipeline, 'output_node.aff_files', ds, '@aff')
    
    # Run the overall workflow
#     workflow.write_graph(graph2use='colored')
    qsub_exec=spawn.find_executable('qsub')

    # Can we provide the QSUB options using an environment variable QSUB_OPTIONS otherwise, we use the default options
    try:    
        qsubargs=os.environ['QSUB_OPTIONS']
    except KeyError:                
        print 'The environtment variable QSUB_OPTIONS is not set up, we cannot queue properly the process. Using the default script options.'
      	qsubargs='-l h_rt=01:00:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=1.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
        print qsubargs

    # We can use qsub or not depending of this environment variable, by default we use it.
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
