#! /usr/bin/env python

import nipype.interfaces.utility        as niu          # utility
import nipype.interfaces.io             as nio          # Input Output
import nipype.pipeline.engine           as pe           # pypeline engine
import nipype.interfaces.niftyseg       as niftyseg     # NiftySeg
import nipype.interfaces.niftyreg       as niftyreg     # NiftyReg
from nipype                             import config, logging
from extract_roi_statistics             import ExtractRoiStatistics
from normalise_roi_average_values       import NormaliseRoiAverageValues

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
        self.parser.add_argument('--trans',dest='input_trans', type=str, \
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
            choices=self.roi_choices, default=self.roi_choices[0], \
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

def create_mask_from_functional():
    workflow = pe.Workflow(name='mask_func')
    workflow.base_dir = os.getcwd()
    workflow.base_output_dir='mask_func'
    # Create all the required nodes
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['functional_files']),
            name='input_node')
    otsu_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='otsu_filter', iterfield=['in_file'])
    erosion_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='erosion_filter', iterfield=['in_file'])
    lconcomp_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='lconcomp_filter', iterfield=['in_file'])
    dilation_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='dilation_filter', iterfield=['in_file'])
    fill_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='fill_filter', iterfield=['in_file'])
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['mask_files']),
            name='output_node')
    # Define the node options
    otsu_filter.inputs.operation='otsu'
    erosion_filter.inputs.operation='ero'
    erosion_filter.inputs.operand_value=1
    lconcomp_filter.inputs.operation='lconcomp'
    dilation_filter.inputs.operation='dil'
    dilation_filter.inputs.operand_value=5
    fill_filter.inputs.operation='fill'
    fill_filter.inputs.output_datatype='char'
    # Create the connections
    workflow.connect(input_node, 'functional_files', otsu_filter, 'in_file')
    workflow.connect(otsu_filter, 'out_file', erosion_filter, 'in_file')
    workflow.connect(erosion_filter, 'out_file', lconcomp_filter, 'in_file')
    workflow.connect(lconcomp_filter, 'out_file', dilation_filter, 'in_file')
    workflow.connect(dilation_filter, 'out_file', fill_filter, 'in_file')
    workflow.connect(fill_filter, 'out_file', output_node, 'mask_files')    
    
    return workflow
    
def create_mask_from_parcelation():
    workflow = pe.Workflow(name='mask_mri')
    workflow.base_dir = os.getcwd()
    workflow.base_output_dir='mask_mri'
    # Create all the required nodes
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['par_files']),
            name='input_node')
    binary_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='binary_filter', iterfield=['in_file'])
    erosion_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='erosion_filter', iterfield=['in_file'])
    lconcomp_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='lconcomp_filter', iterfield=['in_file'])
    dilation_filter = pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='dilation_filter', iterfield=['in_file'])
    fill_filter = pe.MapNode(interface = niftyseg.UnaryMaths(), \
        name='fill_filter', iterfield=['in_file'])
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['mask_files']),
            name='output_node')
    # Define the node options
    binary_filter.inputs.operation='bin'
    erosion_filter.inputs.operation='ero'
    erosion_filter.inputs.operand_value=1
    lconcomp_filter.inputs.operation='lconcomp'
    dilation_filter.inputs.operation='dil'
    dilation_filter.inputs.operand_value=5
    fill_filter.inputs.operation='fill'
    fill_filter.inputs.output_datatype='char'
    # Create the connections
    workflow.connect(input_node, 'par_files', binary_filter, 'in_file')
    workflow.connect(binary_filter, 'out_file', erosion_filter, 'in_file')
    workflow.connect(erosion_filter, 'out_file', lconcomp_filter, 'in_file')
    workflow.connect(lconcomp_filter, 'out_file', dilation_filter, 'in_file')
    workflow.connect(dilation_filter, 'out_file', fill_filter, 'in_file')
    workflow.connect(fill_filter, 'out_file', output_node, 'mask_files')    
    
    return workflow
    
def regroup_roi(in_file, roi_list):
    import nibabel as nib
    import numpy as np
    import os
    # Load the parcelation
    parcelation=nib.load(in_file)
    data=parcelation.get_data()
    # Create a new empy segmentation
    new_roi_data=np.zeros(data.shape)
    # Extract the relevant roi from the initial parcelation
    for i in roi_list:
        new_roi_data=new_roi_data + np.equal(data,i*np.ones(data.shape))
    new_roi_data = new_roi_data!=0
    # Create a new image based on the initial parcelation
    out_img = nib.Nifti1Image(np.uint8(new_roi_data), parcelation.get_affine())
    out_file=os.getcwd()+'/roi_'+os.path.basename(in_file)
    out_img.set_data_dtype('uint8')
    out_img.set_qform(parcelation.get_qform())
    out_img.set_sform(parcelation.get_sform())
    out_img.set_filename(out_file)
    out_img.to_filename(out_file)
    return out_file


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

def create_reg_avg_value_pipeline(reg_avg_value_var):
    workflow = pe.Workflow(name='reg_avg_value_pipeline')
    workflow.base_dir = os.getcwd()
    workflow.base_output_dir='reg_avg_value'
    # Create the input node interface
    input_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['functional_files',
                    'mri_files',
                    'par_files',
                    'seg_files',
                    'trans_files']),
            name='input_node')
    input_node.inputs.functional_files=reg_avg_value_var.input_img
    input_node.inputs.mri_files=reg_avg_value_var.input_mri
    input_node.inputs.par_files=reg_avg_value_var.input_par
    input_node.inputs.seg_files=reg_avg_value_var.input_seg
    if len(reg_avg_value_var.input_trans) > 0:
        input_node.inputs.trans_files=reg_avg_value_var.input_trans
        # The input transformations are inverted
        invert_affine=pe.MapNode(interface = niftyreg.RegTransform(), \
            name='invert_affine', iterfield=['inv_aff_input'])
        workflow.connect(input_node, 'trans_files', invert_affine, 'inv_aff_input')
    else:
        # First step is to generate quick masks for both MRI and functional images
        func_mask = create_mask_from_functional()
        mri_mask = create_mask_from_parcelation()
        # Connections
        workflow.connect(input_node, 'functional_files', func_mask, 'input_node.functional_files')
        workflow.connect(input_node, 'par_files', mri_mask, 'input_node.par_files')
        # The MRI and functional images are globally registered
        aladin=pe.MapNode(interface = niftyreg.RegAladin(), name='aladin', \
            iterfield=['ref_file', 'flo_file', 'rmask_file', 'fmask_file'])
        if reg_avg_value_var.func_mri_use_affine==False:
            aladin.inputs.rig_only_flag=True
        aladin.inputs.verbosity_off_flag=True
        aladin.inputs.v_val=80
        aladin.inputs.nosym_flag=False
        # Connections
        workflow.connect(input_node, 'functional_files', aladin, 'ref_file')
        workflow.connect(input_node, 'mri_files', aladin, 'flo_file')
        workflow.connect(func_mask, 'output_node.mask_files', aladin, 'rmask_file')
        workflow.connect(mri_mask, 'output_node.mask_files', aladin, 'fmask_file')
    # The grey matter segmentation is exracted
    extract_timepoint=pe.MapNode(interface = niftyseg.BinaryMaths(), \
        name='extract_timepoint', iterfield=['in_file'])
    extract_timepoint.inputs.operation='tp'
    extract_timepoint.inputs.operand_value=int(2)
    # Connections
    workflow.connect(input_node, 'seg_files', extract_timepoint, 'in_file')
    # The segmentation is resampled in the space of functional images
    resample_seg=pe.MapNode(interface = niftyreg.RegResample(), name='resample_seg', \
        iterfield=['ref_file', 'flo_file', 'trans_file'])
    resample_seg.inputs.inter_val='LIN'
    resample_seg.inputs.verbosity_off_flag=True
    workflow.connect(input_node, 'functional_files', resample_seg, 'ref_file')
    workflow.connect(extract_timepoint, 'out_file', resample_seg, 'flo_file')
    workflow.connect(aladin, 'aff_file', resample_seg, 'trans_file')
    # The parcelation is resampled in the space of functional images
    resample_par=pe.MapNode(interface = niftyreg.RegResample(), name='resample_par', \
        iterfield=['ref_file', 'flo_file', 'trans_file'])
    resample_par.inputs.inter_val='NN'
    resample_par.inputs.verbosity_off_flag=True
    workflow.connect(input_node, 'functional_files', resample_par, 'ref_file')
    workflow.connect(input_node, 'par_files', resample_par, 'flo_file')
    if len(reg_avg_value_var.input_trans) > 0:
        workflow.connect(invert_affine, 'out_file', resample_par, 'trans_file')
    else:
        workflow.connect(aladin, 'aff_file', resample_par, 'trans_file')
    # Extract all the regional update values
    extract_uptakes = pe.MapNode(interface = ExtractRoiStatistics(), name='extract_uptakes',
        iterfield=['in_file','roi_file'])
    workflow.connect(input_node, 'functional_files', extract_uptakes, 'in_file')
    workflow.connect(resample_par, 'res_file', extract_uptakes, 'roi_file')
    # The ROI used for normalisation is extracted if required
    if reg_avg_value_var.roi=='gm_cereb':
        # Extract the cerebellum information
        extract_cerebellum = pe.MapNode(interface = 
                                            niu.Function(input_names = ['in_file',
                                                                        'roi_list'],
                                           output_names = ['out_file'],
                                           function=regroup_roi),
                                        name='extract_cerebellum',
                                        iterfield=['in_file'])
        extract_cerebellum.inputs.roi_list=[39,40,41,42,72,73,74]
        workflow.connect(resample_par, 'res_file', extract_cerebellum, 'in_file')
        # Binarise the grey matter segmentation
        binarise_gm_seg=pe.MapNode(interface = niftyseg.BinaryMaths(), \
            name='binarise_gm_seg', iterfield=['in_file'])
        binarise_gm_seg.inputs.operation='thr'
        binarise_gm_seg.inputs.operand_value=reg_avg_value_var.seg_threshold
        workflow.connect(resample_seg, 'res_file', binarise_gm_seg, 'in_file')
        # Extract the interaction between cerebellum and grey matter segmentation
        get_gm_cereb=pe.MapNode(interface = niftyseg.BinaryMaths(), \
            name='get_gm_cereb', iterfield=['in_file', 'operand_file'])
        get_gm_cereb.inputs.operation='mul'
        workflow.connect(binarise_gm_seg, 'out_file', get_gm_cereb, 'in_file')
        workflow.connect(extract_cerebellum, 'out_file', get_gm_cereb, 'operand_file')
        
        extract_cereb_uptake = pe.MapNode(interface = ExtractRoiStatistics(), \
            name='extract_cereb_uptake', iterfield=['in_file','roi_file'])
        workflow.connect(input_node, 'functional_files', extract_cereb_uptake, 'in_file')
        workflow.connect(get_gm_cereb, 'out_file', extract_cereb_uptake, 'roi_file')
        
        # Normalise the uptake values
        norm_uptakes = pe.MapNode(interface = NormaliseRoiAverageValues(), name='norm_uptakes',
            iterfield=['in_file','in_array', 'cereb_array'])
        workflow.connect(input_node, 'functional_files', norm_uptakes, 'in_file')
        workflow.connect(extract_uptakes, 'out_array', norm_uptakes, 'in_array')
        workflow.connect(extract_cereb_uptake, 'out_array', norm_uptakes, 'cereb_array')
        norm_uptakes.inputs.roi=reg_avg_value_var.roi
    else:
        # Normalise the uptake values
        norm_uptakes = pe.MapNode(interface = NormaliseRoiAverageValues(), name='norm_uptakes',
            iterfield=['in_file','in_array'])
        workflow.connect(input_node, 'functional_files', norm_uptakes, 'in_file')
        workflow.connect(extract_uptakes, 'out_array', norm_uptakes, 'in_array')
        norm_uptakes.inputs.roi=reg_avg_value_var.roi

    # Create the output node interface
    output_node = pe.Node(
        interface = niu.IdentityInterface(
            fields=['out_files',
                    'norm_files',
                    'aff_files']),
            name='output_node')
    # Fake connections for testing
    workflow.connect(norm_uptakes, 'out_csv_file', output_node, 'out_files')
    workflow.connect(norm_uptakes, 'out_file', output_node, 'norm_files')
    if len(reg_avg_value_var.input_trans) > 0:
        workflow.connect(invert_affine, 'out_file', output_node, 'aff_files')
    else:
        workflow.connect(aladin, 'aff_file', output_node, 'aff_files')
    
    # Return the created workflow
    return workflow
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
#    workflow.write_graph(graph2use='colored')
    workflow.run(plugin='MultiProc')
    
if __name__ == "__main__":
    main()