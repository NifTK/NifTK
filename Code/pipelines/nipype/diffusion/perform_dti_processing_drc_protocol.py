#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe
import nipype.interfaces.dcm2nii        as mricron
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyseg       as niftyseg
import diffusion_mri_processing         as dmri
import argparse
import os

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')

def find_and_merge_dwi_data (input_bvals, input_bvecs, input_files):

    import os, glob, sys, errno
    import nipype.interfaces.fsl as fsl
    
    def merge_vector_files(input_files):
        import numpy as np
        import os
        result = np.array([])
        files_base, files_ext = os.path.splitext(os.path.basename(input_files[0]))
        for f in input_files:
            if result.size == 0:
                result = np.loadtxt(f)
            else:
                result = np.hstack((result, np.loadtxt(f)))
        output_file = os.path.abspath(files_base + '_merged' + files_ext)
        np.savetxt(output_file, result, fmt = '%.3f')
        return output_file
    
    input_path = os.path.dirname(input_files[0])
    dwis_files = []

    if len(input_bvals) == 0 or len(input_bvecs) == 0:
        print 'I/O Could not any diffusion based images in ', input_path, ', exiting.'
        sys.exit(errno.EIO)

    for bvals_file in input_bvals:
        dwi_base, _ = os.path.splitext(bvals_file)
        dwi_file = dwi_base + '.nii.gz'
        if not os.path.exists(dwi_file):
            dwi_file = dwi_base + '.nii'
        if not os.path.exists(dwi_file):
            print 'I/O The DWI file with base ', dwi_base, ' does not exist, exiting.'
            sys.exit(errno.EIO)
        dwis_files.append(dwi_file)

    merger = fsl.Merge(dimension = 't')
    merger.inputs.in_files = dwis_files
    res = merger.run()
    dwis = res.outputs.merged_file
    bvals = merge_vector_files(input_bvals)
    bvecs = merge_vector_files(input_bvecs)

    fms = glob.glob(input_path + os.sep + '*fieldmap*.nii*')
    if len(fms) == 0:
        print 'I/O Could not find any field map image in ', input_path, ', exiting.'
        sys.exit(errno.EIO)
    if len(fms) < 2:
        print 'I/O Field Map error: either the magnitude or phase is missing in ', input_path, ', exiting.'
        sys.exit(errno.EIO)
    if len(fms) > 2:
        print 'I/O Field Map warning: there are more field map images than expected in ', input_path, ', \nAssuming the first two are relevant...'
    fmmag = fms[0]
    fmph  = fms[1]

    t1s = glob.glob(input_path + os.sep + 'o*MPRAGE*.nii*')
    if len(t1s) == 0:
        t1s = glob.glob(input_path + os.sep + '*MPRAGE*.nii*')
    if len(t1s) == 0:
        print 'I/O Could not find any MPRAGE image in ', input_path, ', exiting.'
        sys.exit(errno.EIO)
    if len(t1s) > 1:
        print 'I/O warning: there is more than 1 MPRAGE image in ', input_path, ', \nAssuming the first one is relevant...'    
    t1 = t1s[0]

    return dwis, bvals, bvecs, fmmag, fmph, t1

parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-i', '--dicoms',
                    dest='dicoms',
                    metavar='dicoms',
                    help='DICOM directory where the files are stored',
                    required=True)
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='Result directory where the output data is to be stored',
                    required=False,
                    default='results')
parser.add_argument('-p', '--prefix',
                    dest='prefix',
                    metavar='prefix',
                    help='prefix to use for the output images',
                    required=False,
                    default='subject')

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

input_dir = os.path.join(result_dir, 'inputs')
if not os.path.exists(input_dir):
    os.mkdir(input_dir)

r = dmri.create_diffusion_mri_processing_workflow('dmri_workflow',
                                                  resample_in_t1 = True, 
                                                  t1_mask_provided = True,
                                                  log_data = True)
r.base_dir = os.getcwd()

dg = pe.Node(nio.DataGrabber(outfields = ['dicom_files']), name='dg')
dg.inputs.template = '*'
dg.inputs.sort_filelist = False
dg.inputs.base_directory = os.path.abspath(args.dicoms)

dcm2nii = pe.Node(interface = mricron.Dcm2nii(), 
                  name = 'dcm2nii')
dcm2nii.inputs.args = '-d n'
dcm2nii.inputs.gzip_output = True
dcm2nii.inputs.anonymize = False
dcm2nii.inputs.reorient = True
dcm2nii.inputs.reorient_and_crop = False
dcm2nii.inputs.output_dir = input_dir

find_and_merge_dwis = pe.Node(interface = niu.Function(input_names = ['input_bvals', 'input_bvecs', 'input_files'],
                                                       output_names = ['dwis', 'bvals', 'bvecs', 'fieldmapmag', 'fieldmapphase', 't1'],
                                                       function = find_and_merge_dwi_data),
                        name = 'find_and_merge_dwis')

mni_to_input = pe.Node(interface=niftyreg.RegAladin(), name='mni_to_input')
mni_to_input.inputs.flo_file = mni_template

mask_resample  = pe.Node(interface = niftyreg.RegResample(), name = 'mask_resample')
mask_resample.inputs.inter_val = 'NN'
mask_resample.inputs.flo_file = mni_template_mask

r.connect(dg, 'dicom_files', dcm2nii, 'source_names')
r.connect(dcm2nii, 'converted_files', find_and_merge_dwis, 'input_files')
r.connect(dcm2nii, 'bvals', find_and_merge_dwis, 'input_bvals')
r.connect(dcm2nii, 'bvecs', find_and_merge_dwis, 'input_bvecs')
r.connect(find_and_merge_dwis, 't1', mni_to_input, 'ref_file')
r.connect(find_and_merge_dwis, 't1', mask_resample, 'ref_file')
r.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
r.connect(mask_resample, 'res_file', r.get_node('input_node'), 'in_t1_mask')
r.connect(find_and_merge_dwis, 'dwis', r.get_node('input_node'), 'in_dwi_4d_file')
r.connect(find_and_merge_dwis, 'bvals', r.get_node('input_node'), 'in_bval_file')
r.connect(find_and_merge_dwis, 'bvecs', r.get_node('input_node'), 'in_bvec_file')
r.connect(find_and_merge_dwis, 'fieldmapmag', r.get_node('input_node'), 'in_fm_magnitude_file')
r.connect(find_and_merge_dwis, 'fieldmapphase', r.get_node('input_node'), 'in_fm_phase_file')
r.connect(find_and_merge_dwis, 't1', r.get_node('input_node'), 'in_t1_file')

mask_eroder = pe.Node(interface = niftyseg.BinaryMaths(), 
                      name = 'mask_eroder')
mask_eroder.inputs.operation = 'ero'
mask_eroder.inputs.operand_value = 3

multiplicater_fa = pe.Node(interface = niftyseg.BinaryMaths(), 
                           name = 'multiplicater_fa')
multiplicater_fa.inputs.operation = 'mul'
multiplicater_md = pe.Node(interface = niftyseg.BinaryMaths(), 
                           name = 'multiplicater_md')
multiplicater_md.inputs.operation = 'mul'
multiplicater_v1 = pe.Node(interface = niftyseg.BinaryMaths(), 
                           name = 'multiplicater_v1')
multiplicater_v1.inputs.operation = 'mul'
multiplicater_colfa = pe.Node(interface = niftyseg.BinaryMaths(), 
                           name = 'multiplicater_colfa')
multiplicater_colfa.inputs.operation = 'mul'

r.connect(mask_resample, 'res_file', mask_eroder, 'in_file')
r.connect(mask_eroder, 'out_file', multiplicater_fa, 'operand_file')
r.connect(mask_eroder, 'out_file', multiplicater_md, 'operand_file')
r.connect(mask_eroder, 'out_file', multiplicater_v1, 'operand_file')
r.connect(mask_eroder, 'out_file', multiplicater_colfa, 'operand_file')
r.connect(r.get_node('output_node'), 'FA', multiplicater_fa, 'in_file')
r.connect(r.get_node('output_node'), 'MD', multiplicater_md, 'in_file')
r.connect(r.get_node('output_node'), 'V1', multiplicater_v1, 'in_file')
r.connect(r.get_node('output_node'), 'COL_FA', multiplicater_colfa, 'in_file')

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = result_dir
ds.inputs.parameterization = False

r.connect(r.get_node('output_node'), 'tensor', ds, '@tensors')
r.connect(multiplicater_fa, 'out_file', ds, '@fa')
r.connect(multiplicater_md, 'out_file', ds, '@md')
r.connect(multiplicater_colfa, 'out_file', ds, '@colfa')
r.connect(multiplicater_v1, 'out_file', ds, '@v1')
r.connect(r.get_node('output_node'), 'predicted_image', ds, '@img')
r.connect(r.get_node('output_node'), 'residual_image', ds, '@res')
r.connect(r.get_node('output_node'), 'parameter_uncertainty_image', ds, '@unc')
r.connect(r.get_node('output_node'), 'dwis', ds, '@dwis')
#r.connect(r.get_node('output_node'), 'transformations', ds, 'transformations')
r.connect(r.get_node('output_node'), 'average_b0', ds, '@b0')

r.write_graph(graph2use = 'colored')

qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
#r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
r.run(plugin='MultiProc')

