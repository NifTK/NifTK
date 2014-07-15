#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import nipype.interfaces.dcm2nii        as mricron
import nipype.interfaces.niftyreg       as niftyreg
import diffusion_mri_processing         as dmri
import argparse
import os

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')

def find_and_merge_dwi_data (input_bvals, input_bvecs, input_files):
    import os, glob
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
    for bvals_file in input_bvals:
        dwi_base, _ = os.path.splitext(bvals_file)
        dwis_files.append(dwi_base + '.nii.gz')
    merger = fsl.Merge(dimension = 't')
    merger.inputs.in_files = dwis_files
    res = merger.run()

    dwis = res.outputs.merged_file
    bvals = merge_vector_files(input_bvals)
    bvecs = merge_vector_files(input_bvecs)
    fms = glob.glob(input_path + os.sep + '*fieldmap*.nii*')
    fmmag = fms[0]
    fmph  = fms[1]
    t1s = glob.glob(input_path + os.sep + '*MPRAGE*.nii*')
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
dcm2nii.inputs.reorient = False
dcm2nii.inputs.reorient_and_crop = False
dcm2nii.inputs.anonymize = False
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

ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = result_dir
ds.inputs.parameterization = False

r.connect(r.get_node('output_node'), 'tensor', ds, '@tensors')
r.connect(r.get_node('output_node'), 'FA', ds, '@fa')
r.connect(r.get_node('output_node'), 'MD', ds, '@md')
r.connect(r.get_node('output_node'), 'COL_FA', ds, '@colfa')
r.connect(r.get_node('output_node'), 'V1', ds, '@v1')
r.connect(r.get_node('output_node'), 'predicted_image', ds, '@img')
r.connect(r.get_node('output_node'), 'residual_image', ds, '@res')
r.connect(r.get_node('output_node'), 'parameter_uncertainty_image', ds, '@unc')
r.connect(r.get_node('output_node'), 'dwis', ds, '@dwis')
r.connect(r.get_node('output_node'), 'transformations', ds, 'transformations')
r.connect(r.get_node('output_node'), 'average_b0', ds, '@b0')

r.write_graph(graph2use = 'colored')

qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'

#r.run()
#r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
#r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
r.run(plugin='MultiProc')

