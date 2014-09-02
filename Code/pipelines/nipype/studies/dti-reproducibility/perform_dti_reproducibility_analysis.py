#! /usr/bin/env python

import nipype.interfaces.utility        as niu            
import nipype.interfaces.io             as nio     
import nipype.pipeline.engine           as pe          
import diffusion_mri_processing         as dmri
import os
import glob
import nipype.interfaces.fsl as fsl

def merge_vector_files(input_files, basename):
        import numpy as np
        import os
        result = np.array([])
        files_base, files_ext = os.path.splitext(os.path.basename(input_files[0]))
        for f in input_files:
            if result.size == 0:
                result = np.loadtxt(f)
            else:
                result = np.hstack((result, np.loadtxt(f)))
        output_file = os.path.abspath(basename + files_ext)
        np.savetxt(output_file, result, fmt = '%.3f')
        return output_file

op_root_dir = '/Users/isimpson/Documents/DWI/DTIReproPaper/data/back_to_back/'
temp_folder = op_root_dir + '/temp/'
subj_data_dirs = ['/Users/isimpson/Data/me/nifti/','']
subj_labels = ['ivor', 'nico']
stamps = [['20130717_133322','20130717_151013'], []]

interp_options = ['LIN', 'CUB']


for interp_option in interp_options:
    for subj_index in range(len(subj_data_dirs)):
    subj_data_dir = subj_data_dirs[subj_index]
    subj_label = subj_labels[subj_index]
    t1 = glob.glob(subj_data_dir+'o*SagMPRAGE*.nii.gz')[0]
    subj_stamps = stamps[subj_index]
    for stamp_index in range(len(subj_stamps)):
        stamp = subj_stamps[subj_index]

        fm = glob.glob(subj_data_dir+stamp+'*fieldmappings*.nii.gz')
        dwi = glob.glob(subj_data_dir+stamp+'*ep2ddiff*.nii.gz')

        bvec = glob.glob(subj_data_dir+stamp+'*.bvec')
        bval = glob.glob(subj_data_dir+stamp+'*.bval')
        combo_index = 0
        for i in range(4):
            for j in range(i+1,4):
                op_folder = op_root_dir+subj_label+'_'+str(stamp_index)+'_'+str(combo_index)+'_'+interp_option+'/'
                if not os.path.exists(op_folder):
                    os.mkdir(op_folder)
                temp_data_basename = op_folder+'merged'
                temp_data = temp_data_basename+'.nii.gz'

                if not os.path.exists(temp_data):
                    dwis_files = [dwi[i], dwi[j]]
                    bvals_files = [bval[i], bval[j]]
                    bvecs_files = [bvec[i], bvec[j]]

                    merger = fsl.Merge(dimension = 't')
                    merger.inputs.in_files = dwis_files
                    merger.inputs.merged_file = temp_data
                    res = merger.run()

                    res.outputs.merged_file
                    merge_vector_files(bvals_files, temp_data_basename)
                    merge_vector_files(bvecs_files, temp_data_basename)


                if os.path.exists(op_folder+'tensors.nii.gz'):
                    r = dmri.create_diffusion_mri_processing_workflow(name = 'dmri_workflow',
                                                      resample_in_t1 = False,
                                                      log_data = False,
                                                      correct_susceptibility = True,
                                                      dwi_interp_type = interp_option,
                                                      t1_mask_provided = False,
                                                      ref_b0_provided = False)
                    r.base_dir = temp_folder

                    r.inputs.input_node.in_dwi_4d_file = os.path.abspath(temp_data)
                    r.inputs.input_node.in_bvec_file = os.path.abspath(temp_data_basename+'.bvec')
                    r.inputs.input_node.in_bval_file = os.path.abspath(temp_data_basename+'.bval')
                    r.inputs.input_node.in_fm_magnitude_file = os.path.abspath(fm[0])
                    r.inputs.input_node.in_fm_phase_file = os.path.abspath(fm[1])
                    r.inputs.input_node.in_t1_file = os.path.abspath(t1)

                    ds = pe.Node(nio.DataSink(), name='ds')
                    ds.inputs.base_directory = op_folder
                    ds.inputs.parameterization = False

                    r.connect(r.get_node('output_node'), 'tensor', ds, '@tensors')
                    r.connect(r.get_node('output_node'), 'FA', ds, '@fa')
                    r.connect(r.get_node('output_node'), 'MD', ds, '@md')
                    r.connect(r.get_node('output_node'), 'COL_FA', ds, '@colfa')
                    r.connect(r.get_node('output_node'), 'dwis', ds, '@dwis')
                    r.connect(r.get_node('output_node'), 'average_b0', ds, '@b0')
                    qsubargs='-l h_rt=00:05:00 -l tmem=1.8G -l h_vmem=1.8G -l vf=2.8G -l s_stack=10240 -j y -b y -S /bin/csh -V'
                    #r.run(plugin='SGE',       plugin_args={'qsub_args': qsubargs})
                    #r.run(plugin='SGEGraph',  plugin_args={'qsub_args': qsubargs})
                    r.run(plugin='MultiProc')

                combo_index += 1









