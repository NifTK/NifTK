# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the RestingStatefMRIPreprocess.sh script
    """

import os

from nipype.interfaces.fsl.base import FSLCommand as RestingStatefMRIPreprocessCommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec as RestingStatefMRIPreprocessCommandInputSpec

from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, OutputMultiPath,
                                    isdefined)

class RestingStatefMRIPreprocessInputSpec(RestingStatefMRIPreprocessCommandInputSpec):
    
    # fmri id - identifier
    in_subjectid = traits.String(argstr="%s", mandatory=True, position = 2, desc="subjec id")
    
    # fmri raw input file
    in_fmri = File(exists=True, mandatory=True, argstr="%s", position = 3, desc="raw fmri input file")
    # anatomical t1 image file
    in_anat = File(exists=True, mandatory=True, argstr="%s", position = 4, desc="anatomical input file")
    # group average
    in_group_avg = File(exists=True, mandatory=True, argstr="%s", position = 5, desc="group average (such as MNI)")
    # anat to group transformation
    in_group_trans = File(exists=True, mandatory=True, argstr="%s", position = 6, desc="group average transformation (transformation field as *.nii.gz)")
    # segmentation of the T1 scan (including white matter, csf and grey matter segmentation)
    in_seg = File(exists=True, mandatory=True, argstr="%s", position = 7, desc="segmentation of the T1 scan (including csf pos1 ,grey pos2 and white matter pos3)")
    in_atlas = File(exists=True, mandatory=True, argstr="%s", position = 8, desc="segmentation of the T1 scan (including csf pos1 ,grey pos2 and white matter pos3)")

class RestingStatefMRIPreprocessOutputSpec(TraitedSpec):
    # preprocessed fmri scan in subject space
    out_fmri   = File(exists=True, genfile = True, desc="preprocessed fMRI scan in subject space")
    out_fmri_group = File(exists=True, genfile = True, desc="preprocessed fMRI scan in group space")
    out_fmri_minout_group = File(exists=True, genfile = True, desc="minimal outlier fMRI volume mapped into group space")

class RestingStatefMRIPreprocess(RestingStatefMRIPreprocessCommand):
    
    _cmd = "sh fmri_prep_single.sh"
    input_spec = RestingStatefMRIPreprocessInputSpec
    output_spec = RestingStatefMRIPreprocessOutputSpec
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_fmri'] = os.path.join(os.path.dirname(self.inputs.in_fmri), self.inputs.in_subjectid + '.fmri_pp.nii.gz')
        print outputs['out_fmri']
        outputs['out_fmri_group'] = os.path.join(os.path.dirname(self.inputs.in_fmri), self.inputs.in_subjectid + '.fmri_pp_group.nii.gz')
        outputs['out_fmri_minout_group'] = os.path.join(os.path.dirname(self.inputs.in_fmri), self.inputs.in_subjectid + '.fmri_reg_group.nii.gz')
        print outputs
        return outputs

