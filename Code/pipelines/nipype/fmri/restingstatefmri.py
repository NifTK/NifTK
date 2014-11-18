# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Interface for the fmri_prep_single.sh script
    """

import os

from nipype.interfaces.fsl.base import FSLCommand as RestingStatefMRIPreprocessCommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec as RestingStatefMRIPreprocessCommandInputSpec

from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, OutputMultiPath,
                                    isdefined)

class RestingStatefMRIPreprocessInputSpec(RestingStatefMRIPreprocessCommandInputSpec):
    
    # fmri raw input file
    in_fmri = File(exists=True, 
                   mandatory=True, 
                   argstr="%s", 
                   position = 3, 
                   desc="raw fmri input file")
    # anatomical t1 image file
    in_t1 = File(exists=True, 
                 mandatory=True, 
                 argstr="%s", 
                 position = 4, 
                 desc="anatomical input file")
    # segmentation of the T1 scan (including white matter, csf and grey matter segmentation)
    in_tissue_segmentation = File(exists=True, 
                                  mandatory=True, 
                                  argstr="%s", 
                                  position = 7, 
                                  desc="segmentation of the T1 scan (including csf pos1 ,grey pos2 and white matter pos3)")
    # atlas 
    in_parcellation = File(exists=True, 
                           mandatory=True, 
                           argstr="%s", 
                           position = 8, 
                           desc="segmentation of the T1 scan (including csf pos1 ,grey pos2 and white matter pos3)")

class RestingStatefMRIPreprocessOutputSpec(TraitedSpec):

    # preprocessed fmri scan in subject space
    out_corrected_fmri = File(exists=True, genfile = True, desc="preprocessed fMRI scan in subject space")
    out_fmri_to_t1_transformation = File(exists=True, genfile = True, desc="fMRI to T1 affine transformation")

class RestingStatefMRIPreprocess(RestingStatefMRIPreprocessCommand):    

    """

    Examples
    --------

    import restingstatefmri as fmri
    
    rs = fmri.RestingStatefMRIPreprocess()
    
    rs.inputs.in_fmri = "func.nii.gz"
    rs.inputs.in_t1 = "anat.nii.gz"
    rs.inputs.in_tissue_segmentation = "seg.nii.gz"
    rs.inputs.in_parcellation = "atlas.nii.gz"

    rs.run()

    """

    _cmd = "fmri_prep_single.sh"
    input_spec = RestingStatefMRIPreprocessInputSpec
    output_spec = RestingStatefMRIPreprocessOutputSpec
    
    def _list_outputs(self):

        outputs = self.output_spec().get()
        outputs['out_corrected_fmri'] = os.path.join(os.path.dirname(self.inputs.in_fmri), 'fmri_pp.nii.gz')
        outputs['out_fmri_to_t1_transformation'] = os.path.join(os.path.dirname(self.inputs.in_fmri), 'fmri_to_t1_transformation.txt')
#        outputs['out_fmri_group'] = os.path.join(os.path.dirname(self.inputs.in_fmri), self.inputs.in_subjectid + '.fmri_pp_group.nii.gz')
#        outputs['out_fmri_minout_group'] = os.path.join(os.path.dirname(self.inputs.in_fmri), self.inputs.in_subjectid + '.fmri_reg_group.nii.gz')
        return outputs

