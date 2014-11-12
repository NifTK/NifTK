# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the CropImage.sh script
"""

import os

from nipype.interfaces.fsl.base import FSLCommand as CROPIMAGECommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec as CROPIMAGECommandInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, OutputMultiPath,
                                    isdefined)

class CropImageInputSpec(CROPIMAGECommandInputSpec):
    
    in_file = File(argstr="%s", exists=True, mandatory=True, position = 2,
                desc="Input target image filename")
    mask_file = File(exists=True, mandatory=True, argstr="%s", position = -3,
                     desc="Mask over the input image")
    out_file = File(exists=False, genfile = True, mandatory=False, argstr="%s", position = -2, 
                     desc="Mask over the input image [default: none]")

class CropImageOutputSpec(TraitedSpec):
    out_file   = File(exists=False, genfile = True, desc="Output cropped image file")

class CropImage(CROPIMAGECommand):

    """
    Examples
    --------
    from cropimage import CropImage
    cropper = CropImage()
    cropper.inputs.in_file = "T1.nii.gz"
    cropper.inputs.mask_file = "mask.nii.gz"
    cropper.inputs.out_file = "T1_cropped.nii.gz"
    cropper.run()
    """
    _cmd = "CropImage.sh"
    _suffix = "_crop_image"
    input_spec = CropImageInputSpec  
    output_spec = CropImageOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_fname(self.inputs.in_file, suffix=self._suffix, ext='.nii.gz')
        return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file) and self.inputs.out_file:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        else:
            outputs['out_file'] = self._gen_filename('out_file')        
        return outputs
        
