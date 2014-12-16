# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
   Interface for niftk image operation tools
"""

import os

from niftk.base import NIFTKCommand, NIFTKCommandInputSpec, getNiftkPath
from nipype.interfaces.base import (TraitedSpec, File, traits)

class CropImageInputSpec(NIFTKCommandInputSpec):
    
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                desc="Input target image filename")
    
    mask_file = File(exists=True, argstr="-m %s",
                     desc="Mask over the input image")

    out_file = File(argstr="-o %s", 
                    desc="output cropped image",
                    name_source = ['in_file'],
                    name_template = '%s_cropped')

class CropImageOutputSpec(TraitedSpec):
    out_file   = File(exists=True, desc="Output cropped image file")

class CropImage(NIFTKCommand):

    """
    Examples
    --------
    import niftk as niftk
    cropper = CropImage()
    cropper.inputs.in_file = "T1.nii.gz"
    cropper.inputs.mask_file = "mask.nii.gz"
    cropper.inputs.out_file = "T1_cropped.nii.gz"
    cropper.run()
    """
    _cmd = getNiftkPath("niftkCropImage")

    input_spec = CropImageInputSpec  
    output_spec = CropImageOutputSpec


        
