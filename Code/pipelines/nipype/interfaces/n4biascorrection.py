# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the CropImage.sh script
"""

import os
import numpy as np
from nibabel import load
import os.path as op
import warnings

from nipype.interfaces.niftyseg.base import NIFTYSEGCommand, NIFTYSEGCommandInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, InputMultiPath,
                                    isdefined)

class N4BiasCorrectionInputSpec(NIFTYSEGCommandInputSpec):
    
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                desc="Input target image filename")
    
    mask_file = File(exists=True, argstr="--inMask %s",
                     desc="Mask over the input image")

    out_file = File(argstr="-o %s", 
                    desc="output bias corrected image",
                    name_source = ['in_file'],
                    name_template = '%s_corrected')

    in_levels = traits.BaseInt(desc='Number of Multi-Scale Levels - optional - default = 3', 
                               argstr='--nlevels %d')

    in_downsampling = traits.BaseInt(desc='Level of Downsampling - optional - default = 1 (no downsampling), downsampling to level 2 is recommended', 
                                     argstr='--sub %d')

    in_maxiter = traits.BaseInt(desc='Maximum number of Iterations - optional - default = 50', 
                                argstr='--niters %d')

    out_biasfield_file = File(argstr="--outBiasField %s", 
                              desc="output bias field file",
                              name_source = ['in_file'],
                              name_template = '%s_biasfield')

    in_convergence = traits.Float(desc='Convergence Threshold - optional - default = 0.001', argstr='-c %f')


class N4BiasCorrectionOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc="output bias corrected image")
    out_biasfield_file = File(desc="output bias field")

class N4BiasCorrection(NIFTYSEGCommand):

    _cmd = "niftkN4BiasFieldCorrection"

    input_spec = N4BiasCorrectionInputSpec  
    output_spec = N4BiasCorrectionOutputSpec

