"""
    Simple interface for the gradient_unwarp executable
"""

from nipype.interfaces.niftyseg.base import NIFTYSEGCommand, NIFTYSEGCommandInputSpec
from nipype.interfaces.base import (TraitedSpec, File, traits)

class GradwarpCorrectionInputSpec(NIFTYSEGCommandInputSpec):
    
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                desc="Input target image filename")
    
    coeff_file = File(argstr="-c %s", exists=True, mandatory=True,
                desc="Spherical harmonics coefficient filename")
    
    scanner_type = traits.BaseString(argstr="-t %s",
                desc="Scanner type: siemens or ge. siemens by default.")
    
    offset_x = traits.BaseFloat(argstr="-off_x %f",
                desc="Scanner offset along the x axis in mm [0].")
    
    offset_y = traits.BaseFloat(argstr="-off_y %f",
                desc="Scanner offset along the y axis in mm [0].")
    
    offset_z = traits.BaseFloat(argstr="-off_z %f",
                desc="Scanner offset along the z axis in mm [0].")

    out_file = File(argstr="-o %s", 
                    desc="output deformation field image",
                    name_source = ['in_file'],
                    name_template = '%s_unwarp_field')

class GradwarpCorrectionOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc="output deformation field image")

class GradwarpCorrection(NIFTYSEGCommand):

    _cmd = "gradient_unwarp"

    input_spec = GradwarpCorrectionInputSpec  
    output_spec = GradwarpCorrectionOutputSpec

