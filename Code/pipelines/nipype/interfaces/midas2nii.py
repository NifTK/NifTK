"""
    Simple interface for the Midas2Nii.sh script
"""

from nipype.interfaces.fsl.base import FSLCommand as MIDAS2NIICommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec as MIDAS2NIICommandInputSpec
from nipype.interfaces.base import (File)

class Midas2NiiInputSpec(MIDAS2NIICommandInputSpec):
    in_file = File(argstr="%s", exists=True, mandatory=True, position = 2,
                desc="Input image filename ** WITH .IMG EXTENSION **")
    out_file = File(argstr="%s", position = -2, name_source = ['in_file'], name_template = '%s_nii',
                 desc="Output file")

class Midas2NiiOutputSpec(TraitedSpec):
    out_file   = File(desc="Output nii image file")
class Midas2Nii(MIDAS2NIICommand):
    """
    Converts MIDAS Analyse formatted images into normal NIFTI ones. 
    The input needs to be the .img file and the header .hdr needs to be present in the same directory
    
    Example
    --------
    from midas2nii import Midas2Nii
    converter = Midas2Nii()
    converter.inputs.in_file = "030583-T1.img"
    converter.run()
    """
    _cmd = "/var/drc/software/32bit/nifti-midas/midas2nii.sh"
    _suffix = "_nii"
    input_spec = Midas2NiiInputSpec  
    output_spec = Midas2NiiOutputSpec
    _output_type = 'NIFTI'
