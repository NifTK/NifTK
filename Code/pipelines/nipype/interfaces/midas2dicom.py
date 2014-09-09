# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the Diffusion distortion simulation script
"""

import os
import os.path
import glob
import subprocess
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, InputMultiPath,
                                    isdefined)

from nipype.utils.filemanip import split_filename


    
def convert_midas2_dicom(midas_code, midas_dirs):
    # Check if 4 or 5 digits long
    if len(midas_code) == 4:
        midas_code = "0"+midas_code
    
    midas_ims = list()    
    # look through various database paths to find a corresponding midas image
    for test_dir in midas_dirs:
        files = glob.glob(os.path.normpath(test_dir) + os.path.sep + midas_code + "-00*-1.hdr")
        if len(files) > 0:
            midas_ims.append(files[0])

    if len(midas_ims) == 0:
        print "NO FILE FOUND"
        return None
    elif len(midas_ims) > 1:
        print "TOO MANY FILES FOUND: " + midas_ims
        return None
    print 'file from search is: ', midas_ims

    # call getdicompath.sh to find the first dicom file
    command = 'sh /var/lib/midas/pkg/x86_64-unknown-linux-gnu/dicom-to-midas/getdicompath.sh ' + midas_ims[0]

    dicom_file = subprocess.check_output(command.split())
    dicom_dir = os.path.dirname(dicom_file)
    print "Dicom directory is: "+dicom_dir
    # return the directory
    return dicom_dir

class Midas2DicomInputSpec(BaseInterfaceInputSpec):
    midas_code = traits.String(mandatory=True, desc="4/5 digit midas code")
    midas_dirs = InputMultiPath(mandatory=True, desc="Midas database directories to search")
    
class Midas2DicomOutputSpec(TraitedSpec):
    dicom_dir   = Directory(exists=True, desc="Dicom directory path of inputted midas image")

class Midas2Dicom(BaseInterface):

    """

    Examples
    --------

    """
    input_spec = Midas2DicomInputSpec  
    output_spec = Midas2DicomOutputSpec
        
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['dicom_dir'] = self.dicom_dir
        return outputs

    def _run_interface(self, runtime):
        self.dicom_dir = convert_midas2_dicom(self.inputs.midas_code, self.inputs.midas_dirs)
        return runtime
        
