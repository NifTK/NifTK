"""
   Interface for niftk filter tools
"""

from niftk.base import NIFTKCommand, NIFTKCommandInputSpec, getNiftkPath
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, InputMultiPath, BaseInterface, BaseInterfaceInputSpec)

from nipype.utils.filemanip import split_filename
import os
import glob
import subprocess






class Midas2NiiInputSpec(NIFTKCommandInputSpec):
    in_file = File(argstr="%s", exists=True, mandatory=True, position = 2,
                desc="Input image filename ** WITH .IMG EXTENSION **")
    out_file = File(argstr="%s", position = -2, name_source = ['in_file'], name_template = '%s_nii',
                 desc="Output file")

class Midas2NiiOutputSpec(TraitedSpec):
    out_file   = File(desc="Output nii image file")

class Midas2Nii(NIFTKCommand):
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
        







class GetAxisOrientationInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(argstr="%s", exists=True, mandatory=True, \
                desc="Input target image filename")

class GetAxisOrientationOutputSpec(TraitedSpec):

    out_dict = traits.Dict(desc='Dictionnary containing the axis orientation.')

class GetAxisOrientation(BaseInterface):

    input_spec = GetAxisOrientationInputSpec  
    output_spec = GetAxisOrientationOutputSpec
    
    def _run_interface(self, runtime):
        input_file = self.inputs.in_file
        self.out_dict=self.get_axis_orientation(input_file)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dict'] = self.out_dict
        return outputs
        

    def get_axis_orientation(self,
                             input):
        outfile=os.path.abspath('fslhd_out.txt')
        f=open(outfile, 'w+')
        os.system('fslhd -x ' + input + ' > '+ outfile)
        fslDict = {}
        for line in f:                
            listedline = line.strip().split('=') # split around the = sign
            if len(listedline) > 1: # we have the = sign in there
                fslDict[listedline[0].replace(" ", "")] = listedline[1].replace(" ", "")           
        f.close()
        out_dict=dict()
        if fslDict['sform_code'] > 0:
            out_dict['i']=fslDict['sform_i_orientation']
            out_dict['j']=fslDict['sform_j_orientation']
            out_dict['k']=fslDict['sform_k_orientation']
        else:
            out_dict['i']=fslDict['qform_i_orientation']
            out_dict['j']=fslDict['qform_j_orientation']
            out_dict['k']=fslDict['qform_k_orientation']
        return out_dict

