"""
    Simple interface to extract the axis information
"""
import os
import os.path

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, traits)

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

