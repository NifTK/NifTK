"""
   Interface for niftk filter tools
"""

from nipype.interfaces.base import (TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec)

import os


class WriteArrayToCsvInputSpec(BaseInterfaceInputSpec):
    in_array = traits.Array(exists=True, mandatory=True,
                        desc="array")           
    in_name = traits.String(mandatory=True, desc="Name of the output file")
    
class WriteArrayToCsvOutputSpec(TraitedSpec):
    out_file   = File(desc="Output file")


class WriteArrayToCsv(BaseInterface):

    """

    Examples
    --------

    """

    input_spec = WriteArrayToCsvInputSpec  
    output_spec = WriteArrayToCsvOutputSpec

    def _run_interface(self, runtime):
        in_array = self.inputs.in_array
        in_name = self.inputs.in_name
        out_file = self._gen_output_filename(in_name) 
        f=open(out_file, 'w+')
        s = in_array.shape
        for i in range(s[0]):
            for j in range(s[1]):
                f.write("%f," % in_array[i,j])
            f.write("\n")
        f.close        
        
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename(self.inputs.in_name); 
        return outputs
    
    def _gen_output_filename(self, in_name):
        outfile = os.path.abspath(in_name + '.csv')
        return outfile
        




