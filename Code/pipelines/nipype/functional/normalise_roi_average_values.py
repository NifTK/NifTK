#!/usr/bin/env python

from nipype.interfaces.base         import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base         import (TraitedSpec, File, traits)
from nipype.utils.filemanip         import split_filename
import nipype.interfaces.niftyseg   as niftyseg

import numpy                        as np
import os.path

class NormaliseRoiAverageValuesInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(argstr="%s", exists=True, mandatory=True,
                   desc="Input image to extract the uptake values")
    in_array = traits.Array(argstr="%s", exists=True, mandatory=True,
                            desc="Array containing the uptake statistics. "+ \
                            "Array order=[Label, mean, std, vol]")
    roi = traits.String(argstr="%s", exists=True, mandatory=True,
                        desc="Name of the roi needed for normalisation")
    cereb_array = traits.Array(argstr="%s", exists=True, mandatory=False,
                               desc="Array containing the grey matter cerebellum "+ \
                               "statistics. Array order=[Label, mean, std, vol]")

    
class NormaliseRoiAverageValuesOutputSpec(TraitedSpec):
    out_csv_file = File(desc="Output array organised as follow: "+ \
        "label index, mean value, std value, roi volume in mm")
    out_file = File(desc="Output array organised as follow: "+ \
        "label index, mean value, std value, roi volume in mm")
    test_roi1=File()
    test_roi2=File()
    test_roi3=File()


class NormaliseRoiAverageValues(BaseInterface):
    
    input_spec = NormaliseRoiAverageValuesInputSpec  
    output_spec = NormaliseRoiAverageValuesOutputSpec
    
    roi1_list=[24,31,76,77,101,102,103,104,105,106,107,108,\
            113,114,119,120,121,122,125,126,133,134,139,140,141,142,143,144,\
            147,148,153,154,155,156,163,164,165,166,167,168,169,170,173,174,\
            175,176,179,180,181,182,185,186,187,188,191,192,195,196,199,200,\
            201,202,203,204,205,206,207,208]
    roi2_list=[24,31,32,33,48,49,76,77,101,102,103,104,105,106,107,108,113,114,\
            117,118,119,120,121,122,123,124,125,126,133,134,139,140,141,142,\
            143,144,147,148,153,154,155,156,163,164,165,166,167,168,169,170,\
            171,172,173,174,175,176,179,180,181,182,185,186,187,188,191,192,\
            195,196,199,200,201,202,203,204,205,206,207,208]
    cereb_list=[39,40,41,42,72,73,74]

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        in_array = self.inputs.in_array
        roi = self.inputs.roi
        cereb_array = self.inputs.cereb_array
        self.normalise_uptake_values(in_file, in_array, roi, cereb_array)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_csv_file'] = self.suvr_file
        outputs['out_file'] = self.norm_file
        return outputs
    
    def normalise_uptake_values(self,
                                in_file,
                                in_array,
                                norm_roi,
                                cereb_array=None):
        _, base, _ = split_filename(in_file)
        self.norm_file=os.path.abspath('norm_'+norm_roi+'_'+base+'.nii.gz')
        self.suvr_file=os.path.abspath('suvr_'+norm_roi+'_'+base+'.csv')
        
        # Create csv file to save the data
        out=open(self.suvr_file,'w')
        out.write('Input Functional image,'+str(in_file)+'\n')
        # Extract the normalisation value
        normalisation_value=0.0
        if norm_roi=='pons':
            i=np.where(in_array[:,0]==35)[0]
            normalisation_value=in_array[i,1]
        elif norm_roi=='cereb':
            total_volume=0.0
            for label in [39,40,41,42,72,73,74]:
                i=np.where(in_array[:,0]==label)[0]
                normalisation_value=normalisation_value+in_array[i,1]*in_array[i,3]
                total_volume=total_volume+in_array[i,3]
            normalisation_value=normalisation_value/total_volume
        elif norm_roi=='gm_cereb' and not cereb_array==None:
            normalisation_value=cereb_array[1,1]
        elif norm_roi=='none':
            normalisation_value=1.0
        # Write down the normalisation value
        out.write('Normalisation,'+str(norm_roi)+',value,'+str(normalisation_value)+'\n')
        out.write('Label index,Initial mean, normalised mean, volume\n')
        # Normalise the input image
        norm_file=niftyseg.BinaryMaths()
        norm_file.inputs.in_file=in_file
        norm_file.inputs.operand_value=np.float(normalisation_value)
        norm_file.inputs.operation='div'   
        norm_file.inputs.out_file=self.norm_file
        norm_file.run()       
        # Normalise all the SUVR
        norm_array=in_array[:,1]/np.float(normalisation_value)
        for i in range(0,len(norm_array)):
            out.write(str(in_array[i,0])+','+str(in_array[i,1])+','+ \
            str(norm_array[i])+','+str(in_array[i,3])+'\n')
        # Extract the large ROI uptake values
        roi1_uptake=0
        roi1_volume=0
        for label in self.roi1_list:
            i=np.where(in_array[:,0]==label)[0]
            roi1_uptake=roi1_uptake+norm_array[i]*in_array[i,3]
            roi1_volume=roi1_volume+in_array[i,3]
        roi1_uptake=np.float(roi1_uptake)/np.float(roi1_volume)
        out.write('region1,,'+str(roi1_uptake)+','+str(roi1_volume)+'\n')
        roi2_uptake=0
        roi2_volume=0
        for label in self.roi2_list:
            i=np.where(in_array[:,0]==label)[0]
            roi2_uptake=roi2_uptake+norm_array[i]*in_array[i,3]
            roi2_volume=roi2_volume+in_array[i,3]
        roi2_uptake=np.float(roi2_uptake)/np.float(roi2_volume)
        out.write('region2,,'+str(roi2_uptake)+','+str(roi2_volume)+'\n')
        out.close()