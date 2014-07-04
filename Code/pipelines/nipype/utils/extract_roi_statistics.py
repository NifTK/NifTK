#!/usr/bin/env python

import numpy as np
import nibabel as nib

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, traits)


class ExtractRoiStatisticsInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input image to extract the statistics from")
    roi_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input image that contains the different roi")
    
class ExtractRoiStatisticsOutputSpec(TraitedSpec):
    out_array   = traits.Array(desc="Output array organised as follow: "+ \
        "label index, mean value, std value, roi volume in mm")


class ExtractRoiStatistics(BaseInterface):

    """

    Examples
    --------

    """

    input_spec = ExtractRoiStatisticsInputSpec  
    output_spec = ExtractRoiStatisticsOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        roi_file = self.inputs.roi_file
        self.stats=self.extract_roi_statistics(in_file, roi_file)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_array'] = self.stats
        return outputs
        

    def extract_roi_statistics(self,
                               in_file,
                               roi_file):
        # Read the input images
        pet_img=nib.load(in_file)
        pet_data=pet_img.get_data()
        roi_data=nib.load(roi_file).get_data()
        # Get the voxel dimention
        vox_dim=np.product(pet_img.get_header().get_zooms())
        # Create an array to store mean uptakes and volumes
        unique_values=np.unique(roi_data)
        unique_values_number=len(unique_values);
        stats_array= np.zeros((unique_values_number,4))
        index=0
        for i in unique_values:
            mask = (roi_data==i)
            values = pet_data[mask]
            stats_array[index,0]=i
            stats_array[index,1]=np.mean(values)
            stats_array[index,2]=np.std(values)
            stats_array[index,3]=np.multiply(np.sum(mask),vox_dim)
            index=index+1
        return stats_array