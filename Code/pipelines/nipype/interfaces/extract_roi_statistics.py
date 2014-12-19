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
        in_img=nib.load(in_file)
        img_data=in_img.get_data()
        roi_data=nib.load(roi_file).get_data()
        # Get the voxel dimention
        vox_dim=np.product(in_img.get_header().get_zooms())
        
        im_shape = img_data.shape
        roi_shape = roi_data.shape
                
        self.number_of_ims = 1
        self.number_of_roi_ims = 1
        if len(im_shape) > 3:
            self.number_of_ims = im_shape[3]
        if len(roi_shape) > 3:
            self.number_of_roi_ims = roi_shape[3]
        
        # Create an array to store mean uptakes and volumes
        unique_values=np.unique(roi_data)
        unique_values_number=len(unique_values);
        stats_array= np.zeros((unique_values_number*self.number_of_ims,4))
        
        for im_index in range(0, self.number_of_ims):
            if self.number_of_ims > 1:
                self.image = img_data[:,:,:,im_index]
            else:
                self.image = img_data[:,:,:]
            
            if self.number_of_roi_ims > 1:
                self.roi = roi_data[:,:,:,im_index]
            else:
                self.roi = roi_data[:,:,:]
            
            index=0
            for i in unique_values:
                mask = (self.roi==i)
                values = self.image[mask]
                stats_array[index+im_index*unique_values_number,0]=i
                stats_array[index+im_index*unique_values_number,1]=np.mean(values)
                stats_array[index+im_index*unique_values_number,2]=np.std(values)
                stats_array[index+im_index*unique_values_number,3]=np.multiply(np.sum(mask),vox_dim)
                index=index+1
        return stats_array