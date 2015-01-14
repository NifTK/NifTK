"""
   Interface for niftk statistical tools
"""

from niftk.base import NIFTKCommand, NIFTKCommandInputSpec, getNiftkPath
from nipype.interfaces.base import (TraitedSpec, File, traits)

import os
import numpy as np
import nibabel as nib



class ComputeDiceScoreInputSpec(NIFTKCommandInputSpec):
    
    in_file1 = File(argstr="%s", exists=True, mandatory=True,
                        desc="First roi image")
    in_file2 = File(argstr="%s", exists=True, mandatory=True,
                        desc="Second roi image")
    
class ComputeDiceScoreOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='Output array containing the dice score.\n'+
        'The first value is the label index and the second is the Dice score')


class ComputeDiceScore(NIFTKCommand):
    
    input_spec = ComputeDiceScoreInputSpec  
    output_spec = ComputeDiceScoreOutputSpec

    def _run_interface(self, runtime):
        roi_file1 = self.inputs.in_file1
        roi_file2 = self.inputs.in_file2
        self.out_dict=self.compute_dice_score(roi_file1, roi_file2)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dict'] = self.out_dict
        return outputs
        

    def compute_dice_score(self,
                           roi_file1,
                           roi_file2):
                               
        # Reorient the input images to remove any nibabel reading error
        reorient1=fsl.Reorient2Std()
        reorient2=fsl.Reorient2Std()
        reorient1.inputs.in_file=roi_file1
        reorient2.inputs.in_file=roi_file2
        reoriented_filename1=reorient1.run().outputs.out_file
        reoriented_filename2=reorient2.run().outputs.out_file
        
        # Read the input images
        in_img1=nib.load(reoriented_filename1).get_data()
        in_img2=nib.load(reoriented_filename2).get_data()
        
        # Get the min and max label values
        min_label_value = np.int32(np.min([np.min(in_img1), np.min(in_img2)]))
        max_label_value = np.int32(np.max([np.max(in_img1), np.max(in_img2)]))

        # Iterate over all label values
        out_dict=dict()
        for l in range(min_label_value,max_label_value+1):
            mask1 = in_img1==l
            mask2 = in_img2==l
            mask3 = (in_img1==l) + (in_img2==l)
            if np.sum(mask1)+np.sum(mask2) != 0:
                out_dict[l]=0.5*(np.sum(mask1)+np.sum(mask2))/np.sum(mask3)
            
        return out_dict






class CalculateAffineDistancesInputSpec(NIFTKCommandInputSpec):
    transformation1_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')
    transformation2_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')
    
    
class CalculateAffineDistancesOutputSpec(TraitedSpec):
    out_array   = traits.Array(desc='Array of distances between the paired transformations')

class CalculateAffineDistances(NIFTKCommand):

    """

    Examples
    --------

    """
    input_spec = CalculateAffineDistancesInputSpec  
    output_spec = CalculateAffineDistancesOutputSpec

    def _run_interface(self, runtime):
        transformation1_list = self.inputs.transformation1_list
        transformation2_list = self.inputs.transformation2_list
        self.distances=self.calculate_distance_between_affines(transformation1_list, transformation2_list)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_array'] = self.distances
        return outputs
        

    def read_file_to_matrix(self, file_name):
        mat = np.zeros((4,4))
        with open(file_name, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ',skipinitialspace=True)
                row_ind = 0
                for row in reader:
                    #print row
                    for col_ind in range(4):
                        if row[col_ind] == ' ':
                            row.pop(col_ind)
                        mat[row_ind,col_ind] = row[col_ind]
                        #print row[col_ind]
                    row_ind = row_ind + 1
        return mat

    def calculate_distance_between_affines(self,list1_aff, list2_aff):
        distances = np.zeros((len(list1_aff),1))    
        
        for i in range(len(list1_aff)):
            # Read affine matrices
            file1 = list1_aff[i]
            file2 = list2_aff[i]
            mat1 = self.read_file_to_matrix(file1)
            mat2 = self.read_file_to_matrix(file2)
            log_mat1 = la.logm(mat1)
            log_mat2 = la.logm(mat2)
            distances[i,0] = ((log_mat2 - log_mat1)*(log_mat2 - log_mat1)).sum()
            
            
        return distances
        




class ExtractRoiStatisticsInputSpec(NIFTKCommandInputSpec):
    
    in_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input image to extract the statistics from")
    roi_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input image that contains the different roi")
    in_label = traits.List(traits.BaseInt,
                           desc = "Label(s) to extract")

class ExtractRoiStatisticsOutputSpec(TraitedSpec):
    out_array   = traits.Array(desc="Output array organised as follow: "+ \
        "label index, mean value, std value, roi volume in mm")


class ExtractRoiStatistics(NIFTKCommand):

    """

    Examples
    --------

    """

    input_spec = ExtractRoiStatisticsInputSpec  
    output_spec = ExtractRoiStatisticsOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        roi_file = self.inputs.roi_file
        labels = self.inputs.in_label
        self.stats=self.extract_roi_statistics(in_file, roi_file, labels)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_array'] = self.stats
        return outputs
        

    def extract_roi_statistics(self,
                               in_file,
                               roi_file,
                               labels):
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
        unique_values = 0
        unique_values_number = 0

        if len(labels) > 0:
            unique_values = labels
        else:
            unique_values = np.unique(roi_data)

        unique_values_number = len(unique_values)

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

