# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the Diffusion distortion simulation script
"""

import os
import numpy as np
import nibabel as nib
import os.path as op
import warnings
import numpy.random as random
import scipy.stats as ss
import scipy.linalg as la
import csv as csv


from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, OutputMultiPath,
                                    isdefined)

from nipype.utils.filemanip import split_filename




class CalculateAffineDistancesInputSpec(BaseInterfaceInputSpec):
    transformation1_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')
    transformation2_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')
    
    
class CalculateAffineDistancesOutputSpec(TraitedSpec):
    out_array   = traits.Array(desc='Array of distances between the paired transformations')

class CalculateAffineDistances(BaseInterface):

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
        
