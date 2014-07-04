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


from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, OutputMultiPath,
                                    isdefined)

from nipype.utils.filemanip import split_filename


def calculate_distance_between_affines(original_image, mask, noise_type, sigma_val):
    # Load the original image
    nib_image = nib.load(original_image)
    data = nib_image.get_data()
    output_data = np.zeros(data.shape)
    mask_data = np.ones(data.shape)
    if isdefined(mask):
        nib_mask = nib.load(mask)
        mask_data = nib_mask.get_data()
    
    data[data<0] = 0
    mask_inds  = mask_data>0
    if noise_type == "gaussian":
        output_data[mask_inds] = ss.norm.rvs(loc=data[mask_inds],scale=sigma_val,size=(mask_inds>0).sum())
    elif noise_type == "rician":
        output_data[mask_inds] = ss.rice.rvs(data[mask_inds],loc=data[mask_inds],scale=sigma_val,size=(mask_inds>0).sum())
    
    nib_output = nib.Nifti1Image(output_data, nib_image.get_affine())
    return nib_output


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

    _suffix = "_noisy"
    input_spec = NoiseAdderInputSpec  
    output_spec = NoiseAdderOutputSpec

    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.nii.gz'
        return outfile
        
    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file) :
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        else:
            outputs['out_file'] = os.path.abspath(self._gen_output_filename(self.inputs.in_file))
        return outputs
    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        noise_type = self.inputs.noise_type
        sigma_val = self.inputs.sigma_val
        out_file = self._list_outputs()['out_file']
        print out_file
        mask_file = self.inputs.mask_file

        nib_output = apply_noise(in_file, mask_file, noise_type, sigma_val)
        nib.save(nib_output, out_file)
        return runtime
        
