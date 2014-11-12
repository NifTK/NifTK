"""
    Simple interface to extract one half of the input image
"""
import os
import os.path
import nibabel              as nib
import numpy                as np

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, traits)
from nipype.utils.filemanip import split_filename

class ExtractSideInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(argstr="%s", exists=True, mandatory=True, \
                desc="Input target image filename")
                
    in_dict = traits.Dict(argstr="%s", mandatory=True, \
                desc='Dictionary containing the axis orientation information. Output of the GetAxisOrientation interface')
                
    in_side = traits.Enum('left', 'right', argstr='%s', mandatory=True, \
                          desc='side to extract')

class ExtractSideOutputSpec(TraitedSpec):

    out_file = traits.File(desc='Image that contains the requested half.')

class ExtractSide(BaseInterface):

    input_spec = ExtractSideInputSpec  
    output_spec = ExtractSideOutputSpec
    
    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + '_halved.nii.gz'
        return os.path.abspath(outfile)
        
    def _run_interface(self, runtime):
        input_file = self.inputs.in_file
        input_dict = self.inputs.in_dict
        input_side = self.inputs.in_side
        self.out_file=self.extract_side(input_file, input_dict, input_side)
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs
        
    def extract_side(self,
                     image_file,
                     dictionary,
                     side_info):
        # Get the input image dimension
        ori_image = nib.load(image_file)
        ori_data = ori_image.get_data()
        ori_shape = np.array(ori_data.shape)
        
        # Check which axis has to be split
        axis_to_split=int()
        for axis in ['i', 'j', 'k']:
            if 'Left-to-Right' in dictionary[axis] or 'Right-to-Left' in dictionary[axis]:
                if axis=='i':
                    axis_to_split=0
                elif axis=='j':
                    axis_to_split=1
                elif axis=='k':
                    axis_to_split=2
        
        # Define the new shape and data
        new_shape=np.array(ori_shape)
        new_shape[axis_to_split]=round(new_shape[axis_to_split]*0.6)
        new_data=np.zeros(new_shape)
        if dictionary['i']=='Left-to-Right' or dictionary['j']=='Left-to-Right' or dictionary['k']=='Left-to-Right':
            if side_info == 'left':
                new_data=ori_data[0:new_shape[0],0:new_shape[1],0:new_shape[2]]
            elif side_info == 'right':
                new_data=ori_data[ori_shape[0]-new_shape[0]:-1,ori_shape[1]-new_shape[1]:-1,ori_shape[2]-new_shape[2]:-1]
        else: # Right-to-Left
            if side_info == 'left':
                new_data=ori_data[ori_shape[0]-new_shape[0]:-1,ori_shape[1]-new_shape[1]:-1,ori_shape[2]-new_shape[2]:-1]
            elif side_info == 'right':
                new_data=ori_data[0:new_shape[0],0:new_shape[1],0:new_shape[2]]
        # Create the new image
        out_file = self._gen_output_filename(image_file)
        new_image = nib.Nifti1Image(new_data, ori_image.get_affine())
        nib.save(new_image, out_file)
                     
        return out_file

