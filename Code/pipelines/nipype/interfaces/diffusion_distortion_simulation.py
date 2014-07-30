# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the Diffusion distortion simulation script
"""

import os
import math
import numpy as np
import numpy.random as random

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, OutputMultiPath,
                                    isdefined)

import dipy.core.gradients as gradients
from nipype.utils.filemanip import split_filename

def generate_distortion(std_trans,std_rot,std_shear):

    print 'generate a distortion with tr=', std_trans, ', rot=', std_rot, ', shear=', std_shear 
    translations   = [0 + std_trans * random.randn(), \
                      0 + std_trans * random.randn(), \
                      0 + std_trans * random.randn()]
    rotations_a    =  0 + std_rot   * random.randn()
    rotations_b    =  0 + std_rot   * random.randn()
    rotations_g    =  0 + std_rot   * random.randn()
    shearings      =  0 + std_shear * random.randn()

    Mtrans=np.identity(4);
    for j in range(3):
        Mtrans[j,3]=translations[j];
    print 'Mtrans is \n', Mtrans

    Mx=np.identity(4);
    Mx[1,1] = math.cos(rotations_g);
    Mx[2,1] = math.sin(rotations_g);
    Mx[1,2] =-math.sin(rotations_g);
    Mx[2,2] = math.cos(rotations_g);
    
    My=np.identity(4);
    My[0,0] = math.cos(rotations_b);
    My[0,2] = math.sin(rotations_b);
    My[2,0] =-math.sin(rotations_b);
    My[2,2] = math.cos(rotations_b);
    
    Mz=np.identity(4);
    Mz[0,0] = math.cos(rotations_a);
    Mz[1,0] = math.sin(rotations_a);
    Mz[0,1] =-math.sin(rotations_a);
    Mz[1,1] = math.cos(rotations_a);
    
    Mrot = np.dot(np.dot(Mx,My), Mz);
    print 'Mrot is \n', Mrot

    Mshear=np.identity(4);
    Mshear[1,0]=shearings;
    print 'Mshear is \n', Mshear
    
    M = np.dot(np.dot(Mtrans,Mshear), Mrot);

    print 'distortion=\n', M

    return M.tolist()


class DistortionGeneratorInputSpec(BaseInterfaceInputSpec):
    
    bval_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input b-factor file to generate appropriate distortions")
    bvec_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input b-vector file to generate appropriate distortions")
    stddev_translation_val = traits.Float(argstr="%f", desc="Variance for the translation")
    stddev_rotation_val = traits.Float(argstr="%f", desc="Variance for the rotation")
    stddev_shear_val = traits.Float(argstr="%f", desc="Variance for the shear")
    
class DistortionGeneratorOutputSpec(TraitedSpec):
    aff_files   = OutputMultiPath(desc="Output distortion matrices")
    bval_files  = OutputMultiPath(desc="Output bval files")
    bvec_files  = OutputMultiPath(desc="Output bvec files")

class DistortionGenerator(BaseInterface):

    """

    Examples
    --------

    """

    _suffix = "_distortion_"
    input_spec = DistortionGeneratorInputSpec  
    output_spec = DistortionGeneratorOutputSpec

    def _gen_output_filenames(self, bval_file, bvec_file):
        _, base, _ = split_filename(bval_file)
        gtab = gradients.gradient_table(bval_file, bvec_file)
        b0_list = list(gtab.b0s_mask)
        out_aff_filenames = []
        out_bval_filenames = []
        out_bvec_filenames = []
        for i in range(len(b0_list)):
            outbase = base + self._suffix + str(i)
            outaff  = outbase + '.txt'
            outbval = outbase + '.bval'
            outbvec = outbase + '.bvec'
            out_aff_filenames.append(outaff)
            out_bval_filenames.append(outbval)
            out_bvec_filenames.append(outbvec)

        return out_aff_filenames, out_bval_filenames, out_bvec_filenames

    def _run_interface(self, runtime):
        std_trans = self.inputs.stddev_translation_val
        std_rot = self.inputs.stddev_rotation_val
        std_shear = self.inputs.stddev_shear_val        
        bval_file = self.inputs.bval_file
        bvec_file = self.inputs.bvec_file
        
        gtab = gradients.gradient_table(bval_file, bvec_file)
        b0_list = list(gtab.b0s_mask)
        
        out_aff_filenames, out_bval_filenames, out_bvec_filenames = self._gen_output_filenames(bval_file, bvec_file)
        
        for i in range(len(b0_list)):
            is_b0 = b0_list[i]
            if is_b0 == True:
                shear = 0
            else:
                shear = std_shear
            distortion = generate_distortion(std_trans,std_rot,shear)
            outaff = out_aff_filenames[i]
            f=open(outaff, 'w+')
            for j in range(len(distortion)):
                l = str(distortion[j])
                l = l.replace ('[', '')
                l = l.replace (']', '')
                l = l.replace (',', ' ')
                f.write("%s\n" % l)
            f.close

            outbval = out_bval_filenames[i]
            f=open(outbval, 'w+')
            f.write("%f" % gtab.bvals[i])
            f.close

            outbvec = out_bvec_filenames[i]
            f=open(outbvec, 'w+')
            for j in range(3):
               f.write("%f\n" % gtab.bvecs[i][j])
            f.close
            
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_aff_filenames, out_bval_filenames, out_bvec_filenames = self._gen_output_filenames(self.inputs.bval_file, self.inputs.bvec_file)
        aff_files = []
        bval_files = []
        bvec_files = []
        for i in range(len(out_aff_filenames)):
            aff_files.append(os.path.abspath(out_aff_filenames[i]))
            bval_files.append(os.path.abspath(out_bval_filenames[i]))
            bvec_files.append(os.path.abspath(out_bvec_filenames[i]))
        outputs['aff_files']  = aff_files
        outputs['bval_files'] = bval_files
        outputs['bvec_files'] = bvec_files
        return outputs
        
