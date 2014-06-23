# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the EPI distortion generation script
"""

import os
import numpy as np
from nibabel import load
import os.path as op
import warnings
import math
import random


from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory, traits, OutputMultiPath,
                                    isdefined)

import dipy.core.gradients as gradients
from nipype.utils.filemanip import split_filename


def generatedistortion(Vtrans,Vrot,shear):

    print 'generate a distortion with tr=', Vtrans, ', rot=', Vrot, ', shear=', shear 
    translations   = [0 + (Vtrans/3)  ** (0.5) * random.random(), \
                      0 + (Vtrans/3)  ** (0.5) * random.random(), \
                      0 + (Vtrans/3)  ** (0.5) * random.random()]
    rotations_a    =  0 +       Vrot  ** (0.5) * random.random()
    rotations_b    =  0 +       Vrot  ** (0.5) * random.random()
    rotations_g    =  0 +       Vrot  ** (0.5) * random.random()
    shearings      =  0 +       shear ** (0.5) * random.random()

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
    
    in_bval_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input b-factor file to generate appropriate distortions")
    in_bvec_file = File(argstr="%s", exists=True, mandatory=True,
                        desc="Input b-vector file to generate appropriate distortions")
    in_var_translation = traits.Float(argstr="%f", desc="Variance for the translation")
    in_var_rotation = traits.Float(argstr="%f", desc="Variance for the rotation")
    in_var_shear = traits.Float(argstr="%f", desc="Variance for the shear")
    
class DistortionGeneratorOutputSpec(TraitedSpec):
    aff_files   = OutputMultiPath(desc="Output distortion matrices")

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
        outfilenames = []
        for i in range(len(b0_list)):
            outfile = base + self._suffix + str(i) + '.txt'
            outfilenames.append(outfile)
        return outfilenames

    def _run_interface(self, runtime):
        Vtrans = self.inputs.in_var_translation
        Vrot = self.inputs.in_var_rotation
        Vshear = self.inputs.in_var_shear        
        bvalfile = self.inputs.in_bval_file
        bvecfile = self.inputs.in_bvec_file
        
        gtab = gradients.gradient_table(bvalfile, bvecfile)
        b0_list = list(gtab.b0s_mask)

        outfilenames = self._gen_output_filenames(bvalfile, bvecfile)

        for i in range(len(b0_list)):
            is_b0 = b0_list[i]
            if is_b0 == True:
                shear = 0
            else:
                shear = Vshear
            distortion = generatedistortion(Vtrans,Vrot,shear)
            outfile = outfilenames[i]
            f=open(outfile, 'w+')
            for i in range(len(distortion)-1):
                l = str(distortion[i])
                l = l.replace ('[', '')
                l = l.replace (']', '')
                l = l.replace (',', ' ')
                f.write("%s\n" % l)
            f.close
            
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outfilenames = self._gen_output_filenames(self.inputs.in_bval_file, self.inputs.in_bvec_file)
        aff_files = []
        for item in outfilenames:
            aff_files.append(os.path.abspath(item))
        outputs['aff_files'] = aff_files
        return outputs
        
