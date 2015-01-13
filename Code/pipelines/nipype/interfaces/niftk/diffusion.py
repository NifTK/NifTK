"""
   Interface for niftk filter tools
"""

from niftk.base import NIFTKCommand, NIFTKCommandInputSpec, getNiftkPath
from nipype.interfaces.base import (TraitedSpec, File, traits)


import os
import math
import numpy as np
import numpy.random as random

import sys
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from string import Template

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


class DistortionGeneratorInputSpec(NIFTKCommandInputSpec):
    
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

class DistortionGenerator(NIFTKCommand):

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
        







class GradwarpCorrectionInputSpec(NIFTKCommandInputSpec):
    
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                desc="Input target image filename")
    
    coeff_file = File(argstr="-c %s", exists=True, mandatory=True,
                desc="Spherical harmonics coefficient filename")
    
    scanner_type = traits.String(argstr="-t %s",
                desc="Scanner type: siemens or ge. siemens by default.")
    
    radius = traits.BaseFloat(argstr="-r %f",
                desc="Gradwarp radius in meter.")
    
    offset_x = traits.BaseFloat(argstr="-off_x %f",
                desc="Scanner offset along the x axis in mm.")
    
    offset_y = traits.BaseFloat(argstr="-off_y %f",
                desc="Scanner offset along the y axis in mm.")
    
    offset_z = traits.BaseFloat(argstr="-off_z %f",
                desc="Scanner offset along the z axis in mm.")

    out_file = File(argstr="-o %s", 
                    desc="output deformation field image",
                    name_source = ['in_file'],
                    name_template = '%s_unwarp_field')

class GradwarpCorrectionOutputSpec(TraitedSpec):

    out_file = File(exists=True, desc="output deformation field image")

class GradwarpCorrection(NIFTKCommandInputSpec):

    _cmd = "gradient_unwarp"

    input_spec = GradwarpCorrectionInputSpec  
    output_spec = GradwarpCorrectionOutputSpec









# A custom function for getting specific noddi path
def getNoddiPath(cmd):
    try:    
        specific_dir=os.environ['NODDIDIR']
        cmd=os.path.join(specific_dir,cmd)
        return cmd
    except KeyError:                
        return cmd

class NoddiInputSpec( NIFTKCommandInputSpec):

    in_dwis = File(exists=True, 
                   desc='The input 4D DWIs image file',
                   mandatory=True)
    in_mask = File(exists=True, 
                   desc='The input mask image file',
                   mandatory=True)
    in_bvals = File(exists=True, 
                   desc='The input bval file',
                   mandatory=True)
    in_bvecs = File(exists=True, 
                   desc='The input bvec file',
                   mandatory=True)
    in_fname = traits.Str('noddi',
                          desc='The output fname to use',
                          usedefault=True)

class NoddiOutputSpec( TraitedSpec):

    out_neural_density = File(genfile=True, desc='The output neural density image file')
    out_orientation_dispersion_index = File(genfile=True, desc='The output orientation dispersion index image file')
    out_csf_volume_fraction = File(genfile=True, desc='The output csf volume fraction image file')
    out_objective_function = File(genfile=True, desc='The output objective function image file')
    out_kappa_concentration = File(genfile=True, desc='The output Kappa concentration image file')
    out_error = File(genfile=True, desc='The output estimation error image file')
    out_fibre_orientations_x = File(genfile=True, desc='The output fibre orientation (x) image file')
    out_fibre_orientations_y = File(genfile=True, desc='The output fibre orientation (y) image file')
    out_fibre_orientations_z = File(genfile=True, desc='The output fibre orientation (z) image file')

    matlab_output = traits.Str()

class Noddi( NIFTKCommand):

    """ NODDI estimation interface for the MATLAB toolbox (http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab)

    Returns
    -------

    output files : 
    out_neural_density  ::  The output neural density image file
    out_orientation_dispersion_index  ::  The output orientation dispersion index image file
    out_csf_volume_fraction  ::  The output csf volume fraction image file
    out_objective_function  ::  The output objective function image file
    out_kappa_concentration  ::  The output Kappa concentration image file
    out_error  ::  The output estimation error image file
    out_fibre_orientations_x  ::  The output fibre orientation (x) image file
    out_fibre_orientations_y  ::  The output fibre orientation (y) image file
    out_fibre_orientations_z  ::  The output fibre orientation (z) image file
    
    Examples
    --------

    >>> n = Noddi()
    >>> n.inputs.in_dwis = subject1_dwis.nii.gz
    >>> n.inputs.in_mask = subject1_mask.nii.gz
    >>> n.inputs.in_bvals = subject1_bvals.bval
    >>> n.inputs.in_bvecs = subject1_bvecs.bvec
    >>> n.inputs.in_fname = 'subject1'
    >>> out = n.run()
    >>> print out.outputs
    """

    input_spec = NoddiInputSpec
    output_spec = NoddiOutputSpec

    def _my_script(self):
        """This is where you implement your script"""

        matlab_scriptname = getNoddiPath('noddi_fitting')

        d = dict(in_dwis=self.inputs.in_dwis,
                 in_mask=self.inputs.in_mask,
                 in_bvals=self.inputs.in_bvals,
                 in_bvecs=self.inputs.in_bvecs,
                 in_fname=self.inputs.in_fname,
                 script_name = matlab_scriptname)

        #this is your MATLAB code template
        script = Template("""
        [~,~,~,~,~,~,~] = script_name(in_dwis, in_mask, in_bvals, in_bvecs, in_fname);
        """).substitute(d)

        return script

    def _run_interface(self, runtime):
        """This is where you implement your script"""

        d = dict(in_dwis=self.inputs.in_dwis,
                 in_mask=self.inputs.in_mask,
                 in_bvals=self.inputs.in_bvals,
                 in_bvecs=self.inputs.in_bvecs,
                 in_fname=self.inputs.in_fname)

        #this is your MATLAB code template
        script = Template("""
        in_dwis = '$in_dwis';
        in_mask = '$in_mask';
        in_bvals = '$in_bvals';
        in_bvecs = '$in_bvecs';
        in_fname = '$in_fname';
        [~,~,~,~,~,~,~] = noddi_fitting(in_dwis, in_mask, in_bvals, in_bvecs, in_fname);
        exit;
        """).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        basename = self.inputs.fname
        outputs['out_neural_density'] = os.path.join(os.getcwd(), basename + '_ficvf.nii')
        outputs['out_orientation_dispersion_index'] = os.path.join(os.getcwd(), basename + '_odi.nii')
        outputs['out_csf_volume_fraction'] = os.path.join(os.getcwd(), basename + '_fiso.nii')
        outputs['out_objective_function'] = os.path.join(os.getcwd(), basename + '_fmin.nii')
        outputs['out_kappa_concentration'] = os.path.join(os.getcwd(), basename + '_kappa.nii')
        outputs['out_error'] = os.path.join(os.getcwd(), basename + '_error_code.nii')
        outputs['out_fibre_orientations_x'] = os.path.join(os.getcwd(), basename + '_fibredirs_xvec.nii')
        outputs['out_fibre_orientations_y'] = os.path.join(os.getcwd(), basename + '_fibredirs_yvec.nii')
        outputs['out_fibre_orientations_z'] = os.path.join(os.getcwd(), basename + '_fibredirs_zvec.nii')
        return outputs
