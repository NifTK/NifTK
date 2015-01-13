# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Interface for the noddi_fitting.m script. NODDI estimation interface for the MATLAB toolbox (http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab)
"""

import os, sys
from nipype.interfaces.base import File, traits
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from string import Template

from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File

# A custom function for getting specific noddi path
def getNoddiPath(cmd):
    try:    
        specific_dir=os.environ['NODDIDIR']
        cmd=os.path.join(specific_dir,cmd)
        return cmd
    except KeyError:                
        return cmd

class NoddiInputSpec( BaseInterfaceInputSpec):

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

class Noddi( BaseInterface):
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
