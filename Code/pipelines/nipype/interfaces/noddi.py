# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for the noddi_fitting.m script
"""

import os
from nipype.interfaces.base import File, traits
from nipype.interfaces.matlab import MatlabCommand, MatlabInputSpec
from string import Template

class NoddiInputSpec( MatlabInputSpec):

    in_dwis = 
    in_mask = 
    in_bvals
    in_bvecs
    in_fname

class NoddiOutputSpec( MatlabInputSpec):
    out_neural_density = File(exists=True)
    out_orientation_dispersion_index = File(exists=True)
    out_csf_volume_fraction = File(exists=True)
    out_objective_function = File(exists=True)
    out_kappa_concentration = File(exists=True)
    out_error = File(exists=True)
    out_fibre_orientations_x = File(exists=True)
    out_fibre_orientations_y = File(exists=True)
    out_fibre_orientations_z = File(exists=True)
    
    

class Noddi( MatlabCommand):
    """ Basic Hello World that displays Hello <name> in MATLAB

    Returns
    -------

    matlab_output : capture of matlab output which may be
                    parsed by user to get computation results

    Examples
    --------

    >>> n = Noddi()
    >>> n.inputs.
    >>> out = hello.run()
    >>> print out.outputs
    """
    input_spec = NoddiInputSpec
    output_spec = NoddiOutputSpec

    def _my_script(self):
        """This is where you implement your script"""

        d = dict(in_dwis=self.inputs.in_dwis,
                 in_mask=self.inputs.in_mask,
                 in_bvals=self.inputs.in_bvals,
                 in_bvecs=self.inputs.in_bvecs,
                 in_fname=self.inputs.in_fname,
                 out_neural_density=self.inputs.out_neural_density,
                 out_orientation_dispersion_index=self.inputs.out_orientation_dispersion_index,
                 out_csf_volume_fraction=self.inputs.out_csf_volume_fraction,
                 out_objective_function=self.inputs.out_objective_function,
                 out_kappa_concentration=self.inputs.out_kappa_concentration,
                 out_error=self.inputs.out_error,
                 out_fibre_orientations_x=self.inputs.out_fibre_orientations_x,
                 out_fibre_orientations_y=self.inputs.out_fibre_orientations_y,
                 out_fibre_orientations_z=self.inputs.out_fibre_orientations_z)
        #this is your MATLAB code template
        script = Template("""
        [neural_density, orientation_dispersion_index, csf_volume_fraction, objective_function, kappa_concentration, error, fibre_orientations] = noddi_fitting(dwis, mask, bvals, bvecs, fname);
        exit;
        """).substitute(d)
        return script


    def run(self, **inputs):
        ## inject your script
        self.inputs.script =  self._my_script()
        results = super(MatlabCommand, self).run( **inputs)
        stdout = results.runtime.stdout
        # attach stdout to outputs to access matlab results
        results.outputs.matlab_output = stdout
        return results


    def _list_outputs(self):
        outputs = self._outputs().get()
        return outputs
