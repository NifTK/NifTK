'''This file creates some workflows for susceptibility distortion correction in diffusion MRI '''

from nipype.interfaces.susceptibility import GenFm, PhaseUnwrap, PmScale
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

def create_fieldmap_susceptibility_workflow(name='susceptibility'):
    input_node =  pe.Node(niu.IdentityInterface(
                fields=['phase_image', 'mag_image', 'etd', 'ped', 'rot']),
                            name='input_node')
    
    # Begin by scaling the phase image
    pm_scale = pe.Node(interface=PmScale(), name = 'pm_scale')
    
    bet = pe.Node(interface=fsl.BET(), name='bet')
    bet.inputs.mask = True
    bet.inputs.no_output = True
    
    pm_unwrap = pe.Node(interface=PhaseUnwrap(), name= 'phase_unwrap')
    
    gen_fm = pe.Node(interface=GenFm(), name='gen_fm')
    
    output_node = pe.Node(niu.IdentityInterface(
                fields=['out_fm', 'out_field']),
                            name='output_node')
    
    pipeline = pe.Workflow('workflow')
    pipeline.connect(input_node, 'phase_image', pm_scale, 'in_pm')
    pipeline.connect(input_node, 'mag_image', bet, 'in_file')
    
    pipeline.connect(bet, 'mask_file', pm_unwrap, 'in_mask')
    pipeline.connect(pm_scale, 'out_pm', pm_unwrap, 'in_fm')
    pipeline.connect(input_node, 'mag_image', pm_unwrap, 'in_mag' )
    
    pipeline.connect(input_node, 'etd', gen_fm, 'in_etd')
    pipeline.connect(input_node, 'rot', gen_fm, 'in_rot')
    pipeline.connect(input_node, 'ped', gen_fm, 'in_ped')
    pipeline.connect(bet, 'mask_file', gen_fm, 'in_mask' )
    pipeline.connect(pm_unwrap, 'out_fm', gen_fm, 'in_ufm')
    
    pipeline.connect(gen_fm, 'out_fm', output_node, 'out_fm')
    pipeline.connect(gen_fm, 'out_field', output_node, 'out_field')
    
    return pipeline



