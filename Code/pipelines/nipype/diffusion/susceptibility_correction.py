'''This file creates some workflows for susceptibility distortion correction in diffusion MRI '''

from nipype.interfaces.susceptibility import GenFm, PhaseUnwrap, PmScale
from nipype.interfaces.niftyreg import RegAladin, RegTransform, RegResample, RegF3D, RegJacobian
from nipype.interfaces.niftyseg import BinaryMaths
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

def create_fieldmap_susceptibility_workflow(name='susceptibility', mask_exists = False, reg_to_t1 = False):
    """Creates a workflow that perform EPI distortion correction using field
    maps and possibly T1 images. 
    
    Example
    -------

    >>> susceptibility_correction = create_fieldmap_susceptibility_workflow(name='susceptibility_workflow')
    >>> susceptibility_correction.inputs.input_node.etd = 2.46
    >>> susceptibility_correction.inputs.input_node.rot = 34.56
    >>> susceptibility_correction.inputs.input_node.ped = '-y'


    Inputs::

        input_node.epi_image - The EPI distorted diffusion image
        input_node.phase_image - The phase difference image of the fieldmap
        input_node.mag_image - The magnitude of the fieldmap (must be single image)
        input_node.etd - The echo time difference in msec (generally 2.46 msec for siemens scanners)
        input_node.rot - The read out time in msec (34.56 msec for standard DRC acquisitions)
        input_node.ped - The phase encode direction (-y for standard DRC acquisition)
        input_node.mask_image - Mask image in the epi_image space (only used where mask_exists = True)
        input_node.in_t1_file - A downsampled t1 image in the epi image space (only used when reg_to_t1 = True)
        
        
    Outputs::

        output_node.out_field - The deformation field that undoes the magnetic susceptibility
        distortion in the space of the epi image
        output_node.out_jac - The thresholded (to remove negative Jacobians) jacobian map of the 
        corrective field 
        output_node.out_epi - The distortion corrected, and divided by the thresholded Jacobian, epi image


    Optional arguments::
        mask_exists -  which requires a mask to be provided in the diffusion image
    space, otherwise bet is run. (default is False)
        TODO: reg_to_t1 - include a step to non-linearly register the field map corrected
        image to the T1 space to refine the correction. 


    """
    input_node =  pe.Node(niu.IdentityInterface(
                fields=['epi_image','phase_image', 'mag_image', 'etd', 'ped', 'rot','mask_image','t1']),
                            name='input_node')
    
    # create nodes to estimate the defomrartion field from the field map images
    pm_scale = pe.Node(interface=PmScale(), name = 'pm_scale')
    pm_unwrap = pe.Node(interface=PhaseUnwrap(), name= 'phase_unwrap')
    gen_fm = pe.Node(interface=GenFm(), name='gen_fm')

    # Create nodes to register the field map defomation field       
    reg_fm_to_b0 = pe.Node(interface=RegAladin(), name='reg_fm_to_b0')
    reg_fm_to_b0.inputs.rig_only_flag = True
    invert_aff = pe.Node(interface=RegTransform(), name='invert_fm_to_b0' )
    resample_mask = pe.Node(interface=RegResample(), name='resample_mask', interp='NN')
    resample_epi = pe.Node(interface=RegResample(), name='resample_epi')
    transform_def_to_b0 = pe.Node(interface=RegTransform(), name='transform_fm_def_in_b0')
    reg_jacobian = pe.Node(interface=RegJacobian(), name='calc_transform_jac')
    thr_jac = pe.Node(interface=BinaryMaths(operation='thr', operand_value = 0.1), name='thr_jac')
    div_jac = pe.Node(interface=BinaryMaths(operation= 'div'), name='div_jac')
        
    output_node = pe.Node(niu.IdentityInterface(
                fields=['out_field','out_epi', 'out_jac']),
                            name='output_node')
    
    pipeline = pe.Workflow(name=name)
    pipeline.base_output_dir=name
    
    # Need to register the magnitude image to the b0 to have the deformation in
    # the b0 space and to propagate the mask (if available)
    pipeline.connect(input_node, 'mag_image', reg_fm_to_b0, 'flo_file')
    pipeline.connect(input_node, 'epi_image', reg_fm_to_b0, 'ref_file')
    
    # we also need the inverse (if we're propagating the mask)
    if mask_exists:
        pipeline.connect(reg_fm_to_b0, 'aff_file', invert_aff, 'inv_aff_input' )
        pipeline.connect(invert_aff, 'out_file', resample_mask, 'trans_file')
        pipeline.connect(input_node, 'mask_image', resample_mask, 'flo_file')
        pipeline.connect(input_node, 'mag_image', resample_mask, 'ref_file')
        pipeline.connect(resample_mask, 'res_file', pm_unwrap, 'in_mask')
        pipeline.connect(resample_mask, 'res_file', gen_fm, 'in_mask' )
    else:
        # create a bet node, in case we don't have a mask   
        bet = pe.Node(interface=fsl.BET(mask=True, no_output=True), name='bet')
        pipeline.connect(input_node, 'mag_image', bet, 'in_file')
        pipeline.connect(bet, 'mask_file', gen_fm, 'in_mask' )
        pipeline.connect(bet, 'mask_file', pm_unwrap, 'in_mask') 


    # Unwrap the phase image
    pipeline.connect(input_node, 'phase_image', pm_scale, 'in_pm')
    pipeline.connect(pm_scale, 'out_pm', pm_unwrap, 'in_fm')
    pipeline.connect(input_node, 'mag_image', pm_unwrap, 'in_mag' )
    
    # Generate the deformation feld from the fieldmap
    pipeline.connect(input_node, 'epi_image', gen_fm, 'in_epi')
    pipeline.connect(input_node, 'etd', gen_fm, 'in_etd')
    pipeline.connect(input_node, 'rot', gen_fm, 'in_rot')
    pipeline.connect(input_node, 'ped', gen_fm, 'in_ped')  
    pipeline.connect(pm_unwrap, 'out_fm', gen_fm, 'in_ufm')
    
     # Finally, we need to resample the deformation field in the averageb0
    pipeline.connect(gen_fm, 'out_field', transform_def_to_b0, 'comp_input2')
    pipeline.connect(reg_fm_to_b0, 'aff_file',transform_def_to_b0, 'comp_input')
    pipeline.connect(input_node, 'mag_image', transform_def_to_b0,'ref2_file')
    pipeline.connect(input_node, 'epi_image', transform_def_to_b0,'ref1_file')
    
    # Resample the epi image using the new deformation
    pipeline.connect(input_node, 'epi_image',resample_epi,'flo_file')
    pipeline.connect(input_node, 'epi_image', resample_epi,'ref_file')
    pipeline.connect(transform_def_to_b0, 'out_file', resample_epi,'trans_file')
    
    if reg_to_t1:

        reg_f3d = pe.Node(interface = RegF3D(**{'nox_flag' : True, 'noz_flag' : True}),
                          name='reg_refine_fm_correction')
        reg_f3d.inputs.lncc_val  = 4
        reg_f3d.inputs.maxit_val = 100
        reg_f3d.inputs.be_val    = 0.05        

        comp_def = pe.Node(interface = RegTransform(),
                           name = 'comp_def')
        resample_epi_2 = pe.Node(interface=RegResample(), name='resample_epi_2')


        pipeline.connect(input_node, 't1', reg_f3d, 'ref_file')
        pipeline.connect(resample_epi, 'res_file', reg_f3d, 'flo_file')
        if mask_exists == True:
            pipeline.connect(input_node, 'mask_image', reg_f3d, 'rmask_file')
            
        pipeline.connect(reg_f3d, 'cpp_file', comp_def, 'comp_input')
        pipeline.connect(transform_def_to_b0, 'out_file', comp_def, 'comp_input2')
        pipeline.connect(input_node, 't1', comp_def, 'ref1_file')

        pipeline.connect(input_node, 'epi_image',resample_epi_2,'flo_file')
        pipeline.connect(input_node, 'epi_image', resample_epi_2,'ref_file')
        pipeline.connect(comp_def, 'out_file', resample_epi_2,'trans_file')

        pipeline.connect(comp_def, 'out_file', reg_jacobian, 'trans_file')
        pipeline.connect(comp_def, 'out_file', output_node, 'out_field')
        pipeline.connect(resample_epi_2, 'res_file', div_jac, 'in_file')

    else:
        pipeline.connect(transform_def_to_b0, 'out_file', output_node, 'out_field')
        pipeline.connect(transform_def_to_b0, 'out_file', reg_jacobian, 'trans_file')
        pipeline.connect(resample_epi, 'res_file', div_jac, 'in_file')    

    # Measure the Jacobian determinant of the transformation    
    pipeline.connect(reg_jacobian, 'jac_det_file', thr_jac, 'in_file')
    
    # Divide the resampled epi image by the Jacobian image
    pipeline.connect(thr_jac, 'out_file', div_jac, 'operand_file')
    
    # Fill out the information in the output node
    pipeline.connect(div_jac, 'out_file', output_node, 'out_epi')
    pipeline.connect(thr_jac, 'out_file', output_node, 'out_jac')   
    
    return pipeline



