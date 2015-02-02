# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The niftk module provides classes for interfacing with the `NIFTK TOOLS
<http://cmictig.cs.ucl.ac.uk>`_ command line tools.

Top-level namespace for Niftk.
"""
# import base, asl, diffusion, filters, fmri, functional, gif, io, neuromorphometricslabels, operations, qc, registration, stats, utils

from .base import ( NIFTKCommand, Info, no_niftk )
from .asl import ( create_asl_processing_workflow )
from .diffusion import ( DistortionGenerator, GradwarpCorrection, Noddi, create_fieldmap_susceptibility_workflow, create_diffusion_mri_processing_workflow )
from .filters import ( N4BiasCorrection )
from .fmri import ( RestingStatefMRIPreprocess )
from .functional import ( ExtractSide, NormaliseRoiAverageValues, create_reg_avg_value_pipeline )
from .gif import ( create_niftyseg_gif_propagation_pipeline_simple, create_niftyseg_gif_propagation_pipeline, create_seg_gif_template_database_workflow_1, create_seg_gif_template_database_workflow_2 )
from .io import ( Midas2Nii, Midas2Dicom, GetAxisOrientation )
from .operations import ( CropImage ) 
from .qc import ( InterSliceCorrelationPlot, MatrixRotationPlot )
from .registration import ( create_linear_coregistration_workflow, create_nonlinear_coregistration_workflow, create_atlas )
from .stats import ( ComputeDiceScore, CalculateAffineDistances )
from .utils import ( WriteArrayToCsv )
