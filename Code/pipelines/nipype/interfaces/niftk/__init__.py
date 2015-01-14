# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The niftk module provides classes for interfacing with the `NIFTK TOOLS
<http://cmictig.cs.ucl.ac.uk>`_ command line tools.

Top-level namespace for Niftk.
"""

from .base import (NIFTKCommand, Info, no_niftk)
from .operations import (CropImage) 
from .filters import ( N4BiasCorrection )
from .io import ( Midas2Nii, Midas2Nii )
from .qc import ( InterSliceCorrelationPlot, MatrixRotationPlot )
from .stats import ( ComputeDiceScore, CalculateAffineDistances )
from .utils import ( WriteArrayToCsv )
from .diffusion import ( DistortionGenerator, GradwarpCorrection, Noddi )
