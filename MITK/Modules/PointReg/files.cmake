#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(CPP_FILES
  Maths/niftkPointRegMaths.cxx
  Registration/niftkArunLeastSquaresPointRegistration.cxx
  Registration/niftkLiuLeastSquaresWithNormalsRegistration.cxx
  Registration/niftkPointBasedRegistration.cxx
  Utilities/niftkHandeyeCalibrateUsingRegistration.cxx
  Utilities/niftkUltrasoundPointerCalibrationCostFunction.cxx
  Utilities/niftkUltrasoundPointerBasedCalibration.cxx
)

