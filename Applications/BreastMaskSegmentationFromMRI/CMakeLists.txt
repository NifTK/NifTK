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

if(VTK_FOUND)

  # I do not understand why Boost is required here, as it is used only by the niftkcommon
  # library where it is declared as a private dependency. It is not used by this application
  # directly.

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME niftkBreastMaskSegmentationFromMRI
    BUILD_CLI
    TARGET_LIBRARIES
      ${Boost_LIBRARIES}
      ${ITK_LIBRARIES}
      vtkCommonCore
      vtkCommonExecutionModel
      vtkCommonMisc
      vtkFiltersCore
      vtkIOImage
      vtkIOLegacy
      niftkcommon
      niftkITK
      niftkITKIO
  )

endif()
