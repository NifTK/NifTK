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

if(SlicerExecutionModel_FOUND AND VTK_FOUND AND MITK_USE_Qt4)

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME niftkBreastDCEandADC
    BUILD_SLICER
    INSTALL_SCRIPT
    TARGET_LIBRARIES
      ${Boost_LIBRARIES}
      ${ITK_LIBRARIES}
      vtkFiltersCore
      vtkIOImage
      vtkIOLegacy
      ${QT_LIBRARIES}
      niftkcommon
      niftkITK
      niftkITKIO
  )

endif()

