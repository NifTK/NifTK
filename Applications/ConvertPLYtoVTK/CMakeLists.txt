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

  NIFTK_CREATE_COMMAND_LINE_APPLICATION(
    NAME niftkConvertPLYtoVTK
    BUILD_CLI
    TARGET_LIBRARIES
      vtkIOLegacy
      vtkIOPLY
      vtkInteractionStyle
      vtkRenderingCore
      vtkRenderingFreeType
      vtkRenderingFreeTypeOpenGL
      vtkRenderingOpenGL
      niftkcommon
  )

endif()

