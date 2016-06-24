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

if(NOT WIN32)
  set(scripts
    cteOverlayThicknessImageOnFreeSurfer.sh
    cteRunCrossSectional.sh
    cteRunCrossSectionalBatch.sh
    cteRunLongitudinal.sh
    cteRegisterAtlasToImage.sh
    cteRegisterThicknessImageToAtlas.sh
    cteRefineLabelImage.sh
    cteExtractStatsUsingAtlas.sh
    cteMaskImageUsingRegions.sh
  )
  foreach(script ${scripts})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${script}.in ${EXECUTABLE_OUTPUT_PATH}/${script} @ONLY )
    MITK_INSTALL(PROGRAMS ${EXECUTABLE_OUTPUT_PATH}/${script})
  endforeach()
endif()
 