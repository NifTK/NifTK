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

macro(NIFTK_INSTALL_CLI)

  set(ARGS ${ARGN})
 
  set(install_directories "")
  list(FIND ARGS DESTINATION _destination_index)
  
  if(_destination_index GREATER -1)
    message(SEND_ERROR "MITK_INSTALL macro must not be called with a DESTINATION parameter.")  
  else()
    if(NOT MACOSX_BUNDLE_NAMES)
      install(${ARGS} DESTINATION bin/cli-modules)
    else()
      foreach(bundle_name ${MACOSX_BUNDLE_NAMES})
        install(${ARGS} DESTINATION ${bundle_name}.app/Contents/MacOS/${_install_DESTINATION}/cli-modules)
      endforeach()
    endif()
  endif()
endmacro()