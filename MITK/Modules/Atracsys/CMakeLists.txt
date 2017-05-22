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

if(Atracsys_FOUND)

  MITK_CREATE_MODULE(
    PACKAGE_DEPENDS Qt4|QtCore Qt5|Core
    DEPENDS MitkCore
    NO_INIT      # Not an MITK MicroServices module
  )

  include_directories(${Atracsys_INCLUDE_DIR})
  target_link_libraries(niftkAtracsys PRIVATE ${Atracsys_LIBRARY})

  # Atracsys comes with helper functions in the samples directory.
  # Add that directory, but only if we are building this data source.
  get_filename_component(_samples_dir "${Atracsys_INCLUDE_DIR}/../samples" ABSOLUTE)
  include_directories(${_samples_dir})

  if (NIFTK_USE_COTIRE AND COMMAND cotire)
    cotire(niftkAtracsys)
  endif()

endif()
