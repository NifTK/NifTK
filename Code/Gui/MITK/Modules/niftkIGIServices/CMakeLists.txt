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

###############################################################################
# Note:
# This Module should not depend on any module in NifTK.
# It should just define pure interfaces, and RAII objects to load them.
# Hence it should just depend on MitkCore.
###############################################################################

MITK_CREATE_MODULE(niftkIGIServices
  EXPORT_DEFINE NIFTKIGISERVICES_EXPORT
  DEPENDS MitkCore
)

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkIGIServices)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
