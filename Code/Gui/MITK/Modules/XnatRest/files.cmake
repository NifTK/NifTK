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
  XnatRest.c
  XnatRestIoapi.c
  XnatRestMiniUnz.c
  XnatRestMiniZip.c
  XnatRestUnzip.c
  XnatRestZip.c
)

if(WIN32)
  set(CPP_FILES ${CPP_FILES} XnatRestIowin32.c)
endif(WIN32)
