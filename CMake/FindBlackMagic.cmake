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

set(BlackMagic_FOUND 0)

set(_platform)
if(WIN32)
  set(_platform Win)
elseif(APPLE)
  set(_platform Mac)
else()
  set(_platform Linux)
endif()

find_path(BlackMagic_INCLUDE_DIR
  NAMES whatever.h
  PATHS "C:/Blackmagic DeckLink SDK 10.6.6/${platform}/include"
)

find_library(BlackMagic_LIBRARY
  NAMES dvp 
  PATHS "C:/Blackmagic DeckLink SDK 10.6.6/${platform}/"
)

if(1)
  set(BlackMagic_FOUND 1)
endif()

