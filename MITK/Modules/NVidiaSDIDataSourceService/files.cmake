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
  Internal/niftkNVidiaSDIInitDialog.cxx
  Internal/niftkNVidiaSDIConfigDialog.cxx
  Internal/niftkNVidiaSDIDataSourceFactory.cxx
  Internal/niftkNVidiaSDIDataSourceActivator.cxx  
  Internal/niftkNVidiaSDIDataSourceService.cxx
  Internal/niftkNVidiaSDIDataType.cxx
)

set(MOC_H_FILES
  Internal/niftkNVidiaSDIInitDialog.h
  Internal/niftkNVidiaSDIConfigDialog.h
  Internal/niftkNVidiaSDIDataSourceService.h
)

if (CUDA_VERSION_MAJOR LESS 7)	
  list(APPEND CPP_FILES   Internal/niftkNVidiaSDIDataSourceImpl.cxx)
  list(APPEND MOC_H_FILES Internal/niftkNVidiaSDIDataSourceImpl.h)
else()
  list(APPEND CPP_FILES   Internal/niftkNV7SDIDataSourceImpl.cxx)
  list(APPEND MOC_H_FILES Internal/niftkNV7SDIDataSourceImpl.h)
endif()

set(UI_FILES
  Internal/niftkNVidiaSDIInitDialog.ui
  Internal/niftkNVidiaSDIConfigDialog.ui
)

set(QRC_FILES
)

