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
  Internal/niftkglhiddenwidget.cpp
)

if (CUDA_VERSION_MAJOR LESS 7)
	
else()
	LIST(APPEND CPP_FILES Internal/niftkNV7SDIDataSourceImpl.cpp)
endif()


set(MOC_H_FILES
  Internal/niftkNVidiaSDIInitDialog.h
  Internal/niftkNVidiaSDIConfigDialog.h
  #Internal/niftkNVidiaSDIDataSourceImpl.h
  Internal/niftkNVidiaSDIDataSourceService.h
  Internal/niftkglhiddenwidget.h
)



if (CUDA_VERSION_MAJOR LESS 7)	
	LIST (APPEND CPP_FILES Internal/niftkNVidiaSDIDataSourceImpl.cxx)
	LIST (MOC_H_FILES Internal/niftkNVidiaSDIDataSourceImpl.h)

else()
	LIST(APPEND MOC_H_FILES Internal/niftkNV7SDIDataSourceImpl.h)
	LIST(APPEND CPP_FILES Internal/niftkNV7SDIDataSourceImpl.cpp)	
endif()

set(UI_FILES
  Internal/niftkNVidiaSDIInitDialog.ui
  Internal/niftkNVidiaSDIConfigDialog.ui
)

set(QRC_FILES
)

