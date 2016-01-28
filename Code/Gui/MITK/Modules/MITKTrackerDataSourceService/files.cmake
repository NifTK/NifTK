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
  Internal/niftkMITKTrackerDialog.cxx
  Internal/niftkMITKTrackerDataSourceActivator.cxx
  Internal/niftkMITKTrackerDataSourceFactory.cxx
  Internal/niftkMITKAuroraCubeDataSourceFactory.cxx
  Internal/niftkMITKAuroraDomeDataSourceFactory.cxx
  Internal/niftkMITKAuroraTableTopDataSourceFactory.cxx
  Internal/niftkMITKPolarisVicraDataSourceFactory.cxx
  Internal/niftkMITKPolarisSpectraDataSourceFactory.cxx
  Internal/niftkMITKTrackerDataSourceService.cxx
  Internal/niftkIGITrackerDataType.cxx
)

set(MOC_H_FILES
  Internal/niftkMITKTrackerDialog.h
)

set(UI_FILES
  Internal/niftkMITKTrackerDialog.ui
)

set(QRC_FILES
)

