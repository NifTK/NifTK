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
set(H_FILES
  Interfaces/niftkIGICleanableDataSourceI.h
  Interfaces/niftkIGIDataSourceI.h
  Interfaces/niftkIGILocalDataSourceI.h
  Interfaces/niftkIGISaveableDataSourceI.h
  Interfaces/niftkIGIBufferedSaveableDataSourceI.h
)

set(CPP_FILES
  Interfaces/niftkIGIDataSourceFactoryServiceI.cxx
  Interfaces/niftkIGIDataSourceFactoryServiceRAII.cxx
  DataType/niftkIGIDataType.cxx
  DataType/niftkQImageDataType.cxx
  DataType/niftkIGITrackerDataType.cxx
  DataSource/niftkIGIDataSource.cxx
  DataSource/niftkIGIDataSourceLocker.cxx
  DataSource/niftkIGIDataSourceBuffer.cxx
  DataSource/niftkIGIDataSourceRingBuffer.cxx
  DataSource/niftkIGIDataSourceLinearBuffer.cxx
  DataSource/niftkIGIDataSourceWaitingBuffer.cxx
  DataSource/niftkSingleFrameDataSourceService.cxx
  DataSource/niftkQImageDataSourceService.cxx
  Threads/niftkIGITimerBasedThread.cxx
  Threads/niftkIGIDataSourceGrabbingThread.cxx
  Threads/niftkIGIDataSourceBackgroundSaveThread.cxx
  Threads/niftkIGIDataSourceBackgroundDeleteThread.cxx
  Dialogs/niftkIGIInitialisationDialog.cxx
  Dialogs/niftkIGIConfigurationDialog.cxx
  Dialogs/niftkIPHostPortExtensionDialog.cxx
  Dialogs/niftkLagDialog.cxx
  Dialogs/niftkConfigFileDialog.cxx
  Conversion/niftkQImageToMitkImageFilter.cxx
  Utils/niftkIGIDataSourceUtils.cxx
)

set(MOC_H_FILES
  Threads/niftkIGITimerBasedThread.h
  Dialogs/niftkIGIInitialisationDialog.h
  Dialogs/niftkIGIConfigurationDialog.h
  Dialogs/niftkIPHostPortExtensionDialog.h
  Dialogs/niftkLagDialog.h
  Dialogs/niftkConfigFileDialog.h
)

set(UI_FILES
  Dialogs/niftkIPHostPortExtensionDialog.ui
  Dialogs/niftkLagDialog.ui
  Dialogs/niftkConfigFileDialog.ui
)

set(QRC_FILES
)
