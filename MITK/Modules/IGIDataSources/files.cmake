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
  DataSource/niftkIGIDataSource.cxx
  DataSource/niftkIGIDataSourceLocker.cxx
  DataSource/niftkIGIDataSourceRingBuffer.cxx
  #DataSource/niftkIGIDataSourceBuffer.cxx
  #DataSource/niftkIGIWaitForSavedDataSourceBuffer.cxx
  DataSource/niftkSingleFrameDataSourceService.cxx
  DataSource/niftkQImageDataSourceService.cxx
  Threads/niftkIGITimerBasedThread.cxx
  Threads/niftkIGIDataSourceGrabbingThread.cxx
  Threads/niftkIGIDataSourceBackgroundSaveThread.cxx
  Threads/niftkIGIDataSourceBackgroundDeleteThread.cxx
  Dialogs/niftkIGIInitialisationDialog.cxx
  Dialogs/niftkIGIConfigurationDialog.cxx
  Dialogs/niftkIPPortDialog.cxx
  Dialogs/niftkIPHostPortDialog.cxx
  Dialogs/niftkIPHostPortExtensionDialog.cxx
  Dialogs/niftkIPHostExtensionDialog.cxx
  Dialogs/niftkLagDialog.cxx
  Conversion/niftkQImageToMitkImageFilter.cxx
  Utils/niftkIGIDataSourceUtils.cxx
)

set(MOC_H_FILES
  Threads/niftkIGITimerBasedThread.h
  Dialogs/niftkIGIInitialisationDialog.h
  Dialogs/niftkIGIConfigurationDialog.h
  Dialogs/niftkIPPortDialog.h
  Dialogs/niftkIPHostPortDialog.h
  Dialogs/niftkIPHostPortExtensionDialog.h
  Dialogs/niftkIPHostExtensionDialog.h
  Dialogs/niftkLagDialog.h
)

set(UI_FILES
  Dialogs/niftkIPPortDialog.ui
  Dialogs/niftkIPHostPortDialog.ui
  Dialogs/niftkIPHostPortExtensionDialog.ui
  Dialogs/niftkIPHostExtensionDialog.ui
  Dialogs/niftkLagDialog.ui
)

set(QRC_FILES
)
