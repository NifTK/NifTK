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
  Interfaces/niftkIGIDataSourceI.cxx
  Interfaces/niftkIGILocalDataSourceI.cxx
  Interfaces/niftkIGISaveableDataSourceI.cxx
  Interfaces/niftkIGICleanableDataSourceI.cxx
  Interfaces/niftkIGIBufferedSaveableDataSourceI.cxx
  Interfaces/niftkIGIDataSourceFactoryServiceI.cxx
  Interfaces/niftkIGIDataSourceFactoryServiceRAII.cxx
  DataType/niftkIGIDataType.cxx
  DataSource/niftkIGIDataSource.cxx
  DataSource/niftkIGIDataSourceLocker.cxx
  DataSource/niftkIGIDataSourceBuffer.cxx
  DataSource/niftkIGIWaitForSavedDataSourceBuffer.cxx
  DataSource/niftkSingleVideoFrameDataSourceService.cxx
  Threads/niftkIGITimerBasedThread.cxx
  Threads/niftkIGIDataSourceGrabbingThread.cxx
  Threads/niftkIGIDataSourceBackgroundSaveThread.cxx
  Threads/niftkIGIDataSourceBackgroundDeleteThread.cxx
  Dialogs/niftkIGIInitialisationDialog.cxx
  Dialogs/niftkIGIConfigurationDialog.cxx
  Dialogs/niftkIPPortDialog.cxx
  Dialogs/niftkIPHostPortDialog.cxx
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
  Dialogs/niftkLagDialog.h
)

set(UI_FILES
  Dialogs/niftkIPPortDialog.ui
  Dialogs/niftkIPHostPortDialog.ui
  Dialogs/niftkLagDialog.ui
)

set(QRC_FILES
)
