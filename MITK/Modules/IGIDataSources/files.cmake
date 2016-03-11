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
  mitkIGITestDataUtils.cxx
  mitkIGIDataType.cxx
  mitkIGIDataSource.cxx
  mitkIGIOpenCVDataType.cxx
  QmitkQImageToMitkImageFilter.cxx
  TrackerControlsWidget.cxx
  QmitkIGITimerBasedThread.cxx
#  QmitkFiducialRegistrationWidgetDialog.cxx
  QmitkIGINiftyLinkDataType.cxx
  QmitkIGINiftyLinkDataSource.cxx
  QmitkIGINiftyLinkDataSourceGui.cxx
  QmitkIGIDataSource.cxx
  QmitkIGIDataSourceBackgroundSaveThread.cxx
  QmitkIGIDataSourceGui.cxx
  QmitkIGILocalDataSource.cxx
  QmitkIGILocalDataSourceGrabbingThread.cxx
  QmitkIGITrackerSource.cxx
  QmitkIGITrackerSourceGui.cxx
  QmitkIGIUltrasonixTool.cxx
  QmitkIGIUltrasonixToolGui.cxx
  QmitkIGIOpenCVDataSource.cxx
  QmitkIGIOpenCVDataSourceGui.cxx
)

set(MOC_H_FILES
  TrackerControlsWidget.h
  QmitkIGITimerBasedThread.h
#  QmitkFiducialRegistrationWidgetDialog.h
  QmitkIGINiftyLinkDataSource.h
  QmitkIGINiftyLinkDataSourceGui.h
  QmitkIGIDataSource.h
  QmitkIGIDataSourceGui.h
  QmitkIGILocalDataSource.h
  QmitkIGITrackerSource.h
  QmitkIGITrackerSourceGui.h
  QmitkIGIUltrasonixTool.h
  QmitkIGIUltrasonixToolGui.h
  QmitkIGIOpenCVDataSource.h
  QmitkIGIOpenCVDataSourceGui.h
)

set(UI_FILES
  TrackerControlsWidget.ui
#  QmitkFiducialRegistrationWidgetDialog.ui
  QmitkIGITrackerSourceGui.ui
  QmitkIGIUltrasonixToolGui.ui
)

set(QRC_FILES
)

# optional audio data source depends on qt-multimedia, which may not be available.
if(QT_QTMULTIMEDIA_INCLUDE_DIR)
  set(CPP_FILES
    ${CPP_FILES}
    AudioDataSource.cxx
    AudioDataSourceGui.cxx
  )
  set(MOC_H_FILES
    ${MOC_H_FILES}
    AudioDataSource.h
    AudioDataSourceGui.h
  )
  set(UI_FILES
    ${UI_FILES}
    AudioDataSourceGui.ui
  )
endif()
