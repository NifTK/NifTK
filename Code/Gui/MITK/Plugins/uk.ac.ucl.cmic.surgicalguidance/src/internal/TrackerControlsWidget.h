#ifndef TRACKERCONTROLSWIDGET_H
#define TRACKERCONTROLSWIDGET_H

//Qt includes
#include <QtCore>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QResource>
#include <QDebug>
#include <QErrorMessage>
#include <QMessageBox>
#include <QDomDocument>

//NiftyLink includes
#include "OIGTLMessage.h"
#include "OIGTLSocketObject.h"
#include "OIGTLTransformMessage.h"
#include "OIGTLTrackingDataMessage.h"
#include "OIGTLStatusMessage.h"
#include "OIGTLStringMessage.h"

#include <QmitkFiducialRegistrationWidget.h>

#include <mitkNavigationDataToPointSetFilter.h>
#include <mitkNavigationDataLandmarkTransformFilter.h>
#include <mitkNavigationDataReferenceTransformFilter.h>
#include <mitkNavigationDataObjectVisualizationFilter.h>
#include <mitkNavigationDataToPointSetFilter.h>
#include <mitkTrackingDeviceSource.h>
#include <mitkSurface.h>
#include <mitkCameraVisualization.h>

class SurgicalGuidanceView;

//GUI
#include "ui_TrackerControlsWidget.h"

/**
 * \class TrackerControlsWidget
 * \brief 
 */

class TrackerControlsWidget : public QWidget
{
  Q_OBJECT

public:
  /// \brief Basic constructor which initializes the socket and sets up signal - slot connections
  TrackerControlsWidget(QObject *parent = 0);
  /// \brief Basic destructor which shuts down
  ~TrackerControlsWidget(void);

  /// \brief
  void SetSurgicalGuidanceViewPointer(SurgicalGuidanceView * p);

  /// \brief This method initializes the registration for the FiducialRegistrationWidget
  void InitializeRegistration();


signals:

protected:

private:

private slots:

  /// \brief This slot is triggered when the tracker is needs connect / disconnect
  void manageTrackerConnection();

  /// \brief This slot is triggered when tools need to be added / removed from the tracker
  void manageToolConnection(void);

  /// \brief UI event handling
  void OnStartTrackingClicked(void);
  void OnGetCurrentPositionClicked(void);
  void OnManageToolsClicked(void);
  void OnFiducialRegistrationClicked(void);

  /// \brief 
  void OnRegisterFiducials( );

  /// \brief This method sets up the filter pipeline.
  void SetupIGTPipeline();
  /// \brief This method initializes all needed filters.
  void InitializeFilters();
  /// \brief This method destroys the filter pipeline.
  void DestroyIGTPipeline();

private:
  bool                            m_listening;
  bool                            m_connecting;
  bool                            m_trackerConnected;
  bool                            m_tracking;
  
  bool                            m_sending1TransMsg;
  bool                            m_sendingMsgStream;
  
  OIGTLSocketObject             * m_socket;
  unsigned long int               m_msgCounter;

  QStringList                     m_toolList;
  QString                         m_currDir;

  OIGTLMessage::Pointer           m_lastMsg;

  mitk::DataNode::Pointer            m_ImageFiducialsDataNode;
  mitk::DataNode::Pointer            m_TrackerFiducialsDataNode;

  QmitkFiducialRegistrationWidget  * m_FiducialRegWidget;
  SurgicalGuidanceView             * m_SGViewPointer;

  bool                               m_FidRegInitialized;

  mitk::Vector3D                                          m_DirectionOfProjectionVector;///< vector for direction of projection of instruments
  mitk::TrackingDeviceSource::Pointer                     m_Source; ///< source that connects to the tracking device
  mitk::NavigationDataLandmarkTransformFilter::Pointer    m_FiducialRegistrationFilter; ///< this filter transforms from tracking coordinates into mitk world coordinates
  mitk::NavigationDataLandmarkTransformFilter::Pointer    m_PermanentRegistrationFilter; ///< this filter transforms from tracking coordinates into mitk world coordinates if needed it is interconnected before the FiducialEegistrationFilter
  mitk::NavigationDataObjectVisualizationFilter::Pointer  m_Visualizer; ///< visualization filter
  mitk::CameraVisualization::Pointer                      m_VirtualView; ///< filter to update the vtk camera according to the reference navigation data

  Ui::TrackerControlsWidget ui;
};

#endif
