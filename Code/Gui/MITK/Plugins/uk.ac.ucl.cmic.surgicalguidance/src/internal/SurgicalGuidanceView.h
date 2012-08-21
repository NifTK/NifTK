/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : $Author$

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#ifndef SurgicalGuidanceView_h
#define SurgicalGuidanceView_h

#include <QPlainTextEdit>
#include <QDebug>

#include <QmitkFiducialRegistrationWidget.h>
#include <QmitkUpdateTimerWidget.h>
#include <QmitkToolSelectionWidget.h>
#include <QmitkToolTrackingStatusWidget.h>
#include "QmitkBaseLegacyView.h"
#include "QmitkAbstractView.h"

#include "mitkCone.h"
#include "mitkSTLFileReader.h"
#include "vtkConeSource.h"
//#include "vnl_vector_fixed.h"


#include "ui_SurgicalGuidanceViewControls.h"
#include "TrackerControlsWidget.h"

#include "OIGTLSocketObject.h"
#include "Common/NiftyLinkXMLBuilder.h"

/**
 * \class SurgicalGuidanceView
 * \brief User interface to provide Image Guided Surgery functionality.
 * \ingroup uk_ac_ucl_cmic_surgicalguidance_internal
*/
class SurgicalGuidanceView : public QmitkBaseLegacyView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

friend class TrackerControlsWidget;
  
public:

  SurgicalGuidanceView();
  virtual ~SurgicalGuidanceView();

  /// \brief Static view ID = uk.ac.ucl.cmic.surgicalguidance
  static const std::string VIEW_ID;

  /// \brief Returns the view ID.
  virtual std::string GetViewID() const;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

  QString CreateTestDeviceDescriptor();

  /// \brief Initialize fiducial registration filters
  void InitializeFilters();

protected slots:

  void sendCrap();

  void sendMessage(OIGTLMessage::Pointer msg, int port);

  /// \brief This function displays handles the messages coming from the tracker
  void handleTrackerData(OIGTLMessage::Pointer msg);

  /// \brief This function displays the received tracker data on the UI
  void displayTrackerData(OIGTLMessage::Pointer msg);

  /// \brief This function interprets the received OIGTL messages
  void interpretMessage(OIGTLMessage::Pointer msg);

  /// \brief This slot is triggered when a client connects to the local server, it changes the UI accordingly
  void clientConnected();
  
  /// \brief This slot is triggered when a client disconnects from the local server, it changes the UI accordingly
  void clientDisconnected();

  /// \brief This slot is triggered to create a default "Cone Representation" for any tracker tool
  mitk::DataNode::Pointer CreateConeRepresentation(const char* label);

  /// \brief This slot is triggered to create a default "Cone Representation" at a given center point for any tracker tool
  mitk::DataNode::Pointer CreateConeRepresentation(const char* label, mitk::Vector3D centerPoint);

  /// \brief This slot is triggered when the user wants to load a surface representation from external STL file
  mitk::Surface::Pointer LoadSurfaceFromSTLFile(QString surfaceFilename);

protected:

  Ui::SurgicalGuidanceViewControls   m_Controls;
  QPlainTextEdit                   * m_consoleDisplay;
  
  mitk::Vector3D                                          m_DirectionOfProjectionVector;  ///< vector for direction of projection of instruments
  mitk::NavigationDataLandmarkTransformFilter::Pointer    m_FiducialRegistrationFilter;   ///< this filter transforms from tracking coordinates into mitk world coordinates
  mitk::NavigationDataLandmarkTransformFilter::Pointer    m_PermanentRegistrationFilter;  ///< this filter transforms from tracking coordinates into mitk world coordinates if needed it is interconnected before the FiducialEegistrationFilter

private slots:
  void OnAddListeningPort();
  void OnRemoveListeningPort();
  void OnTableSelectionChange(int r, int c, int pr = 0, int pc = 0);
  void OnCellDoubleClicked(int r, int c);
  
private:
  unsigned long int            m_msgCounter;
  OIGTLMessage::Pointer        m_lastMsg;
  OIGTLSocketObject          * m_sockPointer;
  QList<OIGTLSocketObject *>   m_sockets;
  QList<XMLBuilderBase *>      m_clientDescriptors;

  TrackerControlsWidget      * m_TrackerControlsWidget;     
  QWidget                    * m_WidgetOnDisplay;        
};

#endif // SurgicalGuidanceView_h

