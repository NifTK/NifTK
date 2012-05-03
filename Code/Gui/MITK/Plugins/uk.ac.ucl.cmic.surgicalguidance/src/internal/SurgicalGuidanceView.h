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

#include <QPlainTextEdit.h>

#include <QmitkFiducialRegistrationWidget.h>
#include <QmitkUpdateTimerWidget.h>
#include <QmitkToolSelectionWidget.h>
#include <QmitkToolTrackingStatusWidget.h>

#include "QmitkAbstractView.h"
#include "ui_SurgicalGuidanceViewControls.h"

#include "OIGTLSocketObject.h"
#include "Common/NiftyLinkXMLBuilder.h"

/**
 * \class SurgicalGuidanceView
 * \brief User interface to provide Image Guided Surgery functionality.
 * \ingroup uk_ac_ucl_cmic_surgicalguidance_internal
*/
class SurgicalGuidanceView : public QmitkAbstractView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
public:

  SurgicalGuidanceView();
  virtual ~SurgicalGuidanceView();

  /// \brief Static view ID = uk.ac.ucl.cmic.surgicalguidance
  static const std::string VIEW_ID;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

protected slots:

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

protected:

  Ui::SurgicalGuidanceViewControls m_Controls;

private slots:
  void OnAddListeningPort();
  void OnRemoveListeningPort();
  void OnTableSelectionChange(int r, int c, int pr = 0, int pc = 0);

private:
  unsigned long int            m_msgCounter;
  OIGTLMessage::Pointer        m_lastMsg;
  OIGTLSocketObject          * m_sockPointer;
  QList<OIGTLSocketObject *>   m_sockets;
  QPlainTextEdit             * m_trackerDataDisplay;

};

#endif // SurgicalGuidanceView_h

