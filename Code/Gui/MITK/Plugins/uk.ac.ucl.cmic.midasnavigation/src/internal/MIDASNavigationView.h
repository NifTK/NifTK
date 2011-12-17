/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-05 17:31:00 +0000 (Mon, 05 Dec 2011) $
 Revision          : $Revision: 7921 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#ifndef MIDASNavigationView_h
#define MIDASNavigationView_h

#include "QmitkMIDASBaseFunctionality.h"

// GUI
#include "ui_MIDASNavigationViewControls.h"
#include "MIDASNavigationViewControlsImpl.h"

// CTK for event handling
#include "service/event/ctkEventHandler.h"
#include "service/event/ctkEventAdmin.h"

// Blueberry
#include <berryIBundleContext.h>

// MITK
#include "mitkDataNode.h"

class ctkPluginContext;
struct ctkEventAdmin;

/**
 * \class MIDASNavigationView
 * \brief MIDASNavigationView provides a MIDAS style navigation with orientation, slice
 * and magnification controllers that apply to the currently selected image in the view.
 * This class uses the ctkEventAdmin service to send and receive messages.
 *
 * \ingroup uk_ac_ucl_cmic_midasnavigation_internal
 * \sa QmitkMIDASBaseFunctionality
*/
class MIDASNavigationView : public QmitkMIDASBaseFunctionality, public ctkEventHandler
{  
  /// \brief This is needed for all Qt objects that should have a Qt meta-object
  /// (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

  /// \brief This class now uses the CTK Event Admin service for event handling and loose coupling.
  Q_INTERFACES(ctkEventHandler)
  
public:

  MIDASNavigationView();
  virtual ~MIDASNavigationView();

  /// \brief static view ID = uk.ac.ucl.cmic.midasnavigationview
  static const std::string VIEW_ID;

  /// \brief Topic that we publish events under, for slice, magnification and orientation.
  static const QString TOPIC;

  /// \brief Returns the view ID.
  virtual std::string GetViewID() const;

  /// \brief Called from framework to instantiate the Qt GUI components.
  virtual void CreateQtPartControl(QWidget *parent);

public Q_SLOTS:

  /// \brief When the axial radio button is pressed we broadcast a MIDASOrientationChanged event.
  void OnAxialRadioButtonToggled(bool);

  /// \brief When the coronal radio button is pressed we broadcast a MIDASOrientationChanged event.
  void OnCoronalRadioButtonToggled(bool);

  /// \brief When the sagittal radio button is pressed we broadcast a MIDASOrientationChanged event.
  void OnSagittalRadioButtonToggled(bool);

  /// \brief When the slice number is changed we broadcast a MIDASSliceNumberChanged event.
  void OnSliceNumberChanged(int, int);

  /// \brief When the magnification factor is changed we broadcast a MIDASSliceNumberChanged event.
  void OnMagnificationFactorChanged(int, int);

  /// \brief Handle events coming from the event admin service.
  void handleEvent(const ctkEvent& event);

private:

  // Publish the event using the CTK Event Admin
  void PublishEvent(QString topic, QVariant value);

  // Enables us to block or unblock signals on all widgets.
  void SetBlockSignals(bool blockSignals);

  // All the controls for the main view part.
  MIDASNavigationViewControlsImpl*  m_NavigationViewControls;

  // For Event Admin, we store a reference to the CTK plugin context
  ctkPluginContext* m_Context;

  // For Event Admin, we store a reference to the CTK event admin service
  ctkServiceReference m_EventAdminRef;

  // For Event Admin, we store a pointer to the actual CTK event admin implementation.
  ctkEventAdmin* m_EventAdmin;
};

#endif // MIDASNavigationView_h

