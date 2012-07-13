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

#include "QmitkAbstractView.h"

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
 * Specifically, we use Qt signals/slots to public/receive events asynchronously
 * and to cope with the fact that in general the received signal will be processed
 * in a different thread.
 *
 * \ingroup uk_ac_ucl_cmic_midasnavigation_internal
*/
class MIDASNavigationView : public QmitkAbstractView, public ctkEventHandler
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

  /// \brief Returns the view ID.
  virtual std::string GetViewID() const;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

signals:

  /// \brief Signal emmitted when the slice number changed.
  void SliceNumberChanged(const ctkDictionary&);

  /// \brief Signal emmitted when the magnification factor changed.
  void MagnificationChanged(const ctkDictionary&);

  /// \brief Signal emmitted when the orientation changed.
  void OrientationChanged(const ctkDictionary&);

  /// \brief Signal emmitted when the time step changed.
  void TimeChanged(const ctkDictionary&);

public Q_SLOTS:

  /// \brief Handle events coming from the event admin service.
  void handleEvent(const ctkEvent& event);

  /// \brief When the axial radio button is pressed we broadcast a OrientationChanged event.
  void OnAxialRadioButtonToggled(bool);

  /// \brief When the coronal radio button is pressed we broadcast a OrientationChanged event.
  void OnCoronalRadioButtonToggled(bool);

  /// \brief When the sagittal radio button is pressed we broadcast a OrientationChanged event.
  void OnSagittalRadioButtonToggled(bool);

  /// \brief When the ortho radio button is pressed we broadcast a OrientationChanged event.
  void OnOrthoRadioButtonToggled(bool);

  /// \brief When the 3D radio button is pressed we broadcast a OrientationChanged event.
  void On3DRadioButtonToggled(bool);

  /// \brief When the slice number is changed we broadcast a SliceNumberChanged event.
  void OnSliceNumberChanged(int, int);

  /// \brief When the magnification factor is changed we broadcast a MagnificationChanged event.
  void OnMagnificationFactorChanged(int, int);

  /// \brief When the time step is changed we broadcast a TimeChanged event.
  void OnTimeStepChanged(int, int);

private:

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

