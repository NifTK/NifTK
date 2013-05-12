/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef TrackedPointerView_h
#define TrackedPointerView_h

#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include "ui_TrackedPointerView.h"
#include <vtkSmartPointer.h>

class vtkMatrix4x4;

/**
 * \class TrackedPointerView
 * \brief User interface to provide controls for a tracked pointer.
 * \ingroup uk_ac_ucl_cmic_igitrackedpointer_internal
*/
class TrackedPointerView : public QmitkBaseView
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  TrackedPointerView();
  virtual ~TrackedPointerView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igitrackedpointer
   */
  static const std::string VIEW_ID;

  /**
   * \brief Returns the view ID.
   */

  virtual std::string GetViewID() const;

protected:

  /**
   *  \brief Called by framework, this method creates all the controls for this view
   */
  virtual void CreateQtPartControl(QWidget *parent);

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus();

protected slots:

protected:

private slots:
  
  /**
   * \brief We listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

  /**
   * \brief When the user selects a new tip to probe matrix, we update the internal transform.
   */
  void OnTipToProbeChanged();

private:

  /**
   * \brief Retrieve's the pref values from preference service, and stored in member variables.
   */
  void RetrievePreferenceValues();

  /**
   * \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
   */
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /**
   * \brief All the controls for the main view part.
   */
  Ui::TrackedPointerView *m_Controls;

  /**
   * \brief Member variables for keeping state between button clicks.
   */
  vtkSmartPointer<vtkMatrix4x4> m_TipToProbeTransform;
  std::string m_TipToProbeFileName;
};

#endif // TrackedPointerView_h
