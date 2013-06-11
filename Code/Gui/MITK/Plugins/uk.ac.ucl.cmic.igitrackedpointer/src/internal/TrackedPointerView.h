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
#include <mitkDataStorage.h>
#include <mitkTrackedPointerManager.h>

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
   * \brief Called from GUI button to create points set and grab current pointer location.
   */
  void OnGrabPoints();

  /**
   * \brief Called from the GUI button to clear all points.
   */
  void OnClearPoints();

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
  bool m_UpdateViewCoordinate;
  mitk::DataStorage* m_DataStorage;
  mitk::TrackedPointerManager::Pointer m_TrackedPointerManager;
};

#endif // TrackedPointerView_h
