/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef TagTrackerView_h
#define TagTrackerView_h

#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include <mitkDataNode.h>
#include <mitkTagTrackingRegistrationManager.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include "ui_TagTrackerViewControls.h"
#include <cv.h>

/**
 * \class TagTrackerView
 * \brief User interface to provide a small plugin to track Augmented Reality tags.
 * \ingroup uk_ac_ucl_cmic_igitagtracker_internal
*/
class TagTrackerView : public QmitkBaseView, public Ui::TagTrackerViewControls
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:

  TagTrackerView();
  virtual ~TagTrackerView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igitagtracker
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
   * \brief We can listen to the event bus to trigger updates.
   */
  void OnUpdate(const ctkEvent& event);

  /**
   * \brief Or we can listed to a manual update button.
   */
  void OnManualUpdate();

  /**
   * \brief if any spin box pressed, we update.
   */
  void OnSpinBoxPressed();

  /**
   * \brief We can toggle, whether or not to update the registration.
   */
  void OnRegistrationEnabledChecked(bool isChecked);

  /**
   * \brief Used to copy the current tracking transformation to the reference.
   */
  void OnGrabReferencePressed();

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
   * \brief Main method to update tag positions.
   */
  void UpdateTags();

  /**
   * \brief If true, we continuously update according to the ping event on the CTK Event Bus.
   */
  bool m_ListenToEventBusPulse;

  /**
   * \brief If true, we dont bother with stereo and triangulation.
   */
  bool m_MonoLeftCameraOnly;

  /**
   * \brief to make sure we only show dialog box once.
   */
  bool m_ShownStereoSameNameWarning;

  /**
   * \brief Store a reference matrix, copied from whatever the current registration is.
   */
  vtkSmartPointer<vtkMatrix4x4> m_ReferenceMatrix;

  /**
   * \brief This gets updated at each successful registration.
   */
  vtkSmartPointer<vtkMatrix4x4> m_CurrentRegistrationMatrix;

  /**
   * \brief THis is made a member variable, so we can use the same one each time to compute relative transformations, as the tracking proceeds.
   */
  mitk::TagTrackingRegistrationManager::Pointer m_TagTrackingRegistrationManager;

  double m_RangesOfRotationalParams[6];
};

#endif // TagTrackerView_h
