/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef CameraCalView_h
#define CameraCalView_h

#include <QmitkBaseView.h>
#include <service/event/ctkEvent.h>
#include "ui_CameraCalView.h"
#include <niftkNiftyCalVideoCalibrationManager.h>
#include <QFuture>
#include <QFutureWatcher>

namespace niftk
{

/**
 * \class CameraCalView
 * \brief User interface to provide controls to do mono/stereo, video camera calibration.
 * \ingroup uk_ac_ucl_cmic_igicameracal_internal
 */
class CameraCalView : public QmitkBaseView
{  
  /**
   * this is needed for all Qt objects that should have a Qt meta-object
   * (everything that derives from QObject and wants to have signal/slots)
   */
  Q_OBJECT

public:

  CameraCalView();
  virtual ~CameraCalView();

  /**
   * \brief Static view ID = uk.ac.ucl.cmic.igicameracal
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
  virtual void CreateQtPartControl(QWidget *parent) override;

  /**
   * \brief Called by framework, sets the focus on a specific widget.
   */
  virtual void SetFocus() override;

protected slots:

protected:

private slots:

  void OnGrabButtonPressed();
  void OnUnGrabButtonPressed();
  void OnSaveButtonPressed();
  void OnBackgroundProcessFinished();

  /**
   * \brief Called when user changes any of the 3 combo boxes.
   *
   * If the niftk::NiftyCalVideoCalibrationManager has had any successful
   * snapshots (video and optionally tracking info), then changing any
   * of the combo boxes will trigger a reset.
   */
  void OnComboBoxChanged();

private:

  void Calibrate();
  double RunCalibration();

  /**
   * \brief Retrieve's the pref values from preference service, and stored in member variables.
   */
  void RetrievePreferenceValues();

  /**
   * \brief BlueBerry's notification about preference changes (e.g. from a preferences dialog).
   */
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*) override;

  Ui::CameraCalView                               *m_Controls;
  niftk::NiftyCalVideoCalibrationManager::Pointer  m_Manager;
  QFuture<double>                                  m_BackgroundProcess;
  QFutureWatcher<double>                           m_BackgroundProcessWatcher;
  QString                                          m_DefaultSaveDirectory;
};

} // end namespace

#endif
