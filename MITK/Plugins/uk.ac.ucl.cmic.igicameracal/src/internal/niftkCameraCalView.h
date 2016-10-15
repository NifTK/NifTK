/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef niftkCameraCalView_h
#define niftkCameraCalView_h

#include <niftkBaseView.h>
#include <service/event/ctkEvent.h>
#include "ui_niftkCameraCalView.h"
#include <niftkNiftyCalVideoCalibrationManager.h>
#include <QFuture>
#include <QFutureWatcher>
#include <ctkDictionary.h>

namespace niftk
{

/**
 * \class CameraCalView
 * \brief User interface to provide controls to do mono/stereo, video camera calibration.
 * \ingroup uk_ac_ucl_cmic_igicameracal_internal
 */
class CameraCalView : public BaseView
{  
  /**
   * this is needed for all Qt objects that should have a Qt meta-object
   * (everything that derives from QObject and wants to have signal/slots)
   */
  Q_OBJECT

public:

  /**
   * \brief Each View for a plugin has its own globally unique ID, this one is
   * "uk.ac.ucl.cmic.igicameracal" and the .cxx file and plugin.xml should match.
   */
  static const QString VIEW_ID;

  CameraCalView();
  virtual ~CameraCalView();

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

  /** Sent by footpedal/hotkey plugin. */
  void OnGrab(const ctkEvent& event);
  void OnUnGrab(const ctkEvent& event);
  void OnClear(const ctkEvent& event);

  /** Sent by DataSources plugin.*/
  void OnUpdate(const ctkEvent& event);

protected:

signals:

  void PauseIGIUpdate(const ctkDictionary&);
  void RestartIGIUpdate(const ctkDictionary&);

private slots:

  void OnGrabButtonPressed();
  void OnUnGrabButtonPressed();
  void OnClearButtonPressed();
  void OnSaveButtonPressed();
  void OnBackgroundGrabProcessFinished();
  void OnBackgroundCalibrateProcessFinished();

  /**
   * \brief Called when user changes any of the 3 combo boxes.
   *
   * If the niftk::NiftyCalVideoCalibrationManager has had any successful
   * snapshots (video and optionally tracking info), then changing any
   * of the combo boxes will trigger a reset.
   */
  void OnComboBoxChanged();

private:

  bool RunGrab();
  void Calibrate();
  double RunCalibration();

  void SetButtonsEnabled(bool isEnabled);

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
  QFuture<bool>                                    m_BackgroundGrabProcess;
  QFutureWatcher<bool>                             m_BackgroundGrabProcessWatcher;
  QFuture<double>                                  m_BackgroundCalibrateProcess;
  QFutureWatcher<double>                           m_BackgroundCalibrateProcessWatcher;
  QString                                          m_DefaultSaveDirectory;
};

} // end namespace

#endif
