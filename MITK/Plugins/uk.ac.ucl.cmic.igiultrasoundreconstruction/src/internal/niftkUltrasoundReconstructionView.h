/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasoundReconstructionView_h
#define niftkUltrasoundReconstructionView_h

#include <niftkBaseView.h>
#include <service/event/ctkEvent.h>

namespace niftk
{

class USReconController;

/// \class UltrasoundReconstructionView
/// \brief Provides a view for Ultrasound Reconstruction.
///
/// \sa niftkBaseView
/// \sa USReconController
class UltrasoundReconstructionView : public BaseView
{
  Q_OBJECT

public:

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.igiultrasoundreconstruction" and the .cxx file and plugin.xml should match.
  static const QString VIEW_ID;

  /// \brief Constructor.
  UltrasoundReconstructionView();

  /// \brief Copy constructor which deliberately throws a runtime exception, as no-one should call it.
  UltrasoundReconstructionView(const UltrasoundReconstructionView& other);

  /// \brief Destructor.
  virtual ~UltrasoundReconstructionView();

signals:

  void PauseIGIUpdate(const ctkDictionary&);
  void RestartIGIUpdate(const ctkDictionary&);

protected:

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget* parent) override;

  /// \brief Called by framework, this method can set the focus on a specific widget,
  /// but we currently do nothing.
  virtual void SetFocus() override;

  /// \brief Retrieve's the pref values from preference service, and store locally.
  virtual void RetrievePreferenceValues();

private slots:

  /// \brief We listen to the event bus to trigger each frame of grabbing of image/tracking data.
  void OnUpdate(const ctkEvent& event);

  /// \brief We listed to the event bus for the footswitch for manual grabbing of image/tracking data.
  void OnGrab(const ctkEvent& event);

  /// \brief We listen to the event bus for when recording started, to know WHEN to START automatically grabbing image/tracking data.
  void OnRecordingStarted(const ctkEvent& event);

  /// \brief We listen to the event bus for when recording started, to know WHEN to STOP automatically grabbing image/tracking data.
  void OnRecordingStopped(const ctkEvent& event);

private:

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*) override;

  /// \brief The Caffe segmentor controller that realises the GUI logic behind the view.
  QScopedPointer<USReconController> m_USReconController;
};

} // end namespace

#endif
