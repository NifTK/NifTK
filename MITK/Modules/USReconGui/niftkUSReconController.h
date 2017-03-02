/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUSReconController_h
#define niftkUSReconController_h

#include <niftkUSReconGuiExports.h>
#include <niftkBaseController.h>

class QWidget;

namespace niftk
{

class USReconGUI;
class USReconControllerPrivate;

/// \class USReconController
/// \brief Controller logic for Ultrasound Reconstruction plugin.
class NIFTKUSRECONGUI_EXPORT USReconController : public BaseController
{

  Q_OBJECT

public:

  USReconController(IBaseView* view);
  virtual ~USReconController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent) override;

  /// \brief We pass in the recording dir, so we can dump images to the same folder.
  void SetRecordingStarted(const QString& recordingDir);

  /// \brief Stops this controller dumping images.
  void SetRecordingStopped();

  /// \brief Called from GUI by IGIUPDATE trigger.
  /// So, the rate at which we dump images is controlled by screen rate.
  void Update();

public slots:

  void OnImageSelectionChanged(const mitk::DataNode*);
  void OnTrackingSelectionChanged(const mitk::DataNode*);
  void OnGrabPressed();
  void OnClearDataPressed();
  void OnSaveDataPressed();
  void OnCalibratePressed();
  void OnReconstructPressed();

protected:

  /// \brief Creates the widget that holds the GUI components of the view.
  /// This function is called from CreateQtPartControl. Derived classes should provide their implementation
  /// that returns an object whose class derives from niftk::BaseGUI.
  virtual BaseGUI* CreateGUI(QWidget* parent);

private slots:

  void OnBackgroundProcessFinished();

private:

  QScopedPointer<USReconControllerPrivate> d_ptr;
  Q_DECLARE_PRIVATE(USReconController);

  void CaptureImages();
  void DoReconstruction();
  void DoReconstructionInBackground(); 
};

} // end namespace

#endif
