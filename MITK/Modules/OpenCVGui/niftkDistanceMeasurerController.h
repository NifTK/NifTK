/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDistanceMeasurerController_h
#define niftkDistanceMeasurerController_h

#include <niftkOpenCVGuiExports.h>
#include <niftkBaseController.h>

class QWidget;

namespace niftk
{

class DistanceMeasurerGUI;
class DistanceMeasurerControllerPrivate;

/// \class DistanceMeasurerController
/// \brief Controller logic for Distance Measurer plugin.
class NIFTKOPENCVGUI_EXPORT DistanceMeasurerController : public BaseController
{

  Q_OBJECT

public:

  DistanceMeasurerController(IBaseView* view);
  virtual ~DistanceMeasurerController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent) override;

  /// \brief Called from GUI by IGIUPDATE trigger.
  void Update();

public slots:

  void OnLeftImageSelectionChanged(const mitk::DataNode*);
  void OnLeftMaskSelectionChanged(const mitk::DataNode*);
  void OnRightImageSelectionChanged(const mitk::DataNode*);
  void OnRightMaskSelectionChanged(const mitk::DataNode*);

protected:

  /// \brief Creates the widget that holds the GUI components of the view.
  /// This function is called from CreateQtPartControl. Derived classes should provide their implementation
  /// that returns an object whose class derives from niftk::BaseGUI.
  virtual BaseGUI* CreateGUI(QWidget* parent);

  virtual void OnNodeRemoved(const mitk::DataNode* node);

protected slots:

  void OnBackgroundProcessFinished();

private:

  double InternalUpdate();

  QScopedPointer<DistanceMeasurerControllerPrivate> d_ptr;
  Q_DECLARE_PRIVATE(DistanceMeasurerController);

};

} // end namespace

#endif
