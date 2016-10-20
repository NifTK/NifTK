/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMaskMergerController_h
#define niftkMaskMergerController_h

#include <niftkIGIGuiExports.h>
#include <niftkBaseController.h>

class QWidget;

namespace niftk
{

class MaskMergerGUI;
class MaskMergerControllerPrivate;

/// \class MaskMergerController
/// \brief Controller logic for Mask Merger plugin.
class NIFTKIGIGUI_EXPORT MaskMergerController : public BaseController
{

  Q_OBJECT

public:

  MaskMergerController(IBaseView* view);
  virtual ~MaskMergerController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent) override;

  /// \brief Called from GUI by IGIUPDATE trigger.
  void Update();

public slots:

protected:

  /// \brief Creates the widget that holds the GUI components of the view.
  /// This function is called from CreateQtPartControl. Derived classes should provide their implementation
  /// that returns an object whose class derives from niftk::BaseGUI.
  virtual BaseGUI* CreateGUI(QWidget* parent);

  virtual void OnNodeRemoved(const mitk::DataNode* node);

protected slots:

private:

  QScopedPointer<MaskMergerControllerPrivate> d_ptr;
  Q_DECLARE_PRIVATE(MaskMergerController);

};

} // end namespace

#endif
