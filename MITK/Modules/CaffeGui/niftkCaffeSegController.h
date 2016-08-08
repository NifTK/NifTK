/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCaffeSegController_h
#define niftkCaffeSegController_h

#include <niftkCaffeGuiExports.h>
#include <niftkBaseController.h>

class QWidget;

namespace niftk
{

class CaffeSegGUI;
class CaffeSegControllerPrivate;

/// \class CaffeSegController
/// \brief Controller logic for Caffe segmentation tester plugin.
class NIFTKCAFFEGUI_EXPORT CaffeSegController : public BaseController
{

  Q_OBJECT

public:

  CaffeSegController(IBaseView* view);
  virtual ~CaffeSegController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent) override;

protected:

  /// \brief Creates the widget that holds the GUI components of the view.
  /// This function is called from CreateQtPartControl. Derived classes should provide their implementation
  /// that returns an object whose class derives from niftk::BaseGUI.
  virtual BaseGUI* CreateGUI(QWidget* parent);

protected slots:

private:

  QScopedPointer<CaffeSegControllerPrivate> d_ptr;
  Q_DECLARE_PRIVATE(CaffeSegController);

};

} // end namespace

#endif
