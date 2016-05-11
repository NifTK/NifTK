/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBaseController_h
#define niftkBaseController_h

#include <niftkCoreGuiExports.h>

#include <QObject>

#include <niftkImageOrientationUtils.h>

class QWidget;

class niftkIBaseView;

namespace mitk
{
class BaseRenderer;
}

namespace niftk
{

class BaseGUI;
class BaseControllerPrivate;

/// \class BaseController
class NIFTKCOREGUI_EXPORT BaseController : public QObject
{

  Q_OBJECT

public:

  BaseController(niftkIBaseView* view);

  virtual ~BaseController();

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent);

protected:

  mitk::DataStorage* GetDataStorage() const;

  void RequestRenderWindowUpdate() const;

  /// Returns the current selection made in the datamanager bundle or an empty list
  /// if there is no selection or if it is empty.
  ///
  /// \see QmitkAbstractView::GetDataManagerSelection()
  QList<mitk::DataNode::Pointer> GetDataManagerSelection() const;

  mitk::SliceNavigationController* GetSliceNavigationController() const;

  /// \brief Retrieves the currently active QmitkRenderWindow, and if it has a 2D mapper will return the current orientation of the view, returning ORIENTATION_UNKNOWN if it can't be found or the view is a 3D view for instance.
  ImageOrientation GetImageOrientation();

  /// \brief Creates the widget that holds the GUI components of the view.
  /// This function is called from CreateQtPartControl. Derived classes should provide their implementation
  /// that returns an object whose class derives from niftk::BaseGUI.
  virtual BaseGUI* CreateGUI(QWidget* parent) = 0;

  /// \brief Gets the widget that holds the GUI components of the view.
  BaseGUI* GetGUI() const;

  /// \brief Gets the segmentor BlueBerry view.
  niftkIBaseView* GetView() const;

  /// \brief Called when the window focus changes, and tracks the current mitk::BaseRenderer*.
  virtual void OnFocusChanged();

  /// \brief Returns the currently focused renderer, as this class is tracking the focus changes.
  /// \return mitk::BaseRenderer* The currently focused renderer, or nullptr if it has not been set.
  virtual mitk::BaseRenderer* GetFocusedRenderer() const;

  /// \brief Called when a data node is added to the data storage.
  /// Empty implementation. Derived classes can override it.
  virtual void OnNodeAdded(const mitk::DataNode* node);

  /// \brief Called when a data node in the data storage has changed.
  /// Empty implementation. Derived classes can override it.
  virtual void OnNodeChanged(const mitk::DataNode* node);

  /// \brief Called when a data node has been removed from the data storage.
  /// Empty implementation. Derived classes can override it.
  virtual void OnNodeRemoved(const mitk::DataNode* node);

  /// \brief Called when a data node has been deleted that has previously been in the data storage.
  /// Empty implementation. Derived classes can override it.
  virtual void OnNodeDeleted(const mitk::DataNode* node);

  /// \brief Called when the visibility of a data node in the data storage has changed.
  /// The renderer is nullptr if the global visibility has changed.
  /// Empty implementation. Derived classes can override it.
  virtual void OnNodeVisibilityChanged(const mitk::DataNode* node, const mitk::BaseRenderer* renderer);

private:

  QScopedPointer<BaseControllerPrivate> d_ptr;

  Q_DECLARE_PRIVATE(BaseController);

};

}

#endif
