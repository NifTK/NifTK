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

#include <mitkDataNode.h>

#include <niftkImageOrientationUtils.h>

class QWidget;

namespace mitk
{
class BaseRenderer;
class DataStorage;
}

namespace niftk
{

class BaseGUI;
class BaseControllerPrivate;
class IBaseView;


/// \class BaseController
class NIFTKCOREGUI_EXPORT BaseController : public QObject
{

  Q_OBJECT

public:

  /// \brief Constructs a BaseController object.
  BaseController(IBaseView* view);

  /// \brief Destructs the BaseController object.
  virtual ~BaseController();

  /// \brief Returns the orientation of the selected render window.
  /// Returns IMAGE_ORIENTATION_UNKNOWN if no window is selected or the selected window is a 3D window.
  ImageOrientation GetOrientation() const;

  /// \brief Returns the index of the displayed slice in the currently selected window.
  /// Returns -1 if no window is selected or the selected window is a 3D window.
  int GetSliceIndex() const;

  /// \brief Returns the selected position in the current editor (render window part).
  /// The selected position is the voxel at the intersection of the crosshair planes.
  mitk::Point3D GetSelectedPosition() const;

  /// \brief Sets the selected position in the current editor (render window part).
  /// The selected position is the voxel at the intersection of the crosshair planes.
  void SetSelectedPosition(const mitk::Point3D& selectedPosition) const;

  /// \brief Sets up the GUI.
  /// This function has to be called from the CreateQtPartControl function of the view.
  virtual void SetupGUI(QWidget* parent);

  /// \brief Called when the BlueBerry view that hosts the GUI for this controller gets activated.
  virtual void OnViewGetsActivated();

  /// \brief Called when the BlueBerry view that hosts the GUI for this controller gets deactivated.
  virtual void OnViewGetsDeactivated();

  /// \brief Called when the BlueBerry view that hosts the GUI for this controller becomes visible.
  virtual void OnViewGetsVisible();

  /// \brief Called when the BlueBerry view that hosts the GUI for this controller becomes hidden.
  virtual void OnViewGetsHidden();

protected:

  mitk::DataStorage* GetDataStorage() const;

  void RequestRenderWindowUpdate() const;

  /// Returns the current selection made in the datamanager bundle or an empty list
  /// if there is no selection or if it is empty.
  ///
  /// \see QmitkAbstractView::GetDataManagerSelection()
  QList<mitk::DataNode::Pointer> GetDataManagerSelection() const;

  mitk::SliceNavigationController* GetSliceNavigationController() const;

  /// \brief Called when the selected slice changes.
  /// This happens when selected window changes in the current editor (render window part)
  /// or when the focus is on a 2D render window in the current editor and the selected
  /// slice changes, either through interaction (e.g. by scrolling with the mouse wheel)
  /// or through an API call.
  /// \param orientation
  ///     The orientation of the 2D window or IMAGE_ORIENTATION_UNKNOWN if a 3D window is selected.
  /// \param sliceIndex
  ///     The index of the current slice or -1 if a 3D window is selected.
  virtual void OnSelectedSliceChanged(niftk::ImageOrientation orientation, int sliceIndex);

  /// \brief Creates the widget that holds the GUI components of the view.
  /// This function is called from CreateQtPartControl. Derived classes should provide their implementation
  /// that returns an object whose class derives from niftk::BaseGUI.
  virtual BaseGUI* CreateGUI(QWidget* parent) = 0;

  /// \brief Gets the widget that holds the GUI components of the view.
  BaseGUI* GetGUI() const;

  /// \brief Updates the GUI based on the current data (model).
  /// This is an empty implementation and it is not used in this class. Derived classes,
  /// however, are encouraged to override this function to their needs for consistency.
  virtual void UpdateGUI() const;

  /// \brief Gets the segmentor BlueBerry view.
  IBaseView* GetView() const;

  /// \brief Called when the window focus changes, and tracks the current mitk::BaseRenderer*.
  virtual void OnFocusChanged();

  /// \brief Returns the currently focused renderer, as this class is tracking the focus changes.
  /// \return mitk::BaseRenderer* The currently focused renderer, or nullptr if it has not been set.
  virtual mitk::BaseRenderer* GetFocused2DRenderer() const;

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

  /// Convenient method to set and reset a wait cursor ("hourglass")
  void WaitCursorOn();

  /// Convenient method to restore the standard cursor
  void WaitCursorOff();

  /// Convenient method to set and reset a busy cursor
  void BusyCursorOn();

  /// Convenient method to restore the standard cursor
  void BusyCursorOff();

private:

  /// Convenient method to restore the standard cursor
  void RestoreOverrideCursor();

  QScopedPointer<BaseControllerPrivate> d_ptr;

  Q_DECLARE_PRIVATE(BaseController)

};

}

#endif
