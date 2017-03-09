/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIBaseView_h
#define niftkIBaseView_h

#include "niftkCoreGuiExports.h"

#include <QHash>

#include <mitkDataStorage.h>
#include <mitkRenderingManager.h>

namespace mitk
{
class BaseRenderer;
class SliceNavigationController;
}

class QmitkRenderWindow;

namespace niftk
{

/// \class IBaseView
/// \brief Public interface to QmitkBaseView.
///
/// The aim of this class is to expose view functionality to the module layer.
/// \sa QmitkBaseView
class NIFTKCOREGUI_EXPORT IBaseView
{

public:

  /// \brief Used to try and get the FocusManager to focus on the current IRenderWindowPart.
  virtual void FocusOnCurrentWindow() const = 0;

  /// \brief Returns every render window of the current IRenderWindowPart.
  virtual QHash<QString,QmitkRenderWindow*> GetQmitkRenderWindows() const = 0;

  /// \brief Retrieves the currently selected RenderWindow from the mitkRenderWindowPart.
  /// \return QmitkRenderWindow* The selected render window or NULL if it no render window is selected.
  virtual QmitkRenderWindow* GetSelectedRenderWindow() const = 0;

  virtual mitk::DataStorage::Pointer GetDataStorage() const = 0;

  /// \brief Request an update of all render windows of the currently active render window part.
  /// \param requestType Specifies the type of render windows for which an update
  ///                    will be requested.
  virtual void RequestRenderWindowUpdate(mitk::RenderingManager::RequestType requestType = mitk::RenderingManager::REQUEST_UPDATE_ALL) = 0;

  /// \brief Gets the list of nodes selected in the data manager.
  /// \return The current selection made in the datamanager bundle or an empty list
  /// if there is no selection or if it is empty.
  virtual QList<mitk::DataNode::Pointer> GetDataManagerSelection() const = 0;

  /// \brief Informs other parts of the workbench that node is selected via the blueberry selection service.
  ///
  /// \note This method should not be used if you have set your own selection provider via
  /// SetSelectionProvider() or your own QItemSelectionModel via GetDataNodeSelectionModel().
  virtual void FireNodeSelected(mitk::DataNode::Pointer node) = 0;

  /// \brief Retrieve the current slice navigation controller from the currently focused render window.
  /// \return mitk::SliceNavigationController* The slice navigation controller for the currenty focused render window,
  ///  or <code>nullptr</code> if it can't be determined.
  virtual mitk::SliceNavigationController* GetSliceNavigationController() = 0;

  /// \brief Tells if the cursor (aka. crosshair) is visible in the active editor.
  virtual bool IsActiveEditorCursorVisible() const = 0;

  /// \brief Shows or hides the cursor (aka. crosshair) is in the active editor.
  virtual void SetActiveEditorCursorVisible(bool visible) const = 0;

  /// \brief Gets the selected position in the active render window part.
  virtual mitk::Point3D GetSelectedPosition() const = 0;

  /// \brief Sets the selected position in the active render window part.
  virtual void SetSelectedPosition(const mitk::Point3D& selectedPosition) = 0;

};

}

#endif
