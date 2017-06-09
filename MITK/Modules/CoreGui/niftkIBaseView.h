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

  /// \brief Gets the current selection made in the Data Manager view.
  /// Returns an empty list if the view is not available or there is no
  /// selection or the selection is empty.
  virtual QList<mitk::DataNode::Pointer> GetDataManagerSelection() const = 0;

  /// \brief Sets the current selection of the Data Manager view.
  /// It also sets the selection to the current view part and notifies the other workbench
  /// parts about the selection change. This is necessary because only the active parts can
  /// notify other parts about the selection change.
  /// \param selectedNodes The list of data nodes to be selected in the Data Manager view.
  virtual void SetDataManagerSelection(const QList<mitk::DataNode::Pointer>& selectedNodes) const = 0;

  /// \brief Sets the current selection of the Data Manager view.
  /// It also sets the selection to the current view part and notifies the other workbench
  /// parts about the selection change. This is necessary because only the active parts can
  /// notify other parts about the selection change.
  /// \param selectedNode The data node to be selected in the Data Manager view.
  virtual void SetDataManagerSelection(mitk::DataNode::Pointer selectedNode) const = 0;

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
