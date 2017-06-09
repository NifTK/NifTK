/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBaseView_h
#define niftkBaseView_h

#include <uk_ac_ucl_cmic_common_Export.h>

#include <mitkDataNode.h>
#include <mitkILifecycleAwarePart.h>
#include <QmitkAbstractView.h>

#include <niftkIBaseView.h>

namespace mitk
{
class DataNode;
class SliceNavigationController;
class BaseRenderer;
}


namespace niftk
{

class BaseViewPrivate;

/**
 * \class BaseView
 * \brief Base view component for plugins listening to focus change events and so on.
 *
 * \ingroup uk_ac_ucl_cmic_common
 */
class COMMON_EXPORT BaseView
  : public QmitkAbstractView,
    public virtual IBaseView,
    public mitk::ILifecycleAwarePart
{
  Q_OBJECT

public:

  typedef QmitkAbstractView SuperClass;

  explicit BaseView();
  virtual ~BaseView();

  /**
   * \brief Called when the window focus changes, and tracks the current mitk::BaseRenderer*.
   */
  virtual void OnFocusChanged();

  /**
   * \brief Returns the activation status
   *
   * \return bool true if activated and false otherwise.
   */
  virtual bool IsActivated();

  /**
   * \brief Returns the visible status
   *
   * \return bool true if visible and false otherwise.
   */
  virtual bool IsVisible();

  /**
   * \brief Gets the currently active data storage.
   */
  mitk::DataStorage::Pointer GetDataStorage() const override;

  /**
   * \brief Retrieves a RenderWindow from the mitkRenderWindowPart.
   * \param id The name of the QmitkRenderWindow, such as "axial", "sagittal", "coronal".
   * \return QmitkRenderWindow* The render window or NULL if it can not be found.
   */
  virtual QmitkRenderWindow* GetRenderWindow(QString id);

  /**
   * \brief Retrieves every RenderWindow from the mitkRenderWindowPart.
   * \return The render windows of the render window part, assiciated to their name.
   */
  virtual QHash<QString,QmitkRenderWindow*> GetQmitkRenderWindows() const;

  /**
   * \brief Retrieves the currently selected RenderWindow from the mitkRenderWindowPart.
   * \return QmitkRenderWindow* The selected render window or NULL if it no render window is selected.
   */
  virtual QmitkRenderWindow* GetSelectedRenderWindow() const override;

  /// \brief Gets the visibility of the cursor (aka. crosshair) in the 2D render windows of the main display.
  virtual bool IsActiveEditorCursorVisible() const override;

  /// \brief Sets the visibility of the cursor (aka. crosshair) in the 2D render windows of the main display.
  virtual void SetActiveEditorCursorVisible(bool visible) const override;

  /// \brief Request an update of all render windows of the currently active render window part.
  /// \param requestType Specifies the type of render windows for which an update
  ///                    will be requested.
  virtual void RequestRenderWindowUpdate(mitk::RenderingManager::RequestType requestType = mitk::RenderingManager::REQUEST_UPDATE_ALL) override;

  /// \brief Gets the current selection made in the Data Manager view.
  /// Returns an empty list if the view is not available or there is no
  /// selection or the selection is empty.
  virtual QList<mitk::DataNode::Pointer> GetDataManagerSelection() const override;

  /// \brief Sets the current selection of the Data Manager view.
  /// It also sets the selection to the current view part and notifies the other workbench
  /// parts about the selection change. This is necessary because only the active parts can
  /// notify other parts about the selection change.
  /// \param selection The list of data nodes to be selected in the Data Manager view.
  virtual void SetDataManagerSelection(const QList<mitk::DataNode::Pointer>& selectedNodes) const override;

  /// \brief Sets the current selection of the Data Manager view.
  /// It also sets the selection to the current view part and notifies the other workbench
  /// parts about the selection change. This is necessary because only the active parts can
  /// notify other parts about the selection change.
  /// \param selectedNode The data node to be selected in the Data Manager view.
  virtual void SetDataManagerSelection(mitk::DataNode::Pointer selectedNode) const override;

  /// \brief Retrieve the current slice navigation controller from the currently focused render window.
  /// \return mitk::SliceNavigationController* The slice navigation controller for the currenty focused render window, or <code>NULL</code> if it can't be determined.
  virtual mitk::SliceNavigationController* GetSliceNavigationController() override;

  /// \brief Used to try and get the FocusManager to focus on the Current IRenderWindowPart.
  virtual void FocusOnCurrentWindow() const override;

  /// \brief Gets the selected position in the active render window part.
  virtual mitk::Point3D GetSelectedPosition() const override;

  /// \brief Sets the selected position in the active render window part.
  virtual void SetSelectedPosition(const mitk::Point3D& selectedPosition) override;

protected:

  /**
   * \see mitk::ILifecycleAwarePart::PartActivated
   */
  virtual void Activated() override;

  /**
   * \see mitk::ILifecycleAwarePart::PartDeactivated
   */
  virtual void Deactivated() override;

  /**
   * \see mitk::ILifecycleAwarePart::PartVisible
   */
  virtual void Visible() override;

  /**
   * \see mitk::ILifecycleAwarePart::PartHidden
   */
  virtual void Hidden() override;

  /**
   * Get the parent Qt widget for this view.
   *
   * \return QWidget* The parent widget passed into CreateQtPartControl.
   */
  virtual QWidget* GetParent();

  /**
   * \brief Normally called from within CreateQtPartControl, we store the parent widget.
   *
   * \param QWidget* The widget passed into CreateQtPartControl.
   */
  virtual void SetParent(QWidget*);

  /**
   * \brief Gets the current render window, and sets it to the given coordinate, specified in millimetres.
   * \param coordinate specified in millimetres.
   */
  void SetViewToCoordinate(const mitk::Point3D &coordinate);

  /// \brief Gets the current selection made in this view part.
  QList<mitk::DataNode::Pointer> GetSelectedNodes() const;

  /// \brief Sets the current selection of this view part.
  /// Additionally, it notifies other parts of the workbench about the selection change.
  /// \param selection The list of data nodes to be selected in this view.
  void SetSelectedNodes(const QList<mitk::DataNode::Pointer>& selectedNodes);

private:

  QScopedPointer<BaseViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(BaseView)
  Q_DISABLE_COPY(BaseView)
};

}

#endif
