/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkBaseView_h
#define QmitkBaseView_h

#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>

#include <mitkDataNode.h>
#include <mitkILifecycleAwarePart.h>
#include <QmitkAbstractView.h>

#include <niftkIBaseView.h>
#include "internal/VisibilityChangeObserver.h"

class QmitkBaseViewPrivate;

namespace mitk {
class DataNode;
class SliceNavigationController;
class BaseRenderer;
}

/**
 * \class QmitkBaseView
 * \brief Base view component for plugins listening to visibility change events,
 * focus changed events and so on.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkBaseView : public QmitkAbstractView,
                                     public virtual niftkIBaseView,
                                     public VisibilityChangeObserver,
                                     public mitk::ILifecycleAwarePart
{
  Q_OBJECT

public:

  typedef QmitkAbstractView SuperClass;

  explicit QmitkBaseView();
  virtual ~QmitkBaseView();

  /**
   * \brief Called when the visibility of a node in the data storage changed.
   * \param node The node in the data storage whose visibility property has been modified.
   */
  virtual void onVisibilityChanged(const mitk::DataNode* node) override;

  /**
   * \brief Called when the window focus changes, and tracks the current mitk::BaseRenderer*.
   */
  virtual void OnFocusChanged();

  /**
   * \brief Returns whether this functionality should be exclusive, or in other words, the only active plugin.
   * \return false Always false.
   */
  virtual bool IsExclusiveFunctionality() const;

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
   * \brief Selects the data node in this view and also in the data manager.
   * It sets the "selected" property of the node. As a (positive) side effect
   * of changing the data manager selection, the "selected" property of the
   * previously selected nodes will be cleared.
   */
  virtual void SetCurrentSelection(mitk::DataNode::Pointer dataNode) override;

  /// \brief \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes) override;

  /**
   * \brief Retrieves a RenderWindow from the mitkRenderWindowPart.
   * \param id The name of the QmitkRenderWindow, such as "axial", "sagittal", "coronal".
   * \return QmitkRenderWindow* The render window or NULL if it can not be found.
   */
  virtual QmitkRenderWindow* GetRenderWindow(QString id);

  /**
   * \brief Retrieves the currently selected RenderWindow from the mitkRenderWindowPart.
   * \return QmitkRenderWindow* The selected render window or NULL if it no render window is selected.
   */
  virtual QmitkRenderWindow* GetSelectedRenderWindow() const override;

  /// \brief Sets the visibility of the cursor (aka. crosshair) in the 2D render windows of the main display.
  /// \return The actual visibility of the cursor when this function is called.
  bool SetMainWindowCursorVisible(bool visible);

  /// \brief Request an update of all render windows of the currently active render window part.
  /// \param requestType Specifies the type of render windows for which an update
  ///                    will be requested.
  virtual void RequestRenderWindowUpdate(mitk::RenderingManager::RequestType requestType = mitk::RenderingManager::REQUEST_UPDATE_ALL) override;

  /// \brief Gets the list of nodes selected in the data manager.
  /// \return The current selection made in the datamanager bundle or an empty list
  /// if there is no selection or if it is empty.
  QList<mitk::DataNode::Pointer> GetDataManagerSelection() const override;

  /// \brief Informs other parts of the workbench that node is selected via the blueberry selection service.
  ///
  /// \note This method should not be used if you have set your own selection provider via
  /// SetSelectionProvider() or your own QItemSelectionModel via GetDataNodeSelectionModel().
  virtual void FireNodeSelected(mitk::DataNode::Pointer node) override;

  /// \brief Convenient method to set and reset a wait cursor ("hourglass")
  virtual void WaitCursorOn() override;

  /// \brief Convenient method to restore the standard cursor
  virtual void WaitCursorOff() override;

  /// \brief Retrieve the current slice navigation controller from the currently focused render window.
  /// \return mitk::SliceNavigationController* The slice navigation controller for the currenty focused render window, or <code>NULL</code> if it can't be determined.
  virtual mitk::SliceNavigationController* GetSliceNavigationController() override;

  /// \brief Returns the currently focused renderer, as this class is tracking the focus changes.
  /// \return mitk::BaseRenderer* The currently focused renderer, or nullptr if it has not been set.
  virtual mitk::BaseRenderer* GetFocusedRenderer() const override;

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
   * \brief Returns the slice number from the slice navigatiob controller from the currently focused render window.
   *
   * \return int The slice number or -1 if it can't be found.
   */
  int GetSliceNumberFromSliceNavigationController();

  /**
   * \brief Gets the current render window, and sets it to the given coordinate, specified in millimetres.
   * \param coordinate specified in millimetres.
   */
  void SetViewToCoordinate(const mitk::Point3D &coordinate);

private:

  void onNodeAddedInternal(const mitk::DataNode*);
  void onNodeRemovedInternal(const mitk::DataNode*);

  mitk::SliceNavigationController* GetSliceNavigationControllerInternal();

  QScopedPointer<QmitkBaseViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(QmitkBaseView);
  Q_DISABLE_COPY(QmitkBaseView);
};

#endif // __QmitkBaseView_h
