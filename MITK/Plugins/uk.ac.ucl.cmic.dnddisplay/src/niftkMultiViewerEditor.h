/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiViewerEditor_h
#define niftkMultiViewerEditor_h

#include <berryQtEditorPart.h>
#include <berryIPartListener.h>
#include <berryIPreferences.h>

#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <berryISelection.h>
#include <berryISelectionProvider.h>
#include <berryISelectionListener.h>

#include <mitkILinkedRenderWindowPart.h>

#include <QmitkAbstractRenderEditor.h>
#include <mitkDataStorage.h>
#include <mitkRenderingManager.h>
#include <mitkIRenderingManager.h>

#include <uk_ac_ucl_cmic_dnddisplay_Export.h>


namespace mitk
{
class DataNode;
}

class QmitkRenderWindow;

namespace niftk
{

/**
 * \class MultiViewerEditor
 * \brief Provides a display with with multiple image viewers on up to 5 x 5 panes of equal
 * size in a grid layout.
 *
 * As of 18th April 2012, this editor inherits from the QmitkAbstractRenderEditor, and hence
 * conforms to the mitk::IRenderWindowPart which is the new Render Window Abstraction provided by
 * MITK on 24.02.2012, apart from the decorations. This editor purposefully implements the methods
 * EnableDecorations, IsDecorationEnabled, GetDecorations to do nothing (see method documentation).
 *
 * \ingroup uk_ac_ucl_cmic_dnddisplay
 */

class MultiViewerEditorPartListener;
class MultiViewerEditorPrivate;
class MultiViewerWidget;

class DNDDISPLAY_EXPORT MultiViewerEditor :
  public QmitkAbstractRenderEditor, public mitk::ILinkedRenderWindowPart
{
  Q_OBJECT

public:

  berryObjectMacro(MultiViewerEditor)

  MultiViewerEditor();
  ~MultiViewerEditor();

  static const QString EDITOR_ID;

  /// \brief Get hold of the internal niftkMultiViewerWidget.
  MultiViewerWidget* GetMultiViewer();

  // -------------------  mitk::IRenderWindowPart  ----------------------

  /**
   * \see mitk::IRenderWindowPart::GetActiveRenderWindow()
   */
  virtual QmitkRenderWindow* GetActiveQmitkRenderWindow() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindows()
   */
  virtual QHash<QString,QmitkRenderWindow*> GetQmitkRenderWindows() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindow(QString)
   */
  virtual QmitkRenderWindow* GetQmitkRenderWindow(const QString& id) const;

  /**
   * \see mitk::IRenderWindowPart::GetSelectionPosition()
   */
  virtual mitk::Point3D GetSelectedPosition(const QString& id = QString()) const;

  /**
   * \see mitk::IRenderWindowPart::SetSelectedPosition()
   */
  virtual void SetSelectedPosition(const mitk::Point3D& position, const QString& id = QString());

  /**
   * \see mitk::IRenderWindowPart::EnableDecorations(), and in this class, deliberately implemented as a no-op.
   */
  virtual void EnableDecorations(bool enable, const QStringList& decorations = QStringList());

  /**
   * \see mitk::IRenderWindowPart::IsDecorationEnabled(), and in this class, only returns false.
   */
  virtual bool IsDecorationEnabled(const QString& decoration) const;

  /**
   * \see mitk::IRenderWindowPart::GetDecorations(), and in this class, always returns empty list.
   */
  virtual QStringList GetDecorations() const;

  /**
   * Get the RenderingManager used by this editor. This default implementation uses the
   * global MITK RenderingManager provided by mitk::RenderingManager::GetInstance().
   *
   * \see mitk::IRenderWindowPart::GetRenderingManager
   */
  virtual mitk::IRenderingManager* GetRenderingManager() const;

  // -------------------  mitk::ILinkedRenderWindowPart  ----------------------

  /**
   * \see mitk::ILinkedRenderWindowPart::GetSlicesRotator().
   */
  mitk::SlicesRotator* GetSlicesRotator() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::GetSlicesSwiveller().
   */
  mitk::SlicesSwiveller* GetSlicesSwiveller() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::EnableSlicingPlanes().
   */
  void EnableSlicingPlanes(bool enable);

  /**
   * \see mitk::ILinkedRenderWindowPart::IsSlicingPlanesEnabled().
   */
  bool IsSlicingPlanesEnabled() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::EnableLinkedNavigation().
   */
  void EnableLinkedNavigation(bool enable);

  /**
   * \see mitk::ILinkedRenderWindowPart::IsLinkedNavigationEnabled().
   */
  bool IsLinkedNavigationEnabled() const;

protected:

  /// \brief Tells the contained niftkMultiViewerWidget to SetFocus().
  virtual void SetFocus();

  /// \brief Called when the preferences object of this editor changed.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Called when the node selection has changed in the given workbench part.
  void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer>& selectedNodes);

  /// \brief Called when a null selection occurs in the given workbench part.
  void OnNullSelection(berry::IWorkbenchPart::Pointer part);

  /// \brief Creates the main Qt GUI element parts.
  virtual void CreateQtPartControl(QWidget* parent);

private slots:

  void ProcessOptions();

  void OnWindowSelected();

  void OnViewerDestroyed(QObject* object);

private:

  /// \brief Gets the current selection made in the Data Manager view.
  /// Returns an empty list if the view is not available or there is no
  /// selection or the selection is empty.
  QList<mitk::DataNode::Pointer> GetDataManagerSelection() const;

  /// \brief Sets the current selection of the Data Manager view.
  /// \param selection The list of data nodes to be selected in the Data Manager view.
  void SetDataManagerSelection(const QList<mitk::DataNode::Pointer>& dataManagerSelection) const;

  /// \brief Gets the current selection made in this editor part.
  QList<mitk::DataNode::Pointer> GetSelectedNodes() const;

  /// \brief Sets the current selection of this editor part.
  /// Additionally, it notifies other parts of the workbench about the selection change,
  /// unless this is suppressed by passing `false` to the `fireNodesSelected` argument.
  /// Suppressing the event is useful when we just want to update the current selection
  /// after the data manager selection has changed.
  /// \param selection The list of data nodes to be selected in this editor.
  /// \param fireNodesSelected Flag to tell if other workbench parts should be notified.
  void SetSelectedNodes(const QList<mitk::DataNode::Pointer>& selectedNodes, bool fireNodesSelected = true);

  const QScopedPointer<MultiViewerEditorPrivate> d;

  friend class MultiViewerEditorPartListener;
  friend class MultiViewerEditorPrivate;
};

}

#endif
