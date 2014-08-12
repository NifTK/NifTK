/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkSingleViewerEditor_h
#define QmitkSingleViewerEditor_h

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

#include <niftkDnDDisplayEnums.h>

namespace mitk
{
  class DataNode;
}

/**
 * \class QmitkSingleViewerEditor
 * \brief Simple image viewer that supports cursor and magnification binding.
 *
 * As of 18th April 2012, this editor inherits from the QmitkAbstractRenderEditor, and hence
 * conforms to the mitk::IRenderWindowPart which is the new Render Window Abstraction provided by
 * MITK on 24.02.2012, apart from the decorations. This editor purposefully implements the methods
 * EnableDecorations, IsDecorationEnabled, GetDecorations to do nothing (see method documentation).
 *
 * \ingroup uk_ac_ucl_cmic_dnddisplay
 */

class QmitkSingleViewerEditorPrivate;
class niftkSingleViewerWidget;
class niftkSingleViewerControls;
class QmitkRenderWindow;

class DNDDISPLAY_EXPORT QmitkSingleViewerEditor :
  public QmitkAbstractRenderEditor, public mitk::ILinkedRenderWindowPart
{
  Q_OBJECT

public:

  berryObjectMacro(QmitkSingleViewerEditor)

  QmitkSingleViewerEditor();
  ~QmitkSingleViewerEditor();

  static const std::string EDITOR_ID;

  /// \brief Get hold of the internal niftkSingleViewerWidget.
  niftkSingleViewerWidget* GetSingleViewer();

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
  void EnableLinkedNavigation(bool linkedNavigationEnabled);

  /**
   * \see mitk::ILinkedRenderWindowPart::IsLinkedNavigationEnabled().
   */
  bool IsLinkedNavigationEnabled() const;

  /// \brief Shows the control panel if the mouse pointer is moved over the pin button.
  virtual bool eventFilter(QObject* object, QEvent* event);

  /// \brief Called when one of the viewers receives the focus.
  void OnFocusChanged();

protected:

  /// \brief Tells the contained niftkSingleViewerWidget to SetFocus().
  virtual void SetFocus();

  /// \brief Called when the preferences object of this editor changed.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Creates the main Qt GUI element parts.
  virtual void CreateQtPartControl(QWidget* parent);

  niftkSingleViewerControls* CreateControlPanel(QWidget* parent);

protected slots:

  void OnTimeGeometryChanged(const mitk::TimeGeometry* timeGeometry);

  /// \brief Called when the popup widget opens/closes, and used to re-render the viewers.
  void OnPopupOpened(bool opened);

  /// \brief Called when the pin button is toggled.
  void OnPinButtonToggled(bool checked);

  /// \brief Called when the selected slice has been changed through the control panel.
  void OnSelectedSliceControlChanged(int selectedSlice);

  /// \brief Called when the time step has been changed through the control panel.
  void OnTimeStepControlChanged(int timeStep);

  /// \brief Called when the magnification has been changed through the control panel.
  void OnMagnificationControlChanged(double magnification);

  /// \brief Called when the show cursor option has been changed through the control panel.
  void OnCursorVisibilityControlChanged(bool visible);

  /// \brief Called when the show direction annotations option has been changed through the control panel.
  void OnShowDirectionAnnotationsControlChanged(bool visible);

  /// \brief Called when the show 3D window option has been changed through the control panel.
  void OnShow3DWindowControlChanged(bool visible);

  /// \brief Called when the window layout has been changed through the control panel.
  void OnWindowLayoutControlChanged(WindowLayout windowLayout);

  /// \brief Called when the binding of cursors in the render windows of a viewer has been changed through the control panel.
  void OnWindowCursorBindingControlChanged(bool);

  /// \brief Called when the binding of magnifications in the render windows of a viewer has been changed through the control panel.
  void OnWindowScaleFactorBindingControlChanged(bool);

  /// \brief Called when the selected position has changed in a render window of a viewer.
  /// Each of the contained viewers will signal when its slice navigation controllers have changed.
  void OnSelectedPositionChanged(const mitk::Point3D& selectedPosition);

  /// \brief Called when the selected time step has changed in a viewer.
  /// Each of the contained viewers will signal when its slice navigation controllers have changed.
  void OnTimeStepChanged(int timeStep);

  /// \brief Called when the scale factor of a viewer has changed by zooming in one of its render windows.
  void OnScaleFactorChanged(WindowOrientation orientation, double scaleFactor);

  /// \brief Called when the window layout of a viewer has changed.
  void OnWindowLayoutChanged(WindowLayout windowLayout);

  /// \brief Called when the show cursor option has been changed in a viewer.
  void OnCursorVisibilityChanged(bool visible);

private:

  const QScopedPointer<QmitkSingleViewerEditorPrivate> d;
};

#endif
