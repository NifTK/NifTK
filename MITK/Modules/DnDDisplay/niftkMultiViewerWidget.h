/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiViewerWidget_h
#define niftkMultiViewerWidget_h

#include <niftkDnDDisplayExports.h>

#include <QColor>
#include <QEvent>
#include <QList>
#include <QWidget>

#include <mitkBaseProperty.h>
#include <mitkRenderingManager.h>

#include "niftkDnDDisplayEnums.h"
#include "niftkMultiViewerVisibilityManager.h"

class QSpinBox;
class QGridLayout;
class QVBoxLayout;
class QHBoxLayout;
class QPushButton;
class QSpacerItem;
class QLabel;
class QRadioButton;
class QCheckBox;
class QStackedLayout;
class QmitkRenderWindow;
class QLine;
class QButtonGroup;
class QToolButton;
class ctkPopupWidget;


namespace niftk
{

class MultiViewerControls;
class SingleViewerWidget;

/**
 * \class MultiViewerWidget
 * \brief Provides up to 5 x 5 image viewing panes, arranged as rows and columns.
 *
 * This is a large, composite widget, containing a central area that can be
 * used to view images, several controls located at the top of the widget and
 * all the necessary plumbing to make it work. This widget is used as the
 * main editor widget of the MultiViewerEditor.
 *
 * The standard viewer layout is up to 5x5 (but normally, 1x1, 1x2, 1x3 or 2x2)
 * image panes, each showing a single 2D image slice.  This class contains
 * m_MaxRows x m_MaxCols SingleViewerWidget each of which itself wraps
 * a MultiWindowWidget which derives from QmitkStdMultiWidget,
 * meaning that we can actually have up to m_MaxRows x m_MaxCols ortho viewers,
 * including the option for 3D render window.
 */
class NIFTKDNDDISPLAY_EXPORT MultiViewerWidget : public QWidget
{
  Q_OBJECT

public:

  enum ViewerBindingOption
  {
    PositionBinding = 1,
    CursorBinding = 2,
    MagnificationBinding = 4,
    VisibilityBinding = 8,
    WindowLayoutBinding = 16,
    GeometryBinding = 32
  };

  /// \brief Constructor which builds up the controls and layout, and sets the selected viewer to the first (0th),
  /// the default drop type to DNDDISPLAY_DROP_SINGLE, and sets the number of rows and columns to those
  /// specified in the constructor parameter list.
  MultiViewerWidget(mitk::RenderingManager* renderingManager, QWidget* parent = nullptr, Qt::WindowFlags flags = 0);

  /// \brief Destructor, where we assume that all Qt widgets will be destroyed automatically.
  /// Note that we don't create or own the MultiViewerVisibilityManager.
  virtual ~MultiViewerWidget();

  /// \brief As each SingleViewerWidget may have its own rendering manager,
  /// we may have to manually ask each viewer to re-render.
  void RequestUpdateAll();

  /// \brief Gets the display convention of the viewer.
  int GetDisplayConvention() const;

  /// \brief Sets the display convention of the viewer.
  /// This changes the convention of the current viewers and default convention for the new viewers as well.
  void SetDisplayConvention(int displayConvention);

  /// \brief Gets the number of rows of viewers.
  int GetNumberOfRows() const;

  /// \brief Gets the number of columns of viewers.
  int GetNumberOfColumns() const;

  /// \brief Sets the number of viewers.
  void SetViewerNumber(int viewerRows, int viewerColumns);

  /// \brief Gets the viewer in the given row and column.
  /// Indexing starts from 0.
  SingleViewerWidget* GetViewer(int row, int column) const;

  /// \brief Gets the viewer binding options.
  int GetBindingOptions() const;

  /// \brief Sets the viewer binding options.
  void SetBindingOptions(int bindingOptions);

  /// \brief Set the background colour on all contained viewers, and we don't currently provide gradient backgrounds.
  void SetBackgroundColour(QColor backgroundColour);

  /// \brief Sets the default interpolation type, which only takes effect when a node is next dropped into a given window.
  void SetInterpolationType(DnDDisplayInterpolationType interpolationType);

  /// \brief Sets the default window layout (axial, coronal etc.), which only takes effect when a node is next dropped into a given window.
  void SetDefaultWindowLayout(WindowLayout windowLayout);

  /// \brief Sets the default single window layout (axial, coronal etc.), which only takes effect when a node is next dropped into a given window.
  void SetDefaultSingleWindowLayout(WindowLayout windowLayout);

  /// \brief Sets the default multiple window layout (2x2, 3H, 3V etc.), which only takes effect when a node is next dropped into a given window.
  void SetDefaultMultiWindowLayout(WindowLayout layout);

  /// \brief Sets the default drop type checkbox.
  void SetDropType(DnDDisplayDropType dropType);

  /// \brief Sets the visibility flag on the drop type checkboxes.
  void SetShowDropTypeControls(bool visible);

  /// \brief Gets the visibility flag of the show option controls on the control panel.
  bool AreShowOptionsVisible() const;

  /// \brief Sets the visibility flag of the show option controls on the control panel.
  void SetShowOptionsVisible(bool visible);

  /// \brief Gets the visibility flag of the window layout controls on the control panel.
  bool AreWindowLayoutControlsVisible() const;

  /// \brief Sets the visibility flag of the window layout controls on the control panel.
  void SetWindowLayoutControlsVisible(bool visible);

  /// \brief Gets the visibility flag of the multi viewer controls on the control panel.
  bool AreViewerNumberControlsVisible() const;

  /// \brief Sets the visibility flag of the multi viewer controls on the control panel.
  void SetViewerNumberControlsVisible(bool visible);

  /// \brief Sets the visibility flag controlling the Magnification Slider.
  void SetShowMagnificationSlider(bool visible);

  /// \brief Returns the flag indicating whether we show 2D cursors.
  bool IsCursorVisible() const;

  /// \brief Sets the flag controlling the visibility of 2D cursors.
  void SetCursorVisible(bool visibile);

  /// \brief Gets the default visibility of the 2D cursors that is applied
  /// for new viewers if the cursor is not bound across the viewers.
  bool GetCursorDefaultVisibility() const;

  /// \brief Sets the default visibility of the 2D cursors that is applied
  /// for new viewers if the cursor is not bound across the viewers.
  void SetCursorDefaultVisibility(bool visibile);

  /// \brief Tells if the direction annotations are visible.
  bool AreDirectionAnnotationsVisible() const;

  /// \brief Sets the visibility of the direction annotations.
  void SetDirectionAnnotationsVisible(bool visible);

  /// \brief Tells if the position annotation is visible.
  bool IsPositionAnnotationVisible() const;

  /// \brief Sets the visibility of the position annotation.
  void SetPositionAnnotationVisible(bool visible);

  /// \brief Tells if the intensity annotation is visible.
  bool IsIntensityAnnotationVisible() const;

  /// \brief Sets the visibility of the intensity annotation.
  void SetIntensityAnnotationVisible(bool visible);

  /// \brief Tells if the property annotation is visible.
  bool IsPropertyAnnotationVisible() const;

  /// \brief Sets the visibility of the property annotation.
  void SetPropertyAnnotationVisible(bool visible);

  /// \brief Sets the list of properties to display as annotation.
  void SetPropertiesForAnnotation(const QStringList& propertiesForAnnotation);

  /// \brief Sets a flag to determine if we remember viewer positions (slice, timestep, magnification) when we switch the window layout.
  void SetRememberSettingsPerWindowLayout(bool rememberSettingsPerWindowLayout);

  /// \brief Sets the slice slider to be tracking.
  void SetSliceTracking(bool tracking);

  /// \brief Sets the time step slider to be tracking.
  void SetTimeStepTracking(bool tracking);

  /// \brief Sets the magnification slider to be tracking.
  void SetMagnificationTracking(bool tracking);

  /// \brief Most likely called from the MultiViewerEditor to request that the currently selected window changes time step.
  void SetTimeStep(int timeStep);

  /// \brief Most likely called from the MultiViewerEditor to request that the currently selected window switches 3D.
  void SetSelectedWindowTo3D();

  /// \brief Shows or hides the cursor.
  virtual bool ToggleCursorVisibility();

  /// \brief Gets the flag indicating whether this viewer is currently in thumnail mode.
  bool GetThumbnailMode() const;

  /// \brief Sets this viewer to Thumbnail Mode, which means a grid of 5 x 5 viewers, and controls disabled.
  void SetThumbnailMode(bool thumbnailMode);

  /// \brief Returns the orientation from the window layout, or WINDOW_ORIENTATION_UNKNOWN if not known (i.e. if 3D window layout is selected).
  WindowOrientation GetOrientation() const;

  /// \brief Will return the selected viewer or the first viewer if none is selected.
  SingleViewerWidget* GetSelectedViewer() const;

  /// \brief Will return the index of the selected viewer or 0 if none is selected.
//  int GetSelectedViewerIndex() const;

  /// \brief Selects the viewer of the given index.
  /// If the index is out of range, it does not do anything.
//  void SetSelectedViewerIndex(int viewerIndex);

  /// \see mitk::IRenderWindowPart::GetActiveRenderWindow(), where we return the currently selected QmitkRenderWindow.
  virtual QmitkRenderWindow* GetSelectedRenderWindow() const;

  /// \see mitk::IRenderWindowPart::GetRenderWindows(), where we return all render windows for all viewers.
  virtual QHash<QString,QmitkRenderWindow*> GetRenderWindows() const;

  /// \see mitk::IRenderWindowPart::GetRenderWindow(QString), where we return the first render window that matches the given id.
  virtual QmitkRenderWindow* GetRenderWindow(const QString& id) const;

  /// \brief Gets the selected position in world coordinates (mm) in the render window with the given id or
  /// in the currently selected render window if no id is given.
  mitk::Point3D GetSelectedPosition(const QString& id = QString()) const;

  /// \brief Sets the selected position in world coordinates (mm) in the render window with the given id or
  /// in the currently selected render window if no id is given.
  virtual void SetSelectedPosition(const mitk::Point3D& pos, const QString& id = QString());

  /// \see mitk::IRenderWindowPart::IsLinkedNavigationEnabled()
  virtual bool IsLinkedNavigationEnabled() const;

  /// \see mitk::IRenderWindowPart::EnableLinkedNavigation()
  virtual void EnableLinkedNavigation(bool enabled);

  /// \brief Tells if the selected viewer is focused.
  bool IsFocused();

  /// \brief Sets the focus to the selected viewer.
  void SetFocused();

  /// \brief Sets the focus to the viewer of the given index.
  void SetSelectedViewer(int viewerIndex);

  /// \brief Shows the control panel if the mouse pointer is moved over the pin button.
  virtual bool eventFilter(QObject* object, QEvent* event);

signals:

  void BindingOptionsChanged(int bindingOptions);

  void WindowSelected();

protected slots:

  /// \brief Called when the selected slice has been changed through the control panel.
  void OnSelectedSliceControlChanged(int selectedSlice);

  /// \brief Called when the time step has been changed through the control panel.
  void OnTimeStepControlChanged(int timeStep);

  /// \brief Called when the magnification has been changed through the control panel.
  void OnMagnificationControlChanged(double magnification);

  /// \brief Called when the show cursor option has been changed through the control panel.
  void OnCursorVisibilityControlChanged(bool visible);

  /// \brief Called when the show direction annotations option has been changed through the control panel.
  void OnShowDirectionAnnotationsControlsChanged(bool visible);

  /// \brief Called when the show position annotation option has been changed through the control panel.
  void OnShowPositionAnnotationControlsChanged(bool visible);

  /// \brief Called when the show intensity annotation option has been changed through the control panel.
  void OnShowIntensityAnnotationControlsChanged(bool visible);

  /// \brief Called when the show property annotation option has been changed through the control panel.
  void OnShowPropertyAnnotationControlsChanged(bool visible);

  /// \brief Called when the list of properties to be displayed as annotation has been changed through the control panel.
  void OnPropertiesForAnnotationControlsChanged();

  /// \brief Called when the window layout has been changed through the control panel.
  void OnWindowLayoutControlChanged(WindowLayout windowLayout);

  /// \brief Called when the binding of cursors in the render windows of a viewer has been changed through the control panel.
  void OnWindowCursorBindingControlChanged(bool);

  /// \brief Called when the binding of magnifications in the render windows of a viewer has been changed through the control panel.
  void OnWindowMagnificationBindingControlChanged(bool);

  /// \brief Called when the number of viewers has been changed through the control panel.
  void OnViewerNumberControlChanged(int rows, int columns);

  /// \brief Called when the viewer position binding has been changed through the control panel.
  void OnViewerPositionBindingControlChanged(bool bound);

  /// \brief Called when the viewer cursor binding has been changed through the control panel.
  void OnViewerCursorBindingControlChanged(bool bound);

  /// \brief Called when the window layout binding across the viewers has been changed through the control panel.
  void OnViewerWindowLayoutBindingControlChanged(bool bound);

  /// \brief Called when the viewer magnification binding has been changed through the control panel.
  void OnViewerMagnificationBindingControlChanged(bool bound);

  /// \brief Called when the viewer visibility binding has been changed through the control panel.
  void OnViewerVisibilityBindingControlChanged(bool bound);

  /// \brief Called when the viewer geometry binding has been changed through the control panel.
  void OnViewerGeometryBindingControlChanged(bool bound);

  /// \brief Called when the drop type has been changed through the control panel.
  void OnDropTypeControlChanged(DnDDisplayDropType dropType);

  /// \brief Called when the drop accumulation has been changed through the control panel.
  void OnDropAccumulateControlChanged(bool checked);

  /// \brief Called when a window of one of the viewers receives the focus.
  void OnWindowSelected();

  /// \brief Called when the previous viewer is requested to be selected.
  void OnSelectPreviousViewer();

  /// \brief Called when the next viewer is requested to be selected.
  void OnSelectNextViewer();

  /// \brief Called when the viewer with of the given index is requested to be selected.
  void OnSelectViewer(int viewerIndex);

  /// \brief Called when the selected position has changed in a render window of a viewer.
  /// Each of the contained viewers will signal when its slice navigation controllers have changed.
  void OnSelectedPositionChanged(const mitk::Point3D& selectedPosition);

  /// \brief Called when the selected time step has changed in a viewer.
  /// Each of the contained viewers will signal when its slice navigation controllers have changed.
  void OnTimeStepChanged(int timeStep);

  /// \brief Called when the cursor position has changed in a render window because of panning or point selection.
  void OnCursorPositionChanged(WindowOrientation orientation, const mitk::Vector2D& cursorPosition);

  /// \brief Called when the scale factor of a viewer has changed by zooming in one of its render windows.
  void OnScaleFactorChanged(WindowOrientation orientation, double scaleFactor);

  /// \brief Called when the window layout of a viewer has changed.
  void OnWindowLayoutChanged(WindowLayout windowLayout);

  /// \brief Called when the geometry of a viewer has changed.
  void OnTimeGeometryChanged(const mitk::TimeGeometry* timeGeometry);

  /// \brief Called when the cursor position binding has changed in a viewer.
  void OnCursorPositionBindingChanged(bool bound);

  /// \brief Called when the scale factor binding has changed in a viewer.
  void OnScaleFactorBindingChanged(bool bound);

  /// \brief Called when the visibility binding has changed in the visibility manager.
  void OnVisibilityBindingChanged(bool bound);

  /// \brief Called when the show cursor option has been changed in a viewer.
  void OnCursorVisibilityChanged(bool visible);

  /// \brief Called when the show direction annotations option has been changed in a viewer.
  void OnDirectionAnnotationsVisibilityChanged(bool visible);

  /// \brief Called when the show position annotation option has been changed in a viewer.
  void OnPositionAnnotationVisibilityChanged(bool visible);

  /// \brief Called when the show intensity annotation option has been changed in a viewer.
  void OnIntensityAnnotationVisibilityChanged(bool visible);

  /// \brief Called when the show property annotation option has been changed in a viewer.
  void OnPropertyAnnotationVisibilityChanged(bool visible);

  /// \brief Called when the popup widget opens/closes, and used to re-render the viewers.
  void OnPopupOpened(bool opened);

  /// \brief Called when the pin button is toggled.
  void OnPinButtonToggled(bool checked);

private:

  /// \brief Creates a new viewer with the given name.
  /// The name is used to construct the name of the renderers, since the renderers must
  /// have a unique name.
  SingleViewerWidget* CreateViewer(const QString& name);

  /// \brief Force all 2D cursor visibility flags.
  void Update2DCursorVisibility();

  /// \brief Force all visible viewers to match the 'currently selected' viewers geometry.
  void UpdateBoundGeometry(bool isBoundNow);

  /// \brief Selects the render window of the given viewer.
  void SetSelectedRenderWindow(int selectedViewerIndex, QmitkRenderWindow* selectedRenderWindow);

  MultiViewerControls* CreateControlPanel(QWidget* parent);

  // Layouts
  QGridLayout* m_TopLevelLayout;
  QGridLayout* m_LayoutForRenderWindows;

  // Widgets
  QToolButton* m_PinButton;
  ctkPopupWidget* m_PopupWidget;

  int m_DisplayConvention;

  // This determines the maximum number of viewers.
  static const int m_MaxViewerRows = 5;
  static const int m_MaxViewerColumns = 5;

  // All the viewer windows.
  QList<SingleViewerWidget*> m_Viewers;

  // Dependencies, injected via constructor.
  // We don't own them, so don't try to delete them.
  mitk::RenderingManager* m_RenderingManager;

  MultiViewerVisibilityManager::Pointer m_VisibilityManager;

  // Member variables for control purposes.
  int m_SelectedViewerIndex;
  int m_ViewerRows;
  int m_ViewerColumns;
  int m_ViewerRowsInNonThumbnailMode;
  int m_ViewerColumnsInNonThumbnailMode;
  bool m_CursorDefaultVisibility;
  QColor m_BackgroundColour;
  bool m_RememberSettingsPerWindowLayout;
  bool m_ThumbnailMode;
  bool m_LinkedNavigationEnabled;
  double m_Magnification;
  WindowLayout m_SingleWindowLayout;
  WindowLayout m_MultiWindowLayout;

  int m_BindingOptions;

  MultiViewerControls* m_ControlPanel;

  unsigned long m_FocusManagerObserverTag;
};

}

#endif
