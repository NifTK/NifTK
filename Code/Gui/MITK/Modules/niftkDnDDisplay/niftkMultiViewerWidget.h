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
#include <mitkMIDASEnums.h>

#include <niftkSingleViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>

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

class niftkMultiViewerControls;

/**
 * \class niftkMultiViewerWidget
 * \brief Provides up to 5 x 5 image viewing panes, arranged as rows and columns.
 *
 * This is a large, composite widget, containing a central area that can be
 * used to view images, several controls located at the top of the widget and
 * all the necessary plumbing to make it work. This widget is used as the
 * main editor widget of the niftkMultiViewerEditor.
 *
 * The standard viewer layout is up to 5x5 (but normally, 1x1, 1x2, 1x3 or 2x2)
 * image panes, each showing a single 2D image slice.  This class contains
 * m_MaxRows x m_MaxCols niftkSingleViewerWidget each of which itself wraps
 * a niftkMultiWindowWidget which derives from QmitkStdMultiWidget,
 * meaning that we can actually have up to m_MaxRows x m_MaxCols ortho viewers,
 * including the option for 3D render window.
 */
class NIFTKDNDDISPLAY_EXPORT niftkMultiViewerWidget : public QWidget
{
  Q_OBJECT

public:

  /// \brief Constructor which builds up the controls and layout, and sets the selected viewer to the first (0th),
  /// the default drop type to DNDDISPLAY_DROP_SINGLE, and sets the number of rows and columns to those
  /// specified in the constructor parameter list.
  niftkMultiViewerWidget(
      niftkMultiViewerVisibilityManager* visibilityManager,
      mitk::RenderingManager* renderingManager,
      mitk::DataStorage::Pointer dataStorage,
      int defaultViewerRows,
      int defaultViewerColumns,
      QWidget* parent = 0, Qt::WindowFlags f = 0);

  /// \brief Destructor, where we assume that all Qt widgets will be destroyed automatically,
  /// and we don't create or own the niftkMultiViewerVisibilityManager, so the remaining thing to
  /// do is to disconnect from the mitk::FocusManager.
  virtual ~niftkMultiViewerWidget();

  /// \brief Used to activate the whole widget.
  void Activated();

  /// \brief Used to de-activate the whole widget.
  void Deactivated();

  /// \brief As each niftkSingleViewerWidget may have its own rendering manager,
  /// we may have to manually ask each viewer to re-render.
  void RequestUpdateAll();

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

  /// \brief Gets the flag controlling whether to see the 3D window in orthogonal (2x2) window layout.
  bool GetShow3DWindowIn2x2WindowLayout() const;

  /// \brief Sets the flag controlling whether to see the 3D window in orthogonal (2x2) window layout.
  void SetShow3DWindowIn2x2WindowLayout(bool enabled);

  /// \brief Sets a flag to determine if we remember viewer positions (slice, timestep, magnification) when we switch the window layout.
  void SetRememberSettingsPerWindowLayout(bool rememberSettingsPerWindowLayout);

  /// \brief Sets the slice index slider to be tracking.
  void SetSliceIndexTracking(bool tracking);

  /// \brief Sets the time step slider to be tracking.
  void SetTimeStepTracking(bool tracking);

  /// \brief Sets the magnification slider to be tracking.
  void SetMagnificationTracking(bool tracking);

  /// \brief Most likely called from the niftkMultiViewerEditor to request that the currently selected window changes time step.
  void SetSelectedTimeStep(int timeStep);

  /// \brief Most likely called from the niftkMultiViewerEditor to request that the currently selected window changes slice index.
  void SetSelectedWindowSliceIndex(int sliceIndex);

  /// \brief Most likely called from the niftkMultiViewerEditor to request that the currently selected window switches 3D.
  void SetSelectedWindowTo3D();

  /// \brief Shows or hides the cursor.
  virtual bool ToggleCursorVisibility();

  /// \brief Sets whether the interaction is enabled, and a single viewer.
  void SetSegmentationModeEnabled(bool enabled);

  /// \brief Gets the flag indicating whether this viewer is currently in segmentation mode, which means a single viewer.
  bool IsSegmentationModeEnabled() const;

  /// \brief Sets this viewer to Thumbnail Mode, which means a grid of 5 x 5 viewers, and controls disabled.
  void SetThumbnailMode(bool enabled);

  /// \brief Gets the flag indicating whether this viewer is currently in thumnail mode.
  bool GetThumbnailMode() const;

  /// \brief Returns the orientation from the window layout, or MIDAS_ORIENTATION_UNKNOWN if not known (i.e. if 3D window layout is selected).
  MIDASOrientation GetOrientation() const;

  // Callback method that gets called by the mitk::FocusManager to indicate the currently focused window.
  void OnFocusChanged();

  /// \brief Will return the selected viewer or the first viewer if none is selected.
  niftkSingleViewerWidget* GetSelectedViewer() const;

  /**
   * \see mitk::IRenderWindowPart::GetActiveRenderWindow(), where we return the currently selected QmitkRenderWindow.
   */
  virtual QmitkRenderWindow* GetSelectedRenderWindow() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindows(), where we return all render windows for all viewers.
   */
  virtual QHash<QString,QmitkRenderWindow*> GetRenderWindows() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindow(QString), where we return the first render window that matches the given id.
   */
  virtual QmitkRenderWindow* GetRenderWindow(const QString& id) const;

  /**
   * \brief Gets the selected position in world coordinates (mm) in the render window with the given id or
   * in the currently selected render window if no id is given.
   */
  mitk::Point3D GetSelectedPosition(const QString& id = QString()) const;

  /**
   * \brief Sets the selected position in world coordinates (mm) in the render window with the given id or
   * in the currently selected render window if no id is given.
   */
  virtual void SetSelectedPosition(const mitk::Point3D& pos, const QString& id = QString());

  /**
   * \see mitk::IRenderWindowPart::EnableLinkedNavigation()
   */
  virtual void EnableLinkedNavigation(bool enable);

  /**
   * \see mitk::IRenderWindowPart::IsLinkedNavigationEnabled()
   */
  virtual bool IsLinkedNavigationEnabled() const;

  /**
   * \brief To be called from the editor, to set the focus to the currently selected
   * viewer, or the first viewer.
   */
  virtual void SetFocus();

  /// \brief Shows the control panel if the mouse pointer is moved over the pin button.
  virtual bool eventFilter(QObject* object, QEvent* event);

signals:

protected slots:

  /// \brief Called when the slice index has been changed through the control panel.
  void OnSliceIndexChanged(int sliceIndex);

  /// \brief Called when the time step has been changed through the control panel.
  void OnTimeStepChanged(int timeStep);

  /// \brief Called when the magnification has been changed through the control panel.
  void OnMagnificationChanged(double magnification);

  /// \brief Called when the show cursor option has been changed through the control panel.
  void OnCursorVisibilityChanged(bool visible);

  /// \brief Called when the show direction annotations option has been changed through the control panel.
  void OnShowDirectionAnnotationsChanged(bool visible);

  /// \brief Called when the show 3D window option has been changed through the control panel.
  void OnShow3DWindowChanged(bool visible);

  /// \brief Called when the window layout has been changed through the control panel.
  void OnWindowLayoutChanged(WindowLayout windowLayout);

  /// \brief Called when the binding of cursors in the render windows of a viewer has been changed through the control panel.
  void OnWindowCursorBindingChanged(bool);

  /// \brief Called when the binding of magnifications in the render windows of a viewer has been changed through the control panel.
  void OnWindowMagnificationBindingChanged(bool);

  /// \brief Called when the number of viewers has been changed through the control panel.
  void OnViewerNumberChanged(int rows, int columns);

  /// \brief Called when the viewer position binding has been changed through the control panel.
  void OnViewerPositionBindingChanged();

  /// \brief Called when the viewer cursor binding has been changed through the control panel.
  void OnViewerCursorBindingChanged();

  /// \brief Called when the window layout binding across the viewers has been changed through the control panel.
  void OnViewerWindowLayoutBindingChanged();

  /// \brief Called when the viewer magnification binding has been changed through the control panel.
  void OnViewerMagnificationBindingChanged();

  /// \brief Called when the viewer geometry binding has been changed through the control panel.
  void OnViewerGeometryBindingChanged();

  /// \brief Called when the drop type has been changed through the control panel.
  void OnDropTypeChanged(DnDDisplayDropType dropType);

  /// \brief Called when the drop accumulation has been changed through the control panel.
  void OnDropAccumulateChanged(bool checked);

  /// \brief When nodes are dropped on one of the contained 25 QmitkRenderWindows, the niftkMultiViewerVisibilityManager sorts out visibility, so here we just set the focus.
  void OnNodesDropped(QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes);

  /// \brief Called when the selected position has changed in a render window of a viewer.
  /// Each of the contained viewers will signal when its slice navigation controllers have changed.
  void OnSelectedPositionChanged(niftkSingleViewerWidget* viewer, QmitkRenderWindow* renderWindow, int sliceIndex);

  /// \brief Called when the cursor position has changed in a render window because of panning or point selection.
  void OnCursorPositionChanged(niftkSingleViewerWidget* viewer, const mitk::Vector3D& cursorPosition);

  /// \brief Called when the scale factor of a viewer has changed by zooming in one of its render windows.
  void OnScaleFactorChanged(niftkSingleViewerWidget* viewer, double scaleFactor);

  /// \brief Called when the window layout of a viewer has changed.
  void OnWindowLayoutChanged(niftkSingleViewerWidget* viewer, WindowLayout windowLayout);

  /// \brief Called when the geometry of a viewer has changed.
  void OnGeometryChanged(niftkSingleViewerWidget* viewer, mitk::TimeGeometry* geometry);

  /// \brief Called when the show cursor option has been changed in a viewer.
  void OnCursorVisibilityChanged(niftkSingleViewerWidget* viewer, bool visible);

  /// \brief Called when the popup widget opens/closes, and used to re-render the viewers.
  void OnPopupOpened(bool opened);

  /// \brief Called when the pin button is toggled.
  void OnPinButtonToggled(bool checked);

protected:

private:

  /// \brief Will return the index of the selected viewer or 0 if none is selected.
  int GetSelectedViewerIndex() const;

  /// \brief Gets the row number, given a viewer index [0, m_MaxRows * m_MaxCols - 1]
  int GetViewerRowFromIndex(int index) const;

  /// \brief Gets the column number, given a viewer index [0, m_MaxRows * m_MaxCols - 1]
  int GetViewerColumnFromIndex(int index) const;

  /// \brief Gets the index, given a row [0, m_MaxRows - 1] and column [0, m_MaxCols - 1] number.
  int GetViewerIndexFromRowAndColumn(int row, int column) const;

  /// \brief Will look at the default window layout, and if its axial, coronal, or sagittal, will use that, otherwise, coronal.
  WindowLayout GetDefaultWindowLayoutForSegmentation() const;

  /// \brief Main method to change the number of viewers.
  void SetViewerNumber(int numberOfRows, int numberOfColumns, bool isThumbnailMode);

  // Called from the QRadioButtons to set the layout.
  void SetLayout(WindowLayout windowLayout);

  /// \brief If a particular viewer is selected, we need to iterate through all viewers, and make the rest unselected.
  void SetSelectedViewerByIndex(int index);

  /// \brief Creates a new viewer.
  niftkSingleViewerWidget* CreateViewer();

  /// \brief Force all 2D cursor visibility flags.
  void Update2DCursorVisibility();

  /// \brief Updates focus manager to auto-focus on the 'currently selected' viewer.
  void UpdateFocusManagerToSelectedViewer();

  /// \brief Force all visible viewers to match the 'currently selected' viewers geometry.
  void UpdateBoundGeometry(bool isBoundNow);

  /// \brief Force all visible viewers to match the 'currently selected' viewers magnification.
  void UpdateBoundMagnification();

  /// \brief Selects the render window of the given viewer.
  void SetSelectedRenderWindow(int selectedViewerIndex, QmitkRenderWindow* selectedRenderWindow);

  /// \brief Sets the flag controlling whether we are listening to the navigation controller events.
  void SetNavigationControllerEventListening(bool enabled);

  /// \brief Gets the flag controlling whether we are listening to the navigation controller events.
  bool GetNavigationControllerEventListening() const;

  niftkMultiViewerControls* CreateControlPanel(QWidget* parent);

  // Layouts
  QGridLayout* m_TopLevelLayout;
  QGridLayout* m_LayoutForRenderWindows;

  // Widgets
  QToolButton* m_PinButton;
  ctkPopupWidget* m_PopupWidget;

  // This determines the maximum number of viewers.
  static const int m_MaxViewerRows = 5;
  static const int m_MaxViewerColumns = 5;

  // All the viewer windows.
  QList<niftkSingleViewerWidget*> m_Viewers;

  // Dependencies, injected via constructor.
  // We don't own them, so don't try to delete them.
  niftkMultiViewerVisibilityManager* m_VisibilityManager;
  mitk::DataStorage* m_DataStorage;
  mitk::RenderingManager* m_RenderingManager;

  // Member variables for control purposes.
  unsigned long m_FocusManagerObserverTag;
  int m_SelectedViewerIndex;
  int m_DefaultViewerRows;
  int m_DefaultViewerColumns;
  int m_ViewerRowsInNonThumbnailMode;
  int m_ViewerColumnsInNonThumbnailMode;
  int m_ViewerRowsBeforeSegmentationMode;
  int m_ViewerColumnsBeforeSegmentationMode;
  bool m_Show3DWindowIn2x2WindowLayout;
  bool m_CursorDefaultVisibility;
  QColor m_BackgroundColour;
  bool m_RememberSettingsPerWindowLayout;
  bool m_IsThumbnailMode;
  bool m_SegmentationModeEnabled;
  bool m_NavigationControllerEventListening;
  double m_Magnification;
  WindowLayout m_SingleWindowLayout;
  WindowLayout m_MultiWindowLayout;

  niftkMultiViewerControls* m_ControlPanel;
};

#endif
