/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMIDASMultiViewWidget_h
#define QmitkMIDASMultiViewWidget_h

#include <uk_ac_ucl_cmic_midaseditor_Export.h>

#include <QColor>
#include <QEvent>
#include <QList>
#include <QWidget>

#include <mitkBaseProperty.h>
#include <mitkMIDASViewKeyPressResponder.h>
#include <mitkRenderingManager.h>
#include <mitkMIDASEnums.h>

#include <QmitkMIDASSingleViewWidget.h>
#include <QmitkMIDASMultiViewVisibilityManager.h>
#include <QmitkMIDASOrientationWidget.h>
#include <QmitkMIDASBindWidget.h>
#include <QmitkMIDASSlidersWidget.h>

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

/**
 * \class QmitkMIDASMultiViewWidget
 * \brief Provides a "standard MIDAS" style layout, with up to 5 x 5 image
 * viewing panes, arranged as rows and columns.
 *
 * This is a large, composite widget, containing a central area that can be
 * used to view images, several controls located at the top of the widget and
 * all the necessary plumbing to make it work. This widget is used as the
 * main editor widget of the QmitkMIDASMultiViewEditor.
 *
 * The standard MIDAS layout is up to 5x5 (but normally, 1x1, 1x2, 1x3 or 2x2)
 * image panes, each showing a single 2D image slice.  This class contains
 * m_MaxRows x m_MaxCols QmitkMIDASSingleViewWidget each of which itself wraps
 * a QmitkMIDASStdMultiViewWidget which derives from QmitkStdMultiWidget,
 * meaning that we can actually have up to m_MaxRows x m_MaxCols ortho viewers,
 * including the option for 3D views, which current MIDAS does not have.
 */
class MIDASEDITOR_EXPORT QmitkMIDASMultiViewWidget : public QWidget, public mitk::MIDASViewKeyPressResponder
{
  Q_OBJECT

public:

  /// \brief Constructor which builds up the controls and layout, and sets the selected window to zero,
  /// the default drop type to MIDAS_DROP_TYPE_SINGLE, and sets the number of rows and columns to those
  /// specified in the constructor parameter list.
  QmitkMIDASMultiViewWidget(
      QmitkMIDASMultiViewVisibilityManager* visibilityManager,
      mitk::RenderingManager* renderingManager,
      mitk::DataStorage::Pointer dataStorage,
      int defaultNumberOfRows,
      int defaultNumberOfColumns,
      QWidget* parent = 0, Qt::WindowFlags f = 0);

  /// \brief Destructor, where we assume that all Qt widgets will be destroyed automatically,
  /// and we don't create or own the QmitkMIDASMultiViewVisibilityManager, so the remaining thing to
  /// do is to disconnect from the mitk::FocusManager.
  virtual ~QmitkMIDASMultiViewWidget();

  /// \brief Used to activate the whole widget.
  void Activated();

  /// \brief Used to de-activate the whole widget.
  void Deactivated();

  /// \brief As each QmitkMIDASSingleViewWidget may have its own rendering manager,
  /// we may have to manually ask each widget to re-render.
  void RequestUpdateAll();

  /// \brief Set the background colour on all contained widgets, and we don't currently provide gradient backgrounds.
  void SetBackgroundColour(QColor backgroundColour);

  /// \brief Sets the default interpolation type, which only takes effect when a node is next dropped into a given window.
  void SetDefaultInterpolationType(MIDASDefaultInterpolationType interpolationType);

  /// \brief Sets the default view (axial, coronal etc.), which only takes effect when a node is next dropped into a given window.
  void SetDefaultViewType(MIDASView midasView);

  /// \brief Sets the default single window layout (axial, coronal etc.), which only takes effect when a node is next dropped into a given window.
  void SetDefaultSingleWindowLayout(MIDASView midasView);

  /// \brief Sets the default multiple window layout (2x2, 3H, 3V etc.), which only takes effect when a node is next dropped into a given window.
  void SetDefaultMultiWindowLayout(MIDASView midasView);

  /// \brief Sets the default drop type checkbox.
  void SetDropTypeWidget(MIDASDropType dropType);

  /// \brief Sets the visibility flag on the drop type checkboxes.
  void SetShowDropTypeWidgets(bool visible);

  /// \brief Sets the visibility flag on the layout buttons.
  void SetShowLayoutButtons(bool visible);

  /// \brief Returns the flag indicating whether we show 2D cursors.
  bool GetShow2DCursors() const;

  /// \brief Sets the flag controlling the visibility of 2D cursors.
  void SetShow2DCursors(bool visibile);

  /// \brief Gets the flag controlling whether we see the 3D window in orthogonal (2x2) view.
  bool GetShow3DWindowInOrthoView() const;

  /// \brief Sets the flag controlling whether we see the 3D window in orthogonal (2x2) view.
  void SetShow3DWindowInOrthoView(bool enabled);

  /// \brief Sets the visibility flag controlling the Magnification Slider.
  void SetShowMagnificationSlider(bool visible);

  /// \brief Sets a flag to determine if we remember view settings (slice, timestep, magnification) when we switch orientation between axial, coronal, sagittal.
  void SetRememberViewSettingsPerOrientation(bool rememberViewSettingsPerOrientation);

  /// \brief Sets the slice select slider to be tracking.
  void SetSliceSelectTracking(bool isTracking);

  /// \brief Sets the magnification select slider to be tracking.
  void SetMagnificationSelectTracking(bool isTracking);

  /// \brief Sets the time select slider to be tracking.
  void SetTimeSelectTracking(bool isTracking);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window changes time step.
  void SetSelectedTimeStep(int timeStep);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window changes slice number.
  void SetSelectedWindowSliceNumber(int sliceNumber);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window changes magnification.
  void SetSelectedWindowMagnification(double magnification);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window switches to axial.
  void SetSelectedWindowToAxial();

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window switches to coronal.
  void SetSelectedWindowToSagittal();

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window switches sagittal.
  void SetSelectedWindowToCoronal();

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window switches 3D.
  void SetSelectedWindowTo3D();

  /// \brief Move anterior a slice.
  bool MoveAnterior();

  /// \brief Move posterior a slice.
  bool MovePosterior();

  /// \brief Switch to Axial.
  bool SwitchToAxial();

  /// \brief Switch to Sagittal.
  bool SwitchToSagittal();

  /// \brief Switch to Coronal.
  bool SwitchToCoronal();

  /// \brief Switch to 3D.
  bool SwitchTo3D();

  /// \brief Switch the from single window to multiple windows or back
  bool ToggleMultiWindowLayout();

  /// \brief Sets whether the interaction is enabled, and a single viewer.
  void SetMIDASSegmentationMode(bool enabled);

  /// \brief Gets the flag indicating whether this widget is currently in MIDAS Segmentation Mode, which means a single viewer.
  bool GetMIDASSegmentationMode() const;

  /// \brief Sets this widget to Thumbnail Mode, which means a grid of 5 x 5 viewers, and controls disabled.
  void SetThumbnailMode(bool enabled);

  /// \brief Gets the flag indicating whether this widget is currently in thumnail mode.
  bool GetThumbnailMode() const;

  /// \brief Returns the slice number from the Slice slider.
  int GetSliceNumber() const;

  /// \brief Returns the orientation from the orientation widgets, or MIDAS_ORIENTATION_UNKNOWN if not known (i.e. 3D view selected).
  MIDASOrientation GetOrientation() const;

  // Callback method that gets called by the mitk::FocusManager to indicate the currently focused window.
  void OnFocusChanged();

  /**
   * \see mitk::IRenderWindowPart::GetActiveRenderWindow(), where we return the currently selected QmitkRenderWindow.
   */
  virtual QmitkRenderWindow* GetSelectedRenderWindow() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindows(), where we return all render windows for all widgets.
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
   * widget, or the first widget.
   */
  virtual void SetFocus();

  virtual bool eventFilter(QObject* object, QEvent* event);

signals:

protected slots:

  // Qt slots, connected to Qt GUI elements.
  void OnSliceNumberChanged(double sliceNumber);
  void OnMagnificationFactorChanged(double magnificationFactor);
  void OnTimeChanged(double timeStep);
  void On1x1ButtonPressed();
  void On1x2ButtonPressed();
  void On1x3ButtonPressed();
  void On2x2ButtonPressed();
  void OnRowsSliderValueChanged(int);
  void OnColumnsSliderValueChanged(int);
  void OnOrientationSelected(MIDASView midasView);
  void OnShow2DCursorsCheckBoxToggled(bool);
  void OnDropSingleRadioButtonToggled(bool);
  void OnDropMultipleRadioButtonToggled(bool);
  void OnDropThumbnailRadioButtonToggled(bool);
  void OnDropAccumulateStateChanged(int);
  void OnBindTypeChanged();

  /// \brief When nodes are dropped on one of the contained 25 QmitkRenderWindows, the QmitkMIDASMultiViewVisibilityManager sorts out visibility, so here we just set the focus.
  void OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);

  /// \brief Each of the contained QmitkMIDASSingleViewWidget will signal when it's slice navigation controllers have changed.
  void OnSelectedPositionChanged(QmitkMIDASSingleViewWidget *widget, QmitkRenderWindow* window, int sliceNumber);

  /// \brief Called when the cursor position is changed on a render window because of panning.
  void OnCursorPositionChanged(QmitkMIDASSingleViewWidget *widget, const mitk::Vector3D& cursorPosition);

  /// \brief Called when the magnification is changed by zooming in a renderer window.
  void OnMagnificationFactorChanged(QmitkMIDASSingleViewWidget *view, double magnificationFactor);

  /// \brief Called when the popup widget opens/closes, and used to re-render the widgets.
  void OnPopupOpened(bool opened);

  /// \brief Called when the pin button is toggled.
  void OnPinButtonToggled(bool checked);

protected:

private:

  /// \brief Utility method to get a list of views to update.
  /// \param doAllVisible if true will ensure the returned vector contains all visible render window, and if false will return just the currently selected window.
  /// \return vector of integers corresponding to widget numbers.
  QList<int> GetViewIndexesToUpdate(bool doAllVisible) const;

  /// \brief Will return m_SelectedView, or if m_SelectedView < 0 will return 0.
  int GetSelectedViewIndex() const;

  /// \brief Gets the row number, given a view index [0, m_MaxRows*m_MaxCols-1]
  int GetRowFromIndex(int i) const;

  /// \brief Gets the column number, given a viewer index [0, m_MaxRows*m_MaxCols-1]
  int GetColumnFromIndex(int i) const;

  /// \brief Gets the index, given a row [0, m_MaxRows-1] and column [0, m_MaxCols-1] number.
 int GetIndexFromRowAndColumn(int r, int c) const;

  /// \brief Will look at the default view type, and if its axial, coronal, or sagittal, will use that, otherwise, coronal.
  MIDASView GetDefaultOrientationForSegmentation() const;

  /// \brief Main method to change the number of views.
  void SetLayoutSize(int numberOfRows, int numberOfColumns, bool isThumbnailMode);

  // Called from the QRadioButtons to set the view.
  void SwitchMIDASView(MIDASView midasView);

  /// \brief If a particular view is selected, we need to iterate through all views, and make the rest unselected.
  void SetSelectedViewIndex(int i);

  /// \brief Method to enable, or disable all widgets, for use when GUI is first started, or the whole screen enabled, disabled.
  void EnableWidgets(bool enabled);

  /// \brief Enables/Disables drop type widgets.
  void EnableDropTypeWidgets(bool enabled);

  /// \brief Enables/Disables slider widgets.
  void EnableSliderWidgets(bool enabled);

  /// \brief Enables/Disables widgets to control layout.
  void EnableLayoutWidgets(bool enabled);

  /// \brief Enables/Disables widgets to control binding / linking.
  void EnableBindWidgets(bool enabled);

  /// \brief Enables/Disables widgets to control orientation.
  void EnableOrientationWidgets(bool enabled);

  /// \brief Creates a view widget.
  QmitkMIDASSingleViewWidget* CreateSingleViewWidget();

  /// \brief Force all 2D cursor visibility flags.
  void Update2DCursorVisibility();

  /// \brief Updates focus manager to auto-focus on the 'currently selected' view
  void UpdateFocusManagerToSelectedView();

  /// \brief Force all visible viewers to match the 'currently selected' viewers geometry.
  void UpdateBoundGeometry(bool isBoundNow);

  /// \brief Force all visible viewers to match the 'currently selected' viewers magnification.
  void UpdateBoundMagnification();

  void SwitchWindows(int selectedViewIndex, QmitkRenderWindow *selectedWindow);

  /// \brief Sets the flag controlling whether we are listening to the navigation controller events.
  void SetNavigationControllerEventListening(bool enabled);

  /// \brief Gets the flag controlling whether we are listening to the navigation controller events.
  bool GetNavigationControllerEventListening() const;

  /// \brief Used to move either anterior/posterior by a certain number of slices.
  bool MoveAnteriorPosterior(int slices);

  // Layouts
  QHBoxLayout                                   *m_TopLevelLayout;
  QGridLayout                                   *m_LayoutToPutControlsOnTopOfWindows;
  QHBoxLayout                                   *m_LayoutForGroupingControls;
  QGridLayout                                   *m_LayoutForTopControls;
  QGridLayout                                   *m_LayoutForLayoutWidgets;
  QVBoxLayout                                   *m_LayoutForDropWidgets;
  QGridLayout                                   *m_LayoutForRenderWindows;

  // Widgets
  QmitkMIDASOrientationWidget                   *m_MIDASOrientationWidget;
  QCheckBox                                     *m_Show2DCursorsCheckBox;
  QmitkMIDASSlidersWidget                       *m_MIDASSlidersWidget;
  QmitkMIDASBindWidget                          *m_MIDASBindWidget;
  QPushButton                                   *m_1x1LayoutButton;
  QPushButton                                   *m_1x2LayoutButton;
  QPushButton                                   *m_1x3LayoutButton;
  QPushButton                                   *m_2x2LayoutButton;
  QSpinBox                                      *m_RowsSpinBox;
  QLabel                                        *m_RowsLabel;
  QSpinBox                                      *m_ColumnsSpinBox;
  QLabel                                        *m_ColumnsLabel;
  QRadioButton                                  *m_DropSingleRadioButton;
  QRadioButton                                  *m_DropMultipleRadioButton;
  QRadioButton                                  *m_DropThumbnailRadioButton;
  QButtonGroup                                  *m_DropButtonGroup;
  QCheckBox                                     *m_DropAccumulateCheckBox;
  QToolButton                                   *m_PinButton;
  QWidget                                       *m_ControlWidget;
  QVBoxLayout                                   *m_ControlWidgetLayout;
  ctkPopupWidget                                *m_PopupWidget;
  QFrame                                        *m_ControlsContainerWidget;

  // This determines the maximum number of QmitkMIDASSingleViewWidget windows.
  static const int m_MaxRows = 5;
  static const int m_MaxCols = 5;

  // All the viewer windows.
  QList<QmitkMIDASSingleViewWidget*>             m_SingleViewWidgets;

  // Dependencies, injected via constructor.
  // We don't own them, so don't try to delete them.
  QmitkMIDASMultiViewVisibilityManager          *m_VisibilityManager;
  mitk::DataStorage                             *m_DataStorage;
  mitk::RenderingManager                        *m_RenderingManager;

  // Member variables for control purposes.
  unsigned long                                  m_FocusManagerObserverTag;
  int                                            m_SelectedViewIndex;
  int                                            m_DefaultNumberOfRows;
  int                                            m_DefaultNumberOfColumns;
  int                                            m_NumberOfRowsInNonThumbnailMode;
  int                                            m_NumberOfColumnsInNonThumbnailMode;
  int                                            m_NumberOfRowsBeforeSegmentationMode;
  int                                            m_NumberOfColumnsBeforeSegmentationMode;
  bool                                           m_Show2DCursors;
  bool                                           m_Show3DWindowInOrthoView;
  QColor                                         m_BackgroundColour;
  bool                                           m_RememberViewSettingsPerOrientation;
  bool                                           m_IsThumbnailMode;
  bool                                           m_IsMIDASSegmentationMode;
  bool                                           m_NavigationControllerEventListening;
  double                                         m_PreviousMagnificationFactor;
  MIDASView                                      m_SingleWindowLayout;
  MIDASView                                      m_MultiWindowLayout;
};

#endif
