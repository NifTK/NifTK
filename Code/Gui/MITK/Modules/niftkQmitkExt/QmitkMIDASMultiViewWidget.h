/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASMULTIVIEWWIDGET_H_
#define QMITKMIDASMULTIVIEWWIDGET_H_

#include "niftkQmitkExtExports.h"

#include <QWidget>
#include <QEvent>
#include "mitkBaseProperty.h"
#include "mitkMIDASKeyPressStateMachine.h"
#include "QmitkMIDASViewEnums.h"
#include "QmitkMIDASSingleViewWidget.h"
#include "QmitkMIDASMultiViewVisibilityManager.h"
#include "QmitkMIDASOrientationWidget.h"
#include "QmitkMIDASSlidersWidget.h"
#include "UpdateMIDASViewingControlsInfo.h"

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

/**
 * \class QmitkMIDASMultiViewWidget
 * \brief Provides a "standard MIDAS" style layout, with up to 5 x 5 image viewing panes, arranged as rows and columns.
 *
 * This is a large, composite widget, containing a central area that can be used to view images, controls around it,
 * and the necessary management logic to manage this view.  This widget is used as the main editor widget of the
 * QmitkMIDASMultiViewEditor.
 *
 * The standard MIDAS layout is up to 5x5 (but normally, 1x1, 1x2, 1x3 or 2x2) image panes, each showing a single
 * 2D image slice.  This class contains m_MaxRows x m_MaxCols QmitkMIDASSingleViewWidget which itself wraps
 * a QmitkMIDASStdMultiViewWidget, meaning that we can actually have up to m_MaxRows x m_MaxCols ortho viewers,
 * including the option for 3D views, which current MIDAS does not have.
 *
 *
 *
 *
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASMultiViewWidget : public QWidget, public mitk::MIDASKeyPressResponder
{
  Q_OBJECT

public:

  /// \brief Constructor which builds up the controls and layout, and sets the selected window to zero, the default drop type to MIDAS_DROP_TYPE_SINGLE, and sets the number of rows and columns to those specified in the parameter list.
  QmitkMIDASMultiViewWidget(
      QmitkMIDASMultiViewVisibilityManager* visibilityManager,
      mitk::DataStorage::Pointer dataStorage,
      int defaultNumberOfRows,
      int defaultNumberOfColumns,
      QWidget* parent = 0, Qt::WindowFlags f = 0);

  /// \brief Destructor, where we assume that all Qt widgets will be destroyed automatically, and we don't create or own the QmitkMIDASMultiViewVisibilityManager, so the remaining thing to do is to disconnect from the mitk::FocusManager.
  virtual ~QmitkMIDASMultiViewWidget();

  /// \brief As each QmitkMIDASSingleViewWidget has its own rendering manager, we have to manually ask each widget to re-render.
  void RequestUpdateAll();

  /// \brief Connects the widget to the FocusManager.
  void Activated();

  /// \brief Disconnects the widget from the FocusManager.
  void Deactivated();

  /// \brief Set the background colour on all contained widgets.
  void SetBackgroundColour(mitk::Color colour);

  /// \brief Sets the default interpolation type, which only takes effect when a node is next dropped into a given window.
  void SetDefaultInterpolationType(MIDASDefaultInterpolationType interpolationType);

  /// \brief Sets the default view, which only takes effect when a node is next dropped into a given window.
  void SetDefaultViewType(MIDASView view);

  /// \brief Sets the default drop type checkbox.
  void SetDropTypeWidget(MIDASDropType dropType);

  /// \brief Sets the visibility flag on the drop type checkboxes.
  void SetShowDropTypeWidgets(bool visible);

  /// \brief Sets the visibility flag on the layout buttons.
  void SetShowLayoutButtons(bool visible);

  /// \brief Sets the flag controlling the visibility of 2D cursors.
  void SetShow2DCursors(bool visibile);

  /// \brief Returns the flag indicating whether we show 2D cursors.
  bool GetShow2DCursors() const;

  /// \brief Sets the flag controlling whether we see studd in the 3D window when in ortho view.
  void SetShow3DViewInOrthoView(bool visible);

  /// \brief Gets the flag controlling whether we see studd in the 3D window when in ortho view.
  bool GetShow3DViewInOrthoView() const;

  /// \brief Sets the visibility flag controlling the Magnification Slider.
  void SetShowMagnificationSlider(bool visible);

  /// \brief Sets a flag to determine if we remember view settings (slice, timestep, magnification) when we switch orientation between axial, coronal, sagittal.
  void SetRememberViewSettingsPerOrientation(bool remember);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window changes time step.
  void SetSelectedTimeStep(int timeStep);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window changes slice number.
  void SetSelectedWindowSliceNumber(int sliceNumber);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window changes magnification.
  void SetSelectedWindowMagnification(int magnification);

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window switches to axial.
  void SetSelectedWindowToAxial();

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window switches to coronal.
  void SetSelectedWindowToSagittal();

  /// \brief Most likely called from the QmitkMIDASMultiViewEditor to request that the currently selected window switches sagittal.
  void SetSelectedWindowToCoronal();

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

  /// \brief Sets whether the interaction is enabled. When false, the slice, camera position, magnification cannot be changed (e.g. for when editing).
  /// \brief Sets this widget to MIDAS Segmentation Mode, which means a single viewer.
  void SetMIDASSegmentationMode(bool enabled);

  /// \brief Gets the flag indicating whether this widget is currently in MIDAS Segmentation Mode, which means a single viewer.
  bool GetMIDASSegmentationMode() const;

  /// \brief Sets this widget to Thumbnail Mode, which means a grid of 5 x 5 viewers, and controls disabled.
  void SetThumbnailMode(bool enabled);

  /// \brief Gets the flag indicating whether this widget is currently in thumnail mode.
  bool GetThumbnailMode() const;

  /// \brief Sets the flag controlling whether we are listening to the navigation controller events.
  void SetNavigationControllerEventListening(bool enabled);

  /// \brief Gets the flag controlling whether we are listening to the navigation controller events.
  bool GetNavigationControllerEventListening() const;

  /// \brief Returns the slice number from the Slice slider.
  int GetSliceNumber() const;

  /// \brief Returns the orientation from the orientation widgets, or MIDAS_ORIENTATION_UNKNOWN if not known (i.e. 3D view selected).
  MIDASOrientation GetOrientation() const;

signals:

  /// \brief Emmitted when an image is dropped and the window selection is changed, so the controls must update, or when mouse wheels cause slice scrolling events.
  void UpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo info);

public slots:

protected slots:

  // Qt slots, connected to Qt GUI elements.
  void OnSliceNumberChanged(int previousSlice, int currentSlice);
  void OnMagnificationFactorChanged(int previousMagnification, int currentMagnification);
  void OnTimeChanged(int previousTime, int currentTime);
  void OnOrientationSelected(bool toggled);
  void On1x1ButtonPressed();
  void On1x2ButtonPressed();
  void On2x1ButtonPressed();
  void On3x1ButtonPressed();
  void On1x3ButtonPressed();
  void On2x2ButtonPressed();
  void On3x2ButtonPressed();
  void On2x3ButtonPressed();
  void On5x5ButtonPressed();
  void OnRowsSliderValueChanged(int);
  void OnColumnsSliderValueChanged(int);
  void OnDropSingleRadioButtonToggled(bool);
  void OnDropMultipleRadioButtonToggled(bool);
  void OnDropThumbnailRadioButtonToggled(bool);
  void OnBindWindowsCheckboxClicked(bool);
  void OnLinkWindowsCheckboxClicked(bool);

  /// \brief When nodes are dropped on one of the contained 25 QmitkRenderWindows, the QmitkMIDASMultiViewVisibilityManager sorts out visibility, so here we just set the focus.
  void OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);

  /// \brief Each of the contained QmitkMIDASSingleViewWidget will signal when it's slice navigation controllers have changed.
  void OnPositionChanged(QmitkMIDASSingleViewWidget *widget, mitk::Point3D voxelLocation, mitk::Point3D millimetreLocation);

private:

  // Called from the QRadioButtons to set the view.
  void SwitchView(MIDASView view);

  // Callback method that gets called by the mitk::FocusManager to indicate the currently focussed window.
  void OnFocusChanged();

  /// \brief Gets the row number, given a viewer index [0, m_MaxRows*m_MaxCols-1]
  unsigned int GetRowFromIndex(unsigned int i);

  /// \brief Gets the column number, given a viewer index [0, m_MaxRows*m_MaxCols-1]
  unsigned int GetColumnFromIndex(unsigned int i);

  /// \brief Gets the index, given a row [0, m_MaxRows-1] and column [0, m_MaxCols-1] number.
  unsigned int GetIndexFromRowAndColumn(unsigned int r, unsigned int c);

  /// \brief Will look at the default view type, and if its axial, coronal, or sagittal, will use that, otherwise, coronal.
  MIDASView GetDefaultOrientationForSegmentation();

  /// \brief Utility method to get a list of viewers to update.
  ///
  /// Normally, if we are in unbound mode, this will be just the currently selected viewer,
  /// but if we are in bind mode, it will be all the visible viewers.
  std::vector<unsigned int> GetViewerIndexesToUpdate(bool doAllVisible, bool isTimeStep);

  /// \brief Main method to change the number of views.
  void SetLayoutSize(unsigned int numberOfRows, unsigned int numberOfColumns, bool isThumbnailMode);

  /// \brief If a particular view is selected, we need to iterate through all views, and make the rest unselected.
  void SetSelectedWindow(unsigned int i);

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

  /// \brief Creates a viewer widget.
  QmitkMIDASSingleViewWidget* CreateSingleViewWidget();

  /// \brief Updates focus manager to auto-focus on the 'current' or 'selected' viewer
  void UpdateFocusManagerToSelectedViewer();

  /// \brief Force all visible viewers to match the 'currently selected' viewers geometry.
  void UpdateBoundGeometry(bool isBound);

  /// \brief Force all 2D cursor visibility flags.
  void Update2DCursorVisibility();

  // Layouts
  QHBoxLayout                                   *m_TopLevelLayout;
  QVBoxLayout                                   *m_LayoutToPutControlsOnTopOfWindows;
  QGridLayout                                   *m_LayoutForRenderWindows;
  QGridLayout                                   *m_LayoutForTopControls;
  QHBoxLayout                                   *m_LayoutForLayoutButtons;
  QHBoxLayout                                   *m_LayoutForRowsAndColumns;
  QHBoxLayout                                   *m_LayoutForOrientation;
  QVBoxLayout                                   *m_LayoutForRightControls;

  // Widgets
  QmitkMIDASOrientationWidget                   *m_MIDASOrientationWidget;
  QmitkMIDASSlidersWidget                       *m_MIDASSlidersWidget;
  QPushButton                                   *m_1x1LayoutButton;
  QPushButton                                   *m_1x2LayoutButton;
  QPushButton                                   *m_2x1LayoutButton;
  QPushButton                                   *m_3x1LayoutButton;
  QPushButton                                   *m_1x3LayoutButton;
  QPushButton                                   *m_2x2LayoutButton;
  QPushButton                                   *m_3x2LayoutButton;
  QPushButton                                   *m_2x3LayoutButton;
  QPushButton                                   *m_5x5LayoutButton;
  QSpinBox                                      *m_RowsSpinBox;
  QLabel                                        *m_RowsLabel;
  QSpinBox                                      *m_ColumnsSpinBox;
  QLabel                                        *m_ColumnsLabel;
  QLabel                                        *m_DropLabel;
  QRadioButton                                  *m_DropSingleRadioButton;
  QRadioButton                                  *m_DropMultipleRadioButton;
  QRadioButton                                  *m_DropThumbnailRadioButton;
  QCheckBox                                     *m_BindWindowsCheckBox;
  QCheckBox                                     *m_LinkWindowsCheckBox;

  // This determines the total number of QmitkMIDASSingleViewWidget windows.
  static const unsigned int m_MaxRows = 5;
  static const unsigned int m_MaxCols = 5;

  // All the viewer windows.
  std::vector<QmitkMIDASSingleViewWidget*>       m_SingleViewWidgets;

  // Dependencies, injected via constructor.
  // We don't own them, so don't delete them.
  QmitkMIDASMultiViewVisibilityManager          *m_VisibilityManager;
  mitk::DataStorage::Pointer                     m_DataStorage;

  // Member variables for control purposes.
  unsigned long                                  m_FocusManagerObserverTag;
  int                                            m_SelectedWindow;
  int                                            m_DefaultNumberOfRows;
  int                                            m_DefaultNumberOfColumns;
  int                                            m_NumberOfRowsInNonThumbnailMode;
  int                                            m_NumberOfColumnsInNonThumbnailMode;
  int                                            m_NumberOfRowsBeforeSegmentationMode;
  int                                            m_NumberOfColumnsBeforeSegmentationMode;
  bool                                           m_InteractionEnabled;
  bool                                           m_Show2DCursors;
  bool                                           m_Show3DViewInOrthoview;
  bool                                           m_IsThumbnailMode;
  bool                                           m_IsMIDASSegmentationMode;
  bool                                           m_NavigationControllerEventListening;
};

#endif /*QMITKMIDASMULTIWIDGET_H_*/
