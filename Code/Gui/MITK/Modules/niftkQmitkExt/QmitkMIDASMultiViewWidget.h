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

#include "QmitkMIDASMultiViewVisibilityManager.h"

class QSpinBox;
class QGridLayout;
class QVBoxLayout;
class QHBoxLayout;
class QPushButton;
class QSpacerItem;
class QLabel;
class QRadioButton;
class QCheckBox;
class QmitkMIDASSingleViewWidget;
class QmitkMIDASRenderWindow;

/**
 * \class UpdateMIDASViewingControlsInfo
 * \brief Simply to pass the current slice, magnification and orientation to QmitkMIDASMultiViewEditor.
 */
struct UpdateMIDASViewingControlsInfo
{
  int currentTime;
  int currentSlice;
  int currentMagnification;
  bool isAxial;
  bool isSagittal;
  bool isCoronal;
};

/**
 * \class UpdateMIDASViewingControlsRangeInfo
 * \brief Simply to pass slice and magnification range information to QmitkMIDASMultiViewEditor.
 */
struct UpdateMIDASViewingControlsRangeInfo
{
  int minTime;
  int maxTime;
  int minSlice;
  int maxSlice;
  int minMagnification;
  int maxMagnification;
};

/**
 * \class QmitkMIDASMultiViewWidget
 * \brief Provides a "standard MIDAS" style layout, with up to 5 x 5 image viewing panes, arranged as rows and columns.
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASMultiViewWidget : public QWidget
{
  Q_OBJECT

public:

  /// \brief Constructor which builds up the controls and layout, and sets the selected window to zero, the default drop type to QmitkMIDASMultiViewVisibilityManager::MIDAS_DROP_TYPE_SINGLE, and sets the number of rows and columns to those specified in the parameter list.
  QmitkMIDASMultiViewWidget(
      QmitkMIDASMultiViewVisibilityManager* visibilityManager,
      int defaultNumberOfRows,
      int defaultNumberOfColumns,
      QWidget* parent = 0, Qt::WindowFlags f = 0);

  /// \brief Destructor, where we assume that all Qt widgets will be destroyed automatically, and we don't create or own the QmitkMIDASMultiViewVisibilityManager, so the remaining thing to do is to disconnect from the mitk::FocusManager.
  virtual ~QmitkMIDASMultiViewWidget();

  /// \brief Set the background colour on all contained widgets.
  void SetBackgroundColour(mitk::Color colour);

  /// \brief Sets the default interpolation type, which only takes effect when a node is next dropped into a given window.
  void SetDefaultInterpolationType(QmitkMIDASMultiViewVisibilityManager::MIDASDefaultInterpolationType interpolationType);

  /// \brief Sets the default orientation, which only takes effect when a node is next dropped into a given window.
  void SetDefaultOrientationType(QmitkMIDASMultiViewVisibilityManager::MIDASDefaultOrientationType interpolationType);

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

  /// \brief As each QmitkMIDASSingleViewWidget has its own rendering manager, we have to manually ask each widget to re-render.
  void RequestUpdateAll();

  /// \brief As each QmitkMIDASSingleViewWidget has its own rendering manager, we have to manually ask each widget to re-render.
  void ForceUpdateAll();

public slots:

signals:

  /// \brief Emmitted when an image is dropped and the window selection is change, so the controls must update.
  void UpdateMIDASViewingControlsRange(UpdateMIDASViewingControlsRangeInfo info);

  /// \brief Emmitted when an image is dropped and the window selection is changed, so the controls must update.
  void UpdateMIDASViewingControlsValues(UpdateMIDASViewingControlsInfo info);

protected:

    // overloaded paint handler
    virtual void paintEvent(QPaintEvent* event);

protected slots:

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

  /// \brief When nodes are dropped on one of the contained 25 QmitkMIDASRenderWindows, the QmitkMIDASMultiViewVisibilityManager sorts out visibility, so here we just set the focus.
  void OnNodesDropped(QmitkMIDASRenderWindow *window, std::vector<mitk::DataNode*> nodes);

private:

  static const unsigned int m_MaxRows = 5;
  static const unsigned int m_MaxCols = 5;

  // Callback method that gets called by the mitk::FocusManager and is responsible for signalling the slice number, magnification, orientation.
  void OnFocusChanged();

  // Internal method that takes the currently selected window, and broadcasts the current slice, mangification and orientation information.
  void PublishNavigationSettings();

  unsigned int GetRowFromIndex(unsigned int i);
  unsigned int GetColumnFromIndex(unsigned int i);
  unsigned int GetIndexFromRowAndColumn(unsigned int r, unsigned int c);

  void SetLayoutSize(unsigned int numberOfRows, unsigned int numberOfColumns, bool isThumbnailMode);
  void SetSelectedWindow(unsigned int i);
  void EnableWidgetsForThumbnailMode(bool isThumbnailMode);
  void GetStartStopIndexForIteration(unsigned int &start, unsigned int &stop);
  void SetWindowsToOrientation(QmitkMIDASSingleViewWidget::MIDASViewOrientation orientation);

  QGridLayout                                   *m_LayoutForRenderWindows;
  QGridLayout                                   *m_LayoutForLayoutButtons;
  QGridLayout                                   *m_LayoutForDropRadioButtons;
  QHBoxLayout                                   *m_LayoutForTopControls;
  QVBoxLayout                                   *m_LayoutToPutButtonsOnTopOfWindows;
  QHBoxLayout                                   *m_TopLevelLayout;

  QSpacerItem                                   *m_HorizontalSpacerBetweenRadioButtonsAndBindButton;
  QSpacerItem                                   *m_HorizontalSpacerBetweenBindButtonAndLayoutButtons;
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

  QmitkMIDASMultiViewVisibilityManager          *m_VisibilityManager; // We don't own this, so don't delete it.
  std::vector<QmitkMIDASSingleViewWidget*>       m_SingleViewWidgets; // Should be automatically destroyed by Qt.

  unsigned long                                  m_FocusManagerObserverTag;
  int                                            m_SelectedWindow;
  int                                            m_DefaultNumberOfRows;
  int                                            m_DefaultNumberOfColumns;
  int                                            m_NumberOfRowsInNonThumbnailMode;
  int                                            m_NumberOfColumnsInNonThumbnailMode;
};
#endif /*QMITKMIDASMULTIWIDGET_H_*/
