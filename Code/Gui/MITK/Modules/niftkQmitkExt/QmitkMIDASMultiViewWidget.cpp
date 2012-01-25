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

#include "QmitkMIDASMultiViewWidget.h"
#include <QPushButton>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSpacerItem>
#include <QSize>
#include <QSpinBox>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDragLeaveEvent>
#include <QDropEvent>
#include <QRadioButton>
#include <QCheckBox>
#include <QLabel>
#include <QDebug>
#include <QMessageBox>
#include "mitkFocusManager.h"
#include "mitkGlobalInteraction.h"
#include "mitkTimeSlicedGeometry.h"
#include "QmitkMIDASRenderWindow.h"
#include "QmitkMIDASSingleViewWidget.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"

QmitkMIDASMultiViewWidget::QmitkMIDASMultiViewWidget(
    QmitkMIDASMultiViewVisibilityManager* visibilityManager,
    int defaultNumberOfRows,
    int defaultNumberOfColumns,
    QWidget* parent, Qt::WindowFlags f)
: QWidget(parent, f)
, m_LayoutForRenderWindows(NULL)
, m_LayoutForLayoutButtons(NULL)
, m_LayoutForDropRadioButtons(NULL)
, m_LayoutForTopControls(NULL)
, m_LayoutToPutButtonsOnTopOfWindows(NULL)
, m_TopLevelLayout(NULL)
, m_VisibilityManager(visibilityManager)
, m_FocusManagerObserverTag(0)
, m_SelectedWindow(-1)
, m_DefaultNumberOfRows(defaultNumberOfRows)
, m_DefaultNumberOfColumns(defaultNumberOfColumns)
{
  assert(visibilityManager);

  int gridMargins = 0;
  int gridSpacing = 2;

  m_TopLevelLayout = new QHBoxLayout(this);
  m_TopLevelLayout->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_TopLevelLayout"));
  m_TopLevelLayout->setSpacing(0);

  m_LayoutToPutButtonsOnTopOfWindows = new QVBoxLayout();
  m_LayoutToPutButtonsOnTopOfWindows->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutToPutButtonsOnTopOfWindows"));
  m_LayoutToPutButtonsOnTopOfWindows->setSpacing(0);

  m_LayoutForTopControls = new QHBoxLayout();
  m_LayoutForTopControls->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForTopControls"));
  m_LayoutForTopControls->setSpacing(0);

  m_LayoutForLayoutButtons = new QGridLayout();
  m_LayoutForLayoutButtons->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForLayoutButtons"));
  m_LayoutForLayoutButtons->setContentsMargins(gridMargins, gridMargins, gridMargins, gridMargins);
  m_LayoutForLayoutButtons->setVerticalSpacing(gridSpacing);
  m_LayoutForLayoutButtons->setHorizontalSpacing(gridSpacing);

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForRenderWindows"));
  m_LayoutForRenderWindows->setContentsMargins(gridMargins, gridMargins, gridMargins, gridMargins);
  m_LayoutForRenderWindows->setVerticalSpacing(gridSpacing);
  m_LayoutForRenderWindows->setHorizontalSpacing(gridSpacing);

  m_LayoutForDropRadioButtons = new QGridLayout();
  m_LayoutForDropRadioButtons->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForDropRadioButtons"));
  m_LayoutForDropRadioButtons->setContentsMargins(gridMargins, gridMargins, gridMargins, gridMargins);
  m_LayoutForDropRadioButtons->setVerticalSpacing(gridSpacing);
  m_LayoutForDropRadioButtons->setHorizontalSpacing(gridSpacing);

  m_HorizontalSpacerBetweenRadioButtonsAndBindButton = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
  m_HorizontalSpacerBetweenBindButtonAndLayoutButtons = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);

  m_BindWindowsCheckBox = new QCheckBox(this);
  m_BindWindowsCheckBox->setText("bind");
  m_BindWindowsCheckBox->setToolTip("bind windows together to control multiple windows");
  m_BindWindowsCheckBox->setWhatsThis("bind windows together so that the slice selection, magnification, and orientation from the selected window is propagated across all windows. ");

  m_1x1LayoutButton = new QPushButton(this);
  m_1x1LayoutButton->setText("1x1");
  m_1x1LayoutButton->setToolTip("display 1 row and 1 column of image viewers");

  m_1x2LayoutButton = new QPushButton(this);
  m_1x2LayoutButton->setText("1x2");
  m_1x2LayoutButton->setToolTip("display 1 row and 2 columns of image viewers");

  m_2x1LayoutButton = new QPushButton(this);
  m_2x1LayoutButton->setText("2x1");
  m_2x1LayoutButton->setToolTip("display 2 rows and 1 column of image viewers");

  m_1x3LayoutButton = new QPushButton(this);
  m_1x3LayoutButton->setText("1x3");
  m_1x3LayoutButton->setToolTip("display 1 row and 3 columns of image viewers");

  m_3x1LayoutButton = new QPushButton(this);
  m_3x1LayoutButton->setText("3x1");
  m_3x1LayoutButton->setToolTip("display 3 rows and 1 column of image viewers");

  m_2x2LayoutButton = new QPushButton(this);
  m_2x2LayoutButton->setText("2x2");
  m_2x2LayoutButton->setToolTip("display 2 rows and 2 columns of image viewers");

  m_3x2LayoutButton = new QPushButton(this);
  m_3x2LayoutButton->setText("3x2");
  m_3x2LayoutButton->setToolTip("display 3 rows and 2 columns of image viewers");

  m_2x3LayoutButton = new QPushButton(this);
  m_2x3LayoutButton->setText("2x3");
  m_2x3LayoutButton->setToolTip("display 2 rows and 2 columns of image viewers");

  m_5x5LayoutButton = new QPushButton(this);
  m_5x5LayoutButton->setText("5x5");
  m_5x5LayoutButton->setToolTip("display 5 rows and 5 columns of image viewers");

  m_RowsSpinBox = new QSpinBox(this);
  m_RowsSpinBox->setMinimum(1);
  m_RowsSpinBox->setMaximum(m_MaxRows);
  m_RowsSpinBox->setValue(1);
  m_RowsSpinBox->setToolTip("click the arrows or type to change the number of rows");

  m_RowsLabel = new QLabel(this);
  m_RowsLabel->setText("rows");

  m_ColumnsSpinBox = new QSpinBox(this);
  m_ColumnsSpinBox->setMinimum(1);
  m_ColumnsSpinBox->setMaximum(m_MaxCols);
  m_ColumnsSpinBox->setValue(1);
  m_ColumnsSpinBox->setToolTip("click the arrows or type to change the number of columns");

  m_ColumnsLabel = new QLabel(this);
  m_ColumnsLabel->setText("columns");

  m_DropLabel = new QLabel(this);
  m_DropLabel->setText("drop:");

  m_DropSingleRadioButton = new QRadioButton(this);
  m_DropSingleRadioButton->setText("single");
  m_DropSingleRadioButton->setToolTip("drop images into a single window");

  m_DropMultipleRadioButton = new QRadioButton(this);
  m_DropMultipleRadioButton->setText("multiple");
  m_DropMultipleRadioButton->setToolTip("drop images across multiple windows");

  m_DropThumbnailRadioButton = new QRadioButton(this);
  m_DropThumbnailRadioButton->setText("all");
  m_DropThumbnailRadioButton->setToolTip("drop multiple images into any window, and the application will spread them across all windows and provide evenly spaced slices through the image");

  for (unsigned int i = 0; i < m_MaxRows*m_MaxCols; i++)
  {
    QmitkMIDASSingleViewWidget *widget = new QmitkMIDASSingleViewWidget(this, tr("QmitkMIDASRenderWindow %1").arg(i), -5, 20);
    widget->SetContentsMargins(0);
    widget->SetSpacing(0);
    widget->setObjectName(tr("QmitkMIDASSingleViewWidget %1").arg(i));

    QmitkMIDASRenderWindow* widgetWindow = widget->GetRenderWindow();
    connect(widgetWindow, SIGNAL(NodesDropped(QmitkMIDASRenderWindow*,std::vector<mitk::DataNode*>)), m_VisibilityManager, SLOT(OnNodesDropped(QmitkMIDASRenderWindow*,std::vector<mitk::DataNode*>)));
    connect(widgetWindow, SIGNAL(NodesDropped(QmitkMIDASRenderWindow*,std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkMIDASRenderWindow*,std::vector<mitk::DataNode*>)));
    connect(widget, SIGNAL(SliceChanged(QmitkMIDASRenderWindow*, unsigned int)), this, SLOT(OnSliceChanged(QmitkMIDASRenderWindow*, unsigned int)));

    m_VisibilityManager->RegisterWidget(widget);
    m_SingleViewWidgets.push_back(widget);
  }

  /************************************
   * Now arrange stuff.
   ************************************/

  for (unsigned int i = 0; i <  m_MaxRows*m_MaxCols; i++)
  {
    m_LayoutForRenderWindows->addWidget(m_SingleViewWidgets[i], this->GetRowFromIndex(i), this->GetColumnFromIndex(i));
  }

  m_LayoutForLayoutButtons->addWidget(m_1x1LayoutButton, 0, 0);
  m_LayoutForLayoutButtons->addWidget(m_1x2LayoutButton, 0, 1);
  m_LayoutForLayoutButtons->addWidget(m_2x1LayoutButton, 0, 2);
  m_LayoutForLayoutButtons->addWidget(m_1x3LayoutButton, 0, 3);
  m_LayoutForLayoutButtons->addWidget(m_3x1LayoutButton, 0, 4);
  m_LayoutForLayoutButtons->addWidget(m_2x2LayoutButton, 0, 5);
  m_LayoutForLayoutButtons->addWidget(m_3x2LayoutButton, 0, 6);
  m_LayoutForLayoutButtons->addWidget(m_2x3LayoutButton, 0, 7);
  m_LayoutForLayoutButtons->addWidget(m_5x5LayoutButton, 0, 8);
  m_LayoutForLayoutButtons->addWidget(m_RowsLabel, 0, 9);
  m_LayoutForLayoutButtons->addWidget(m_RowsSpinBox, 0, 10);
  m_LayoutForLayoutButtons->addWidget(m_ColumnsLabel, 0, 11);
  m_LayoutForLayoutButtons->addWidget(m_ColumnsSpinBox, 0, 12);

  m_LayoutForDropRadioButtons->addWidget(m_DropLabel, 0, 0);
  m_LayoutForDropRadioButtons->addWidget(m_DropSingleRadioButton, 0, 1);
  m_LayoutForDropRadioButtons->addWidget(m_DropMultipleRadioButton, 0, 2);
  m_LayoutForDropRadioButtons->addWidget(m_DropThumbnailRadioButton, 0, 3);

  m_LayoutForTopControls->addLayout(m_LayoutForDropRadioButtons);
  m_LayoutForTopControls->addItem(m_HorizontalSpacerBetweenRadioButtonsAndBindButton);
  m_LayoutForTopControls->addWidget(m_BindWindowsCheckBox);
  m_LayoutForTopControls->addItem(m_HorizontalSpacerBetweenBindButtonAndLayoutButtons);
  m_LayoutForTopControls->addLayout(m_LayoutForLayoutButtons);

  m_LayoutToPutButtonsOnTopOfWindows->addLayout(m_LayoutForTopControls);
  m_LayoutToPutButtonsOnTopOfWindows->addLayout(m_LayoutForRenderWindows);

  m_TopLevelLayout->addLayout(m_LayoutToPutButtonsOnTopOfWindows);

  itk::SimpleMemberCommand<QmitkMIDASMultiViewWidget>::Pointer onFocusChangedCommand =
    itk::SimpleMemberCommand<QmitkMIDASMultiViewWidget>::New();
  onFocusChangedCommand->SetCallbackFunction( this, &QmitkMIDASMultiViewWidget::OnFocusChanged );

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);

  connect(m_1x1LayoutButton, SIGNAL(pressed()), this, SLOT(On1x1ButtonPressed()));
  connect(m_1x2LayoutButton, SIGNAL(pressed()), this, SLOT(On1x2ButtonPressed()));
  connect(m_2x1LayoutButton, SIGNAL(pressed()), this, SLOT(On2x1ButtonPressed()));
  connect(m_1x3LayoutButton, SIGNAL(pressed()), this, SLOT(On1x3ButtonPressed()));
  connect(m_3x1LayoutButton, SIGNAL(pressed()), this, SLOT(On3x1ButtonPressed()));
  connect(m_2x2LayoutButton, SIGNAL(pressed()), this, SLOT(On2x2ButtonPressed()));
  connect(m_3x2LayoutButton, SIGNAL(pressed()), this, SLOT(On3x2ButtonPressed()));
  connect(m_2x3LayoutButton, SIGNAL(pressed()), this, SLOT(On2x3ButtonPressed()));
  connect(m_5x5LayoutButton, SIGNAL(pressed()), this, SLOT(On5x5ButtonPressed()));
  connect(m_RowsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnRowsSliderValueChanged(int)));
  connect(m_ColumnsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnColumnsSliderValueChanged(int)));
  connect(m_DropSingleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropSingleRadioButtonToggled(bool)));
  connect(m_DropMultipleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropMultipleRadioButtonToggled(bool)));
  connect(m_DropThumbnailRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropThumbnailRadioButtonToggled(bool)));
  connect(m_BindWindowsCheckBox, SIGNAL(clicked(bool)), this, SLOT(OnBindWindowsCheckboxClicked(bool)));

  m_DropSingleRadioButton->blockSignals(true);
  m_DropSingleRadioButton->setChecked(true);
  m_DropSingleRadioButton->blockSignals(false);

  this->m_VisibilityManager->SetDropType(QmitkMIDASMultiViewVisibilityManager::MIDAS_DROP_TYPE_SINGLE);

  this->SetLayoutSize(m_DefaultNumberOfRows, m_DefaultNumberOfColumns, false);
  this->SetSelectedWindow(0);
}

QmitkMIDASMultiViewWidget::~QmitkMIDASMultiViewWidget()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  focusManager->RemoveObserver(m_FocusManagerObserverTag);
}

unsigned int QmitkMIDASMultiViewWidget::GetRowFromIndex(unsigned int i)
{
  if (i < 0 || i >= m_MaxRows*m_MaxCols)
  {
    return 0;
  }
  else
  {
    return i / m_MaxCols; // Note, intentionally integer division
  }
}

unsigned int QmitkMIDASMultiViewWidget::GetColumnFromIndex(unsigned int i)
{
  if (i < 0 || i >= m_MaxRows*m_MaxCols)
  {
    return 0;
  }
  else
  {
    return i % m_MaxCols; // Note, intentionally modulus.
  }
}

void QmitkMIDASMultiViewWidget::On1x1ButtonPressed()
{
  this->SetLayoutSize(1,1, false);
}

void QmitkMIDASMultiViewWidget::On1x2ButtonPressed()
{
  this->SetLayoutSize(1,2, false);
}

void QmitkMIDASMultiViewWidget::On2x1ButtonPressed()
{
  this->SetLayoutSize(2,1, false);
}

void QmitkMIDASMultiViewWidget::On3x1ButtonPressed()
{
  this->SetLayoutSize(3,1, false);
}

void QmitkMIDASMultiViewWidget::On1x3ButtonPressed()
{
  this->SetLayoutSize(1,3, false);
}

void QmitkMIDASMultiViewWidget::On2x2ButtonPressed()
{
  this->SetLayoutSize(2,2, false);
}

void QmitkMIDASMultiViewWidget::On3x2ButtonPressed()
{
  this->SetLayoutSize(3,2, false);
}

void QmitkMIDASMultiViewWidget::On2x3ButtonPressed()
{
  this->SetLayoutSize(2,3, false);
}

void QmitkMIDASMultiViewWidget::On5x5ButtonPressed()
{
  this->SetLayoutSize(5,5, false);
}

void QmitkMIDASMultiViewWidget::OnRowsSliderValueChanged(int r)
{
  this->SetLayoutSize((unsigned int)r, (unsigned int)m_ColumnsSpinBox->value(), false);
}

void QmitkMIDASMultiViewWidget::OnColumnsSliderValueChanged(int c)
{
  this->SetLayoutSize((unsigned int)m_RowsSpinBox->value(), (unsigned int)c, false);
}

void QmitkMIDASMultiViewWidget::OnBindWindowsCheckboxClicked(bool isBound)
{
  if (isBound)
  {
    mitk::TimeSlicedGeometry::Pointer selectedGeometry = m_SingleViewWidgets[m_SelectedWindow]->GetGeometry();
    QmitkMIDASSingleViewWidget::MIDASViewOrientation orientation = m_SingleViewWidgets[m_SelectedWindow]->GetViewOrientation();
    int sliceNumber = m_SingleViewWidgets[m_SelectedWindow]->GetSliceNumber();
    int magnification = m_SingleViewWidgets[m_SelectedWindow]->GetMagnificationFactor();
    int timeStepNumber = m_SingleViewWidgets[m_SelectedWindow]->GetTime();

    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      m_SingleViewWidgets[i]->SetBound(true);
      m_SingleViewWidgets[i]->SetBoundGeometry(selectedGeometry);
      m_SingleViewWidgets[i]->SetViewOrientation(orientation, false);
      m_SingleViewWidgets[i]->SetSliceNumber(sliceNumber);
      m_SingleViewWidgets[i]->SetTime(timeStepNumber);
      m_SingleViewWidgets[i]->SetMagnificationFactor(magnification);
    }
  }
  else
  {
    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      m_SingleViewWidgets[i]->SetBound(false);
    }
    this->PublishNavigationSettings();
  }
}

unsigned int QmitkMIDASMultiViewWidget::GetIndexFromRowAndColumn(unsigned int r, unsigned int c)
{
  return r*m_MaxCols + c;
}

void QmitkMIDASMultiViewWidget::EnableWidgetsForThumbnailMode(bool isThumbnailMode)
{
  bool isEnabledInThumbnailMode = !isThumbnailMode;
  m_1x1LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_1x2LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_2x1LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_3x1LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_2x2LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_1x3LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_3x2LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_2x3LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_5x5LayoutButton->setEnabled(isEnabledInThumbnailMode);
  m_RowsSpinBox->setEnabled(isEnabledInThumbnailMode);
  m_RowsLabel->setEnabled(isEnabledInThumbnailMode);
  m_ColumnsSpinBox->setEnabled(isEnabledInThumbnailMode);
  m_ColumnsLabel->setEnabled(isEnabledInThumbnailMode);
}

void QmitkMIDASMultiViewWidget::SetLayoutSize(unsigned int numberOfRows, unsigned int numberOfColumns, bool isThumbnailMode)
{

  // We need to remember the "previous" number of rows and columns, so when we switch out
  // of thumbnail mode, we know how many rows and columns to revert to.
  if (isThumbnailMode)
  {
    m_NumberOfRowsInNonThumbnailMode = m_RowsSpinBox->value();
    m_NumberOfColumnsInNonThumbnailMode = m_ColumnsSpinBox->value();
  }
  else
  {
    // otherwise we remember the "next" (the number we are being asked for in this method call) number of rows and columns.
    m_NumberOfRowsInNonThumbnailMode = numberOfRows;
    m_NumberOfColumnsInNonThumbnailMode = numberOfColumns;
  }

  // Remember, all widgets are created, we just make them visible/invisible.
  for(unsigned int r = 0; r < m_MaxRows; r++)
  {
    for (unsigned int c = 0; c < m_MaxCols; c++)
    {
      int viewerIndex = this->GetIndexFromRowAndColumn(r, c);
      bool active = true;

      if (r >= numberOfRows || c >= numberOfColumns)
      {
        active = false;
      }
      m_SingleViewWidgets[viewerIndex]->setVisible(active);
    }
  }

  m_RowsSpinBox->blockSignals(true);
  m_RowsSpinBox->setValue(numberOfRows);
  m_RowsSpinBox->blockSignals(false);
  m_ColumnsSpinBox->blockSignals(true);
  m_ColumnsSpinBox->setValue(numberOfColumns);
  m_ColumnsSpinBox->blockSignals(false);

  // Test the current m_Selected window, and reset to 0 if it now points to an invisible window.
  int selectedWindow = m_SelectedWindow;
  if (this->GetRowFromIndex(selectedWindow) >= numberOfRows || this->GetColumnFromIndex(selectedWindow) >= numberOfColumns)
  {
    selectedWindow = 0;
  }
  this->SetSelectedWindow(selectedWindow);
}

void QmitkMIDASMultiViewWidget::SetBackgroundColour(mitk::Color colour)
{
  QColor background(colour[0] * 255, colour[1] * 255, colour[2] * 255);

  for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetBackgroundColor(background);
  }
  this->RequestUpdateAll();
}

void QmitkMIDASMultiViewWidget::SetSelectedWindow(unsigned int selectedIndex)
{
  if (selectedIndex >= 0 && selectedIndex < m_SingleViewWidgets.size())
  {
    m_SelectedWindow = selectedIndex;

    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (i == selectedIndex)
      {
        m_SingleViewWidgets[i]->SetSelected(true);
      }
      else
      {
        m_SingleViewWidgets[i]->SetSelected(false);
      }
    }
    this->RequestUpdateAll();
    this->PublishNavigationSettings();
  }
}

void QmitkMIDASMultiViewWidget::SetDefaultInterpolationType(QmitkMIDASMultiViewVisibilityManager::MIDASDefaultInterpolationType interpolationType)
{
  m_VisibilityManager->SetDefaultInterpolationType(interpolationType);
}

void QmitkMIDASMultiViewWidget::SetDefaultOrientationType(QmitkMIDASMultiViewVisibilityManager::MIDASDefaultOrientationType orientationType)
{
  m_VisibilityManager->SetDefaultOrientationType(orientationType);
}

void QmitkMIDASMultiViewWidget::OnDropSingleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(QmitkMIDASMultiViewVisibilityManager::MIDAS_DROP_TYPE_SINGLE);

    this->SetLayoutSize(m_NumberOfRowsInNonThumbnailMode, m_NumberOfColumnsInNonThumbnailMode, false);
    this->EnableWidgetsForThumbnailMode(false);
  }
}

void QmitkMIDASMultiViewWidget::OnDropMultipleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(QmitkMIDASMultiViewVisibilityManager::MIDAS_DROP_TYPE_MULTIPLE);

    this->SetLayoutSize(m_NumberOfRowsInNonThumbnailMode, m_NumberOfColumnsInNonThumbnailMode, false);
    this->EnableWidgetsForThumbnailMode(false);
  }
}

void QmitkMIDASMultiViewWidget::OnDropThumbnailRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(QmitkMIDASMultiViewVisibilityManager::MIDAS_DROP_TYPE_ALL);

    this->SetLayoutSize(m_MaxRows, m_MaxCols, true);
    this->EnableWidgetsForThumbnailMode(true);
  }
}

void QmitkMIDASMultiViewWidget::OnSliceChanged(QmitkMIDASRenderWindow *window, unsigned int sliceNumber)
{
  this->PublishNavigationSettings();
}

void QmitkMIDASMultiViewWidget::OnNodesDropped(QmitkMIDASRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(window->GetRenderer());
}

void QmitkMIDASMultiViewWidget::PublishNavigationSettings()
{
  if (this->isVisible()) // this is to stop any initial updates before this widget is fully rendered on screen.
  {
    UpdateMIDASViewingControlsInfo currentInfo;

    currentInfo.minTime = this->m_SingleViewWidgets[m_SelectedWindow]->GetMinTime();
    currentInfo.maxTime = this->m_SingleViewWidgets[m_SelectedWindow]->GetMaxTime();
    currentInfo.minSlice = this->m_SingleViewWidgets[m_SelectedWindow]->GetMinSlice();
    currentInfo.maxSlice = this->m_SingleViewWidgets[m_SelectedWindow]->GetMaxSlice();
    currentInfo.minMagnification = this->m_SingleViewWidgets[m_SelectedWindow]->GetMinMagnification();
    currentInfo.maxMagnification = this->m_SingleViewWidgets[m_SelectedWindow]->GetMaxMagnification();

    currentInfo.currentTime = this->m_SingleViewWidgets[m_SelectedWindow]->GetTime();
    currentInfo.currentSlice = this->m_SingleViewWidgets[m_SelectedWindow]->GetSliceNumber();
    currentInfo.currentMagnification = this->m_SingleViewWidgets[m_SelectedWindow]->GetMagnificationFactor();

    QmitkMIDASSingleViewWidget::MIDASViewOrientation orientation = this->m_SingleViewWidgets[m_SelectedWindow]->GetViewOrientation();
    if (orientation == QmitkMIDASSingleViewWidget::MIDAS_VIEW_AXIAL)
    {
      currentInfo.isAxial = true;
      currentInfo.isSagittal = false;
      currentInfo.isCoronal = false;
    }
    else if (orientation == QmitkMIDASSingleViewWidget::MIDAS_VIEW_SAGITTAL)
    {
      currentInfo.isAxial = false;
      currentInfo.isSagittal = true;
      currentInfo.isCoronal = false;
    }
    else if (orientation == QmitkMIDASSingleViewWidget::MIDAS_VIEW_CORONAL)
    {
      currentInfo.isAxial = false;
      currentInfo.isSagittal = false;
      currentInfo.isCoronal = true;
    }

    emit UpdateMIDASViewingControlsValues(currentInfo);
  }
}

void QmitkMIDASMultiViewWidget::OnFocusChanged()
{
  vtkRenderWindow* focusedRenderWindow = NULL;
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer::ConstPointer baseRenderer = focusManager->GetFocused();

  int selectedWindow = -1;

  if (baseRenderer.IsNotNull())
  {
    focusedRenderWindow = baseRenderer->GetRenderWindow();
    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i]->ContainsVtkRenderWindow(focusedRenderWindow))
      {
        selectedWindow = i;
        break;
      }
    }
  }

  if (selectedWindow != -1)
  {
    this->SetSelectedWindow(selectedWindow);
  }
}

void QmitkMIDASMultiViewWidget::paintEvent(QPaintEvent* event)
{
  this->RequestUpdateAll();
}

void QmitkMIDASMultiViewWidget::RequestUpdateAll()
{
  for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (m_SingleViewWidgets[i]->isVisible())
    {
      m_SingleViewWidgets[i]->RequestUpdate();
    }
  }
}

void QmitkMIDASMultiViewWidget::ForceUpdateAll()
{
  for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (m_SingleViewWidgets[i]->isVisible())
    {
      m_SingleViewWidgets[i]->ForceUpdate();
    }
  }
}

void QmitkMIDASMultiViewWidget::GetStartStopIndexForIteration(unsigned int &start, unsigned int &stop)
{
  if (this->m_BindWindowsCheckBox->isChecked())
  {
    start = 0;
    stop = this->m_SingleViewWidgets.size() -1;
  }
  else
  {
    start = m_SelectedWindow;
    stop = m_SelectedWindow;
  }
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowMagnification(int magnificationFactor)
{
  unsigned int start;
  unsigned int stop;
  this->GetStartStopIndexForIteration(start, stop);

  for (unsigned int i = start; i <= stop; i++)
  {
    if (this->m_SingleViewWidgets[i]->isVisible())
    {
      this->m_SingleViewWidgets[i]->SetMagnificationFactor(magnificationFactor);
    }
  }
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowSliceNumber(int sliceNumber)
{
  unsigned int start;
  unsigned int stop;
  this->GetStartStopIndexForIteration(start, stop);

  for (unsigned int i = start; i <= stop; i++)
  {
    if (this->m_SingleViewWidgets[i]->isVisible())
    {
      this->m_SingleViewWidgets[i]->SetSliceNumber(sliceNumber);
    }
  }
}

void QmitkMIDASMultiViewWidget::SetSelectedTimeStep(int timeStep)
{
  unsigned int start;
  unsigned int stop;
  if (m_DropThumbnailRadioButton->isChecked())
  {
    start = 0;
    stop = this->m_SingleViewWidgets.size() -1;
  }
  else
  {
    this->GetStartStopIndexForIteration(start, stop);
  }

  for (unsigned int i = start; i <= stop; i++)
  {
    if (this->m_SingleViewWidgets[i]->isVisible())
    {
      this->m_SingleViewWidgets[i]->SetTime(timeStep);
    }
  }
}

void QmitkMIDASMultiViewWidget::SetWindowsToOrientation(QmitkMIDASSingleViewWidget::MIDASViewOrientation orientation)
{
  unsigned int start;
  unsigned int stop;
  this->GetStartStopIndexForIteration(start, stop);

  for (unsigned int i = start; i <= stop; i++)
  {
    if (this->m_SingleViewWidgets[i]->isVisible())
    {
      this->m_SingleViewWidgets[i]->SetViewOrientation(orientation, false);
    }
  }
  this->PublishNavigationSettings();
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowToAxial()
{
  this->SetWindowsToOrientation(QmitkMIDASSingleViewWidget::MIDAS_VIEW_AXIAL);
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowToSagittal()
{
  this->SetWindowsToOrientation(QmitkMIDASSingleViewWidget::MIDAS_VIEW_SAGITTAL);
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowToCoronal()
{
  this->SetWindowsToOrientation(QmitkMIDASSingleViewWidget::MIDAS_VIEW_CORONAL);
}

bool QmitkMIDASMultiViewWidget::MoveAnterior()
{
  bool actuallyDidSomething = false;

  unsigned int currentSlice = this->m_SingleViewWidgets[m_SelectedWindow]->GetSliceNumber();
  unsigned int maxSlice = this->m_SingleViewWidgets[m_SelectedWindow]->GetMaxSlice();
  unsigned int nextSlice = currentSlice+1;
  if (nextSlice <= maxSlice)
  {
    this->SetSelectedWindowSliceNumber(nextSlice);
    actuallyDidSomething = true;
  }

  this->PublishNavigationSettings();
  return actuallyDidSomething;
}

bool QmitkMIDASMultiViewWidget::MovePosterior()
{
  bool actuallyDidSomething = false;

  unsigned int currentSlice = this->m_SingleViewWidgets[m_SelectedWindow]->GetSliceNumber();
  unsigned int maxSlice = this->m_SingleViewWidgets[m_SelectedWindow]->GetMinSlice();
  if (currentSlice > 0)
  {
    unsigned int nextSlice = currentSlice-1;
    if (nextSlice >= maxSlice)
    {
      this->SetSelectedWindowSliceNumber(nextSlice);
      actuallyDidSomething = true;
    }
  }

  this->PublishNavigationSettings();
  return actuallyDidSomething;
}

bool QmitkMIDASMultiViewWidget::SwitchToAxial()
{
  this->SetSelectedWindowToAxial();
  return true;
}

bool QmitkMIDASMultiViewWidget::SwitchToSagittal()
{
  this->SetSelectedWindowToSagittal();
  return true;
}

bool QmitkMIDASMultiViewWidget::SwitchToCoronal()
{
  this->SetSelectedWindowToCoronal();
  return true;
}
