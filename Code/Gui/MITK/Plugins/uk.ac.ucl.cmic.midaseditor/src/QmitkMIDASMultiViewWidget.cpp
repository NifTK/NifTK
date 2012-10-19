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
#include <QStackedLayout>
#include <QButtonGroup>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkGeometry3D.h>
#include <mitkIRenderWindowPart.h>
#include <QmitkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <ctkPopupWidget.h>

#include "mitkMIDASViewKeyPressResponder.h"
#include "QmitkMIDASSingleViewWidget.h"
#include "mitkMIDASOrientationUtils.h"

QmitkMIDASMultiViewWidget::QmitkMIDASMultiViewWidget(
    QmitkMIDASMultiViewVisibilityManager* visibilityManager,
    mitk::RenderingManager* renderingManager,
    mitk::DataStorage::Pointer dataStorage,
    int defaultNumberOfRows,
    int defaultNumberOfColumns,
    QWidget* parent, Qt::WindowFlags f)
: QWidget(parent, f)
, mitk::MIDASViewKeyPressResponder()
, m_TopLevelLayout(NULL)
, m_LayoutToPutControlsOnTopOfWindows(NULL)
, m_LayoutForGroupingControls(NULL)
, m_LayoutForTopControls(NULL)
, m_LayoutForLayoutWidgets(NULL)
, m_LayoutForDropWidgets(NULL)
, m_LayoutForRenderWindows(NULL)
, m_MIDASOrientationWidget(NULL)
, m_MIDASSlidersWidget(NULL)
, m_MIDASBindWidget(NULL)
, m_1x1LayoutButton(NULL)
, m_1x2LayoutButton(NULL)
, m_1x3LayoutButton(NULL)
, m_2x2LayoutButton(NULL)
, m_RowsSpinBox(NULL)
, m_RowsLabel(NULL)
, m_ColumnsSpinBox(NULL)
, m_ColumnsLabel(NULL)
, m_DropSingleRadioButton(NULL)
, m_DropMultipleRadioButton(NULL)
, m_DropThumbnailRadioButton(NULL)
, m_DropAccumulateCheckBox(NULL)
, m_PopupPushButton(NULL)
, m_PopupWidget(NULL)
, m_ControlsContainerWidget(NULL)
, m_VisibilityManager(visibilityManager)
, m_DataStorage(dataStorage)
, m_RenderingManager(renderingManager)
, m_FocusManagerObserverTag(0)
, m_SelectedWindow(0)
, m_DefaultNumberOfRows(defaultNumberOfRows)
, m_DefaultNumberOfColumns(defaultNumberOfColumns)
, m_InteractionEnabled(false)
, m_Show2DCursors(false)
, m_Show3DViewInOrthoview(false)
, m_IsThumbnailMode(false)
, m_IsMIDASSegmentationMode(false)
, m_NavigationControllerEventListening(false)
, m_Dropped(false)
, m_InteractorsEnabled(false)
{
  assert(visibilityManager);

  this->setFocusPolicy(Qt::StrongFocus);

  /************************************
   * Create stuff.
   ************************************/

  m_TopLevelLayout = new QHBoxLayout(this);
  m_TopLevelLayout->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_TopLevelLayout"));
  m_TopLevelLayout->setContentsMargins(0, 0, 0, 0);
  m_TopLevelLayout->setSpacing(0);

  int buttonRowHeight = 10;
  m_PopupPushButton = new QPushButton(this);
  m_PopupPushButton->setContentsMargins(0,0,0,0);
  m_PopupPushButton->setFlat(true);
  m_PopupPushButton->setMinimumHeight(buttonRowHeight);
  m_PopupPushButton->setMaximumHeight(buttonRowHeight);

  m_PopupWidget = new ctkPopupWidget(m_PopupPushButton);
  m_PopupWidget->setOrientation(Qt::Vertical);
  m_PopupWidget->setAnimationEffect(ctkBasePopupWidget::ScrollEffect);
  m_PopupWidget->setHorizontalDirection(Qt::LeftToRight);
  m_PopupWidget->setVerticalDirection(ctkBasePopupWidget::TopToBottom);
  m_PopupWidget->setAutoShow(true);
  m_PopupWidget->setAutoHide(true);
  m_PopupWidget->setEffectDuration(100);
  m_PopupWidget->setContentsMargins(0,0,0,0);
  m_PopupWidget->setLineWidth(0);

  m_ControlsContainerWidget = new QFrame(m_PopupWidget);
  m_ControlsContainerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  m_ControlsContainerWidget->setContentsMargins(0, 0, 0, 0);
  m_ControlsContainerWidget->setLineWidth(0);

  m_LayoutForGroupingControls = new QHBoxLayout(m_PopupWidget);
  m_LayoutForGroupingControls->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForGroupingControls"));
  m_LayoutForGroupingControls->setContentsMargins(0, 0, 0, 0);
  m_LayoutForGroupingControls->setSpacing(0);

  m_LayoutToPutControlsOnTopOfWindows = new QGridLayout();
  m_LayoutToPutControlsOnTopOfWindows->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutToPutControlsOnTopOfWindows"));
  m_LayoutToPutControlsOnTopOfWindows->setContentsMargins(0, 0, 0, 0);
  m_LayoutToPutControlsOnTopOfWindows->setSpacing(0);

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForRenderWindows"));
  m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
  m_LayoutForRenderWindows->setVerticalSpacing(0);
  m_LayoutForRenderWindows->setHorizontalSpacing(0);

  m_LayoutForDropWidgets = new QVBoxLayout();
  m_LayoutForDropWidgets->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForDropWidgets"));
  m_LayoutForDropWidgets->setContentsMargins(2, 0, 2, 0);
  m_LayoutForDropWidgets->setSpacing(0);

  m_LayoutForLayoutWidgets = new QGridLayout();
  m_LayoutForLayoutWidgets->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForLayoutWidgets"));
  m_LayoutForLayoutWidgets->setContentsMargins(2, 0, 2, 0);
  m_LayoutForLayoutWidgets->setVerticalSpacing(0);
  m_LayoutForLayoutWidgets->setHorizontalSpacing(2);

  m_LayoutForTopControls = new QGridLayout();
  m_LayoutForTopControls->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForTopControls"));
  m_LayoutForTopControls->setContentsMargins(5, 0, 5, 0);
  m_LayoutForTopControls->setVerticalSpacing(0);
  m_LayoutForTopControls->setHorizontalSpacing(5);

  m_MIDASSlidersWidget = new QmitkMIDASSlidersWidget(m_ControlsContainerWidget);
  m_MIDASSlidersWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

  m_MIDASOrientationWidget = new QmitkMIDASOrientationWidget(m_ControlsContainerWidget);

  m_MIDASBindWidget = new QmitkMIDASBindWidget(m_ControlsContainerWidget);

  m_1x1LayoutButton = new QPushButton(m_ControlsContainerWidget);
  m_1x1LayoutButton->setText("1x1");
  m_1x1LayoutButton->setToolTip("display 1 row and 1 column of image viewers");

  m_1x2LayoutButton = new QPushButton(m_ControlsContainerWidget);
  m_1x2LayoutButton->setText("1x2");
  m_1x2LayoutButton->setToolTip("display 1 row and 2 columns of image viewers");

  m_1x3LayoutButton = new QPushButton(m_ControlsContainerWidget);
  m_1x3LayoutButton->setText("1x3");
  m_1x3LayoutButton->setToolTip("display 1 row and 3 columns of image viewers");

  m_2x2LayoutButton = new QPushButton(m_ControlsContainerWidget);
  m_2x2LayoutButton->setText("2x2");
  m_2x2LayoutButton->setToolTip("display 2 rows and 2 columns of image viewers");

  m_RowsSpinBox = new QSpinBox(m_ControlsContainerWidget);
  m_RowsSpinBox->setMinimum(1);
  m_RowsSpinBox->setMaximum(m_MaxRows);
  m_RowsSpinBox->setValue(1);
  m_RowsSpinBox->setToolTip("click the arrows or type to change the number of rows");

  m_RowsLabel = new QLabel(m_ControlsContainerWidget);
  m_RowsLabel->setText("rows");

  m_ColumnsSpinBox = new QSpinBox(m_ControlsContainerWidget);
  m_ColumnsSpinBox->setMinimum(1);
  m_ColumnsSpinBox->setMaximum(m_MaxCols);
  m_ColumnsSpinBox->setValue(1);
  m_ColumnsSpinBox->setToolTip("click the arrows or type to change the number of columns");

  m_ColumnsLabel = new QLabel(m_ControlsContainerWidget);
  m_ColumnsLabel->setText("columns");

  m_DropSingleRadioButton = new QRadioButton(m_ControlsContainerWidget);
  m_DropSingleRadioButton->setText("single");
  m_DropSingleRadioButton->setToolTip("drop images into a single window");
  m_DropSingleRadioButton->setLayoutDirection(Qt::LeftToRight);

  m_DropMultipleRadioButton = new QRadioButton(m_ControlsContainerWidget);
  m_DropMultipleRadioButton->setText("multiple");
  m_DropMultipleRadioButton->setToolTip("drop images across multiple windows");
  m_DropMultipleRadioButton->setLayoutDirection(Qt::LeftToRight);

  m_DropThumbnailRadioButton = new QRadioButton(m_ControlsContainerWidget);
  m_DropThumbnailRadioButton->setText("all");
  m_DropThumbnailRadioButton->setToolTip("drop multiple images into any window, and the application will spread them across all windows and provide evenly spaced slices through the image");
  m_DropThumbnailRadioButton->setLayoutDirection(Qt::LeftToRight);

  m_DropAccumulateCheckBox = new QCheckBox(m_ControlsContainerWidget);
  m_DropAccumulateCheckBox->setText("accumulate");
  m_DropAccumulateCheckBox->setToolTip("dropped images accumulate, meaning you can repeatedly add more images without resetting the geometry");
  m_DropAccumulateCheckBox->setLayoutDirection(Qt::LeftToRight);

  m_DropButtonGroup = new QButtonGroup(m_ControlsContainerWidget);
  m_DropButtonGroup->addButton(m_DropSingleRadioButton);
  m_DropButtonGroup->addButton(m_DropMultipleRadioButton);
  m_DropButtonGroup->addButton(m_DropThumbnailRadioButton);

  /************************************
   * Now arrange stuff.
   ************************************/

  m_LayoutForDropWidgets->addWidget(m_DropSingleRadioButton);
  m_LayoutForDropWidgets->addWidget(m_DropMultipleRadioButton);
  m_LayoutForDropWidgets->addWidget(m_DropThumbnailRadioButton);
  m_LayoutForDropWidgets->addWidget(m_DropAccumulateCheckBox);

  m_LayoutForLayoutWidgets->addWidget(m_1x1LayoutButton,  0, 0);
  m_LayoutForLayoutWidgets->addWidget(m_1x2LayoutButton,  0, 1);
  m_LayoutForLayoutWidgets->addWidget(m_1x3LayoutButton,  1, 0);
  m_LayoutForLayoutWidgets->addWidget(m_2x2LayoutButton,  1, 1);
  m_LayoutForLayoutWidgets->addWidget(m_RowsLabel,        2, 0);
  m_LayoutForLayoutWidgets->addWidget(m_RowsSpinBox,      2, 1);
  m_LayoutForLayoutWidgets->addWidget(m_ColumnsLabel,     3, 0);
  m_LayoutForLayoutWidgets->addWidget(m_ColumnsSpinBox,   3, 1);

  m_LayoutForTopControls->addWidget(m_MIDASSlidersWidget,     0, 0, 3, 1);
  m_LayoutForTopControls->addLayout(m_LayoutForLayoutWidgets, 0, 1, 3, 1);
  m_LayoutForTopControls->addWidget(m_MIDASOrientationWidget, 0, 2, 3, 1);
  m_LayoutForTopControls->addLayout(m_LayoutForDropWidgets,   0, 3, 3, 1);
  m_LayoutForTopControls->addWidget(m_MIDASBindWidget,        0, 4, 3, 1);

  m_LayoutForTopControls->setColumnMinimumWidth(0, 50);
  m_LayoutForTopControls->setColumnStretch(0, 5);
  m_LayoutForTopControls->setColumnStretch(1, 1);
  m_LayoutForTopControls->setColumnStretch(2, 0);
  m_LayoutForTopControls->setColumnStretch(3, 0);
  m_LayoutForTopControls->setColumnStretch(4, 0);

  m_LayoutForGroupingControls->addLayout(m_LayoutForTopControls);
  m_LayoutToPutControlsOnTopOfWindows->addWidget(m_PopupPushButton, 0, 0);
  m_LayoutToPutControlsOnTopOfWindows->setRowMinimumHeight(0, buttonRowHeight);
  m_LayoutToPutControlsOnTopOfWindows->addLayout(m_LayoutForRenderWindows, 1, 0);
  m_TopLevelLayout->addLayout(m_LayoutToPutControlsOnTopOfWindows);

  /************************************
   * Now initialise stuff.
   ************************************/

  // Default to dropping into single window.
  m_DropSingleRadioButton->setChecked(true);
  this->m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_SINGLE);

  // We have the default rows and columns passed in via constructor args, in initialise list.
  m_RowsSpinBox->setValue(m_DefaultNumberOfRows);
  m_ColumnsSpinBox->setValue(m_DefaultNumberOfColumns);
  this->SetLayoutSize(m_DefaultNumberOfRows, m_DefaultNumberOfColumns, false);

  // Default all widgets off except layout widgets, until something dropped.
  this->EnableWidgets(false);
  this->EnableLayoutWidgets(true);

  // Connect Qt Signals to make it all hang together.
  connect(m_MIDASSlidersWidget->m_SliceSelectionWidget, SIGNAL(valueChanged(double)), this, SLOT(OnSliceNumberChanged(double)));
  connect(m_MIDASSlidersWidget->m_MagnificationFactorWidget, SIGNAL(valueChanged(double)), this, SLOT(OnMagnificationFactorChanged(double)));
  connect(m_MIDASSlidersWidget->m_TimeSelectionWidget, SIGNAL(valueChanged(double)), this, SLOT(OnTimeChanged(double)));
  connect(m_1x1LayoutButton, SIGNAL(pressed()), this, SLOT(On1x1ButtonPressed()));
  connect(m_1x2LayoutButton, SIGNAL(pressed()), this, SLOT(On1x2ButtonPressed()));
  connect(m_1x3LayoutButton, SIGNAL(pressed()), this, SLOT(On1x3ButtonPressed()));
  connect(m_2x2LayoutButton, SIGNAL(pressed()), this, SLOT(On2x2ButtonPressed()));
  connect(m_RowsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnRowsSliderValueChanged(int)));
  connect(m_ColumnsSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnColumnsSliderValueChanged(int)));
  connect(m_MIDASOrientationWidget, SIGNAL(ViewChanged(MIDASView)), this, SLOT(OnOrientationSelected(MIDASView)));
  connect(m_DropSingleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropSingleRadioButtonToggled(bool)));
  connect(m_DropMultipleRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropMultipleRadioButtonToggled(bool)));
  connect(m_DropThumbnailRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnDropThumbnailRadioButtonToggled(bool)));
  connect(m_DropAccumulateCheckBox, SIGNAL(stateChanged(int)), this, SLOT(OnDropAccumulateStateChanged(int)));
  connect(m_MIDASBindWidget, SIGNAL(BindTypeChanged(MIDASBindType)), this, SLOT(OnBindModeSelected(MIDASBindType)));
  connect(m_PopupWidget, SIGNAL(popupOpened(bool)), this, SLOT(OnPopupOpened(bool)));

  // We listen to FocusManager to detect when things have changed focus, and hence to highlight the "current window".
  itk::SimpleMemberCommand<QmitkMIDASMultiViewWidget>::Pointer onFocusChangedCommand =
    itk::SimpleMemberCommand<QmitkMIDASMultiViewWidget>::New();
  onFocusChangedCommand->SetCallbackFunction( this, &QmitkMIDASMultiViewWidget::OnFocusChanged );

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
}

QmitkMIDASMultiViewWidget::~QmitkMIDASMultiViewWidget()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }
  this->Deactivated();
}

QmitkMIDASSingleViewWidget* QmitkMIDASMultiViewWidget::CreateSingleViewWidget()
{
  QmitkMIDASSingleViewWidget *widget = new QmitkMIDASSingleViewWidget(tr("QmitkRenderWindow"),
                                                                      -5, 20,
                                                                      this,
                                                                      m_RenderingManager,
                                                                      m_DataStorage);
  widget->setObjectName(tr("QmitkMIDASSingleViewWidget"));
  widget->setVisible(false);

  connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), m_VisibilityManager, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)));
  connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)));
  connect(widget, SIGNAL(PositionChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, mitk::Index3D, mitk::Point3D, int, MIDASOrientation)), this, SLOT(OnPositionChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, mitk::Index3D,mitk::Point3D, int, MIDASOrientation)));
  connect(widget, SIGNAL(MagnificationFactorChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, double)), this, SLOT(OnMagnificationFactorChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, double)));

  return widget;
}

void QmitkMIDASMultiViewWidget::RequestUpdateAll()
{
  std::vector<unsigned int> listToUpdate = this->GetViewerIndexesToUpdate(true);
  for (unsigned int i = 0; i < listToUpdate.size(); i++)
  {
    if (listToUpdate[i] >= 0 && listToUpdate[i] < this->m_SingleViewWidgets.size())
    {
      m_SingleViewWidgets[listToUpdate[i]]->RequestUpdate();
    }
  }
}

void QmitkMIDASMultiViewWidget::SetDefaultInterpolationType(MIDASDefaultInterpolationType interpolationType)
{
  m_VisibilityManager->SetDefaultInterpolationType(interpolationType);
}

void QmitkMIDASMultiViewWidget::SetDefaultViewType(MIDASView view)
{
  m_VisibilityManager->SetDefaultViewType(view);
}

void QmitkMIDASMultiViewWidget::SetDropTypeWidget(MIDASDropType dropType)
{
  if (dropType == MIDAS_DROP_TYPE_SINGLE)
  {
    m_DropSingleRadioButton->setChecked(true);
  }
  else if (dropType == MIDAS_DROP_TYPE_MULTIPLE)
  {
    m_DropMultipleRadioButton->setChecked(true);
  }
  else if (dropType == MIDAS_DROP_TYPE_ALL)
  {
    m_DropThumbnailRadioButton->setChecked(true);
  }
  else
  {
    MITK_ERROR << "QmitkMIDASMultiViewWidget::SetDropTypeWidget: Invalid MIDASDropType=" << dropType << std::endl;
  }
}

void QmitkMIDASMultiViewWidget::SetShowDropTypeWidgets(bool visible)
{
  m_DropSingleRadioButton->setVisible(visible);
  m_DropMultipleRadioButton->setVisible(visible);
  m_DropThumbnailRadioButton->setVisible(visible);
  m_DropAccumulateCheckBox->setVisible(visible);
}

void QmitkMIDASMultiViewWidget::SetShowLayoutButtons(bool visible)
{
  m_1x1LayoutButton->setVisible(visible);
  m_1x2LayoutButton->setVisible(visible);
  m_1x3LayoutButton->setVisible(visible);
  m_2x2LayoutButton->setVisible(visible);
}

void QmitkMIDASMultiViewWidget::SetShowMagnificationSlider(bool visible)
{
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->setVisible(visible);
}

void QmitkMIDASMultiViewWidget::SetShow2DCursors(bool visible)
{
  m_Show2DCursors = visible;
  this->Update2DCursorVisibility();
}

bool QmitkMIDASMultiViewWidget::GetShow2DCursors() const
{
  return m_Show2DCursors;
}

void QmitkMIDASMultiViewWidget::SetShow3DViewInOrthoView(bool visible)
{
  m_Show3DViewInOrthoview = visible;
  for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetDisplay3DViewInOrthoView(visible);
  }
  this->RequestUpdateAll();
}

bool QmitkMIDASMultiViewWidget::GetShow3DViewInOrthoView() const
{
  return m_Show3DViewInOrthoview;
}

void QmitkMIDASMultiViewWidget::SetRememberViewSettingsPerOrientation(bool remember)
{
  for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetRememberViewSettingsPerOrientation(remember);
  }
}

void QmitkMIDASMultiViewWidget::EnableSliderWidgets(bool enabled)
{
  m_MIDASSlidersWidget->SetEnabled(enabled);
}

void QmitkMIDASMultiViewWidget::EnableOrientationWidgets(bool enabled)
{
  m_MIDASOrientationWidget->SetEnabled(enabled);
}

void QmitkMIDASMultiViewWidget::EnableBindWidgets(bool enabled)
{
  m_MIDASBindWidget->SetEnabled(enabled);
}

void QmitkMIDASMultiViewWidget::EnableDropTypeWidgets(bool enabled)
{
  m_DropSingleRadioButton->setEnabled(enabled);
  m_DropMultipleRadioButton->setEnabled(enabled);
  m_DropThumbnailRadioButton->setEnabled(enabled);
  m_DropAccumulateCheckBox->setEnabled(enabled);
}

void QmitkMIDASMultiViewWidget::EnableLayoutWidgets(bool enabled)
{
  m_1x1LayoutButton->setEnabled(enabled);
  m_1x2LayoutButton->setEnabled(enabled);
  m_1x3LayoutButton->setEnabled(enabled);
  m_2x2LayoutButton->setEnabled(enabled);
  m_RowsSpinBox->setEnabled(enabled);
  m_RowsLabel->setEnabled(enabled);
  m_ColumnsSpinBox->setEnabled(enabled);
  m_ColumnsLabel->setEnabled(enabled);
}

void QmitkMIDASMultiViewWidget::EnableWidgets(bool enabled)
{
  this->EnableDropTypeWidgets(enabled);
  this->EnableSliderWidgets(enabled);
  this->EnableLayoutWidgets(enabled);
  this->EnableOrientationWidgets(enabled);
  this->EnableBindWidgets(enabled);
}

void QmitkMIDASMultiViewWidget::SetThumbnailMode(bool enabled)
{
  this->m_IsThumbnailMode = enabled;

  if (enabled)
  {
    m_NumberOfRowsInNonThumbnailMode = m_RowsSpinBox->value();
    m_NumberOfColumnsInNonThumbnailMode = m_ColumnsSpinBox->value();
    this->EnableSliderWidgets(false);
    this->EnableLayoutWidgets(false);
    this->EnableOrientationWidgets(false);
    this->EnableBindWidgets(false);
    this->SetLayoutSize(m_MaxRows, m_MaxCols, true);
  }
  else
  {
    this->EnableSliderWidgets(true);
    this->EnableLayoutWidgets(true);
    this->EnableOrientationWidgets(true);
    this->EnableBindWidgets(true);
    this->SetLayoutSize(m_NumberOfRowsInNonThumbnailMode, m_NumberOfColumnsInNonThumbnailMode, false);
  }
}

bool QmitkMIDASMultiViewWidget::GetThumbnailMode() const
{
  return this->m_IsThumbnailMode;
}

void QmitkMIDASMultiViewWidget::SetMIDASSegmentationMode(bool enabled)
{
  this->m_IsMIDASSegmentationMode = enabled;

  if (enabled)
  {
    this->m_NumberOfRowsBeforeSegmentationMode = m_RowsSpinBox->value();
    this->m_NumberOfColumnsBeforeSegmentationMode = m_ColumnsSpinBox->value();
    this->EnableLayoutWidgets(false);
    this->EnableBindWidgets(false);
    this->SetLayoutSize(1, 1, false);
    this->SetSelectedWindow(0);
    this->UpdateFocusManagerToSelectedViewer();
  }
  else
  {
    this->EnableLayoutWidgets(true);
    this->EnableBindWidgets(true);
    this->SetLayoutSize(m_NumberOfRowsBeforeSegmentationMode, m_NumberOfColumnsBeforeSegmentationMode, false);
  }
}

bool QmitkMIDASMultiViewWidget::GetMIDASSegmentationMode() const
{
  return this->m_IsMIDASSegmentationMode;
}

MIDASView QmitkMIDASMultiViewWidget::GetDefaultOrientationForSegmentation() const
{
  assert(m_VisibilityManager);

  MIDASView viewForSegmentation = m_VisibilityManager->GetDefaultViewType();

  if (   viewForSegmentation != MIDAS_VIEW_AXIAL
      && viewForSegmentation != MIDAS_VIEW_SAGITTAL
      && viewForSegmentation != MIDAS_VIEW_CORONAL
     )
  {
    viewForSegmentation = MIDAS_VIEW_CORONAL;
  }

  return viewForSegmentation;
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

void QmitkMIDASMultiViewWidget::SetLayoutSize(unsigned int numberOfRows, unsigned int numberOfColumns, bool isThumbnailMode)
{
  // Work out required number of widgets, and hence if we need to create any new ones.
  unsigned int requiredNumberOfWidgets = numberOfRows * numberOfColumns;
  unsigned int currentNumberOfWidgets = m_SingleViewWidgets.size();

  // If we have the right number of widgets, there is nothing to do, so early exit.
  if (requiredNumberOfWidgets == currentNumberOfWidgets)
  {
    return;
  }

  /////////////////////////////////////////
  // Start: Rebuild the number of widgets.
  // NOTE:  The order of widgets in
  //        m_SingleViewWidgets and
  //        m_VisibilityManager must match.
  /////////////////////////////////////////

  if (requiredNumberOfWidgets > currentNumberOfWidgets)
  {
    // create some more widgets
    unsigned int additionalWidgets = requiredNumberOfWidgets - m_SingleViewWidgets.size();
    for (unsigned int i = 0; i < additionalWidgets; i++)
    {
      QmitkMIDASSingleViewWidget *widget = this->CreateSingleViewWidget();
      widget->hide();

      this->m_SingleViewWidgets.push_back(widget);
      this->m_VisibilityManager->RegisterWidget(widget);
      this->m_VisibilityManager->SetAllNodeVisibilityForWindow(currentNumberOfWidgets+i, false);
    }
  }
  else if (requiredNumberOfWidgets < currentNumberOfWidgets)
  {
    // destroy surplus widgets
    this->m_VisibilityManager->DeRegisterWidgets(requiredNumberOfWidgets, m_SingleViewWidgets.size()-1);

    for (unsigned int i = requiredNumberOfWidgets; i < m_SingleViewWidgets.size(); i++)
    {
      delete m_SingleViewWidgets[i];
    }

    m_SingleViewWidgets.erase(m_SingleViewWidgets.begin() + requiredNumberOfWidgets,
                              m_SingleViewWidgets.end()
                             );
  }

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

  // Make all current widgets inVisible, as we are going to destroy layout.
  for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->hide();
  }

  // Put all widgets in the grid.
  // Prior experience suggests we always need a new grid,
  // because otherwise widgets don't appear to remove properly.

  m_LayoutToPutControlsOnTopOfWindows->removeItem(m_LayoutForRenderWindows);
  delete m_LayoutForRenderWindows;

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForRenderWindows"));
  m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
  m_LayoutForRenderWindows->setVerticalSpacing(0);
  m_LayoutForRenderWindows->setHorizontalSpacing(0);

  m_LayoutToPutControlsOnTopOfWindows->addLayout(m_LayoutForRenderWindows, 1, 0);

  unsigned int widgetCounter = 0;
  for (unsigned int r = 0; r < numberOfRows; r++)
  {
    for (unsigned int c = 0; c < numberOfColumns; c++)
    {
      m_LayoutForRenderWindows->addWidget(m_SingleViewWidgets[widgetCounter], r, c);
      m_SingleViewWidgets[widgetCounter]->show();
      m_SingleViewWidgets[widgetCounter]->setEnabled(true);
      widgetCounter++;
    }
  }

  ////////////////////////////////////////
  // End: Rebuild the number of widgets.
  ////////////////////////////////////////

  // Update row/column widget without triggering another layout size change.
  m_RowsSpinBox->blockSignals(true);
  m_RowsSpinBox->setValue(numberOfRows);
  m_RowsSpinBox->blockSignals(false);
  m_ColumnsSpinBox->blockSignals(true);
  m_ColumnsSpinBox->setValue(numberOfColumns);
  m_ColumnsSpinBox->blockSignals(false);

  // Test the current m_Selected window, and reset to 0 if it now points to an invisible window.
  int selectedWindow = this->GetSelectedWindowIndex();
  if (this->GetRowFromIndex(selectedWindow) >= numberOfRows || this->GetColumnFromIndex(selectedWindow) >= numberOfColumns)
  {
    selectedWindow = 0;
  }
  // Pass NULL for the selected vtkRenderWindow, to make sure that new windows don't look selected
  this->SwitchWindows(selectedWindow, NULL);

  // Now the number of viewers has changed, we need to make sure they are all in synch with all the right properties.
  this->Update2DCursorVisibility();
  this->SetShow3DViewInOrthoView(this->m_Show3DViewInOrthoview);

  // Make sure that if we are bound, we re-synch the geometry, or magnification.
  if (this->m_MIDASBindWidget->IsGeometryBound())
  {
    this->UpdateBoundGeometry(true);
  }
  if (this->m_MIDASBindWidget->IsMagnificationBound())
  {
    this->UpdateBoundMagnification(true);
  }
}

unsigned int QmitkMIDASMultiViewWidget::GetRowFromIndex(unsigned int i) const
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

unsigned int QmitkMIDASMultiViewWidget::GetColumnFromIndex(unsigned int i) const
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

unsigned int QmitkMIDASMultiViewWidget::GetIndexFromRowAndColumn(unsigned int r, unsigned int c) const
{
  return r*m_MaxCols + c;
}

void QmitkMIDASMultiViewWidget::On1x1ButtonPressed()
{
  this->SetLayoutSize(1,1, false);
}

void QmitkMIDASMultiViewWidget::On1x2ButtonPressed()
{
  this->SetLayoutSize(1,2, false);
}

void QmitkMIDASMultiViewWidget::On1x3ButtonPressed()
{
  this->SetLayoutSize(1,3, false);
}

void QmitkMIDASMultiViewWidget::On2x2ButtonPressed()
{
  this->SetLayoutSize(2,2, false);
}

void QmitkMIDASMultiViewWidget::OnRowsSliderValueChanged(int r)
{
  this->SetLayoutSize((unsigned int)r, (unsigned int)m_ColumnsSpinBox->value(), false);
}

void QmitkMIDASMultiViewWidget::OnColumnsSliderValueChanged(int c)
{
  this->SetLayoutSize((unsigned int)m_RowsSpinBox->value(), (unsigned int)c, false);
}

void QmitkMIDASMultiViewWidget::OnPositionChanged(QmitkMIDASSingleViewWidget *widget, QmitkRenderWindow* window, mitk::Index3D voxelLocation, mitk::Point3D millimetreLocation, int sliceNumber, MIDASOrientation orientation)
{
  for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (m_SingleViewWidgets[i] == widget)
    {
      std::vector<QmitkRenderWindow*> windows = m_SingleViewWidgets[i]->GetSelectedWindows();
      if (windows.size() == 1 && window == windows[0] && sliceNumber != m_MIDASSlidersWidget->m_SliceSelectionWidget->value())
      {
        // This should only be used to update the sliceNumber on the GUI, so must not trigger a further update.
        m_MIDASSlidersWidget->m_SliceSelectionWidget->blockSignals(true);
        m_MIDASSlidersWidget->m_SliceSelectionWidget->setValue(sliceNumber);
        m_MIDASSlidersWidget->m_SliceSelectionWidget->blockSignals(false);
      }
    }
  }
}

void QmitkMIDASMultiViewWidget::OnMagnificationFactorChanged(QmitkMIDASSingleViewWidget *widget, QmitkRenderWindow* window, double magnificationFactor)
{
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->blockSignals(true);
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->setValue(magnificationFactor);
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->blockSignals(false);
  if (this->m_MIDASBindWidget->IsMagnificationBound())
  {
    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != widget)
      {
        m_SingleViewWidgets[i]->SetMagnificationFactor(magnificationFactor);
      }
    }
  }
}

void QmitkMIDASMultiViewWidget::OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes)
{
  // See also QmitkMIDASMultiViewVisibilityManager::OnNodesDropped which should trigger first.
  if (!this->m_DropThumbnailRadioButton->isChecked())
  {
    this->EnableWidgets(true);
  }

  // This does not trigger OnFocusChanged() the very first time, as when creating the editor, the first widget already has focus.
  mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(window->GetRenderer());
  if (!m_Dropped)
  {
    this->OnFocusChanged();
    m_Dropped = true;
  }

  int selectedWindow = this->GetSelectedWindowIndex();
  int magnification = m_SingleViewWidgets[selectedWindow]->GetMagnificationFactor();
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->setValue(magnification);

  MIDASView view = m_SingleViewWidgets[selectedWindow]->GetView();
  m_MIDASOrientationWidget->SetToView(view);

  this->Update2DCursorVisibility();
  this->RequestUpdateAll();
}

void QmitkMIDASMultiViewWidget::SwitchWindows(int selectedViewer, vtkRenderWindow *selectedWindow)
{
  if (selectedViewer >= 0 && selectedViewer < (int)m_SingleViewWidgets.size())
  {
    // This, to turn off borders on all other windows.
    this->SetSelectedWindow(selectedViewer);

    // This to specifically set the border round one sub-pane for if its an ortho-view.
    if (selectedWindow != NULL)
    {
      int numberOfNodes = m_VisibilityManager->GetNodesInWindow(selectedViewer);
      if (numberOfNodes > 0)
      {
        this->m_SingleViewWidgets[selectedViewer]->SetSelectedWindow(selectedWindow);
      }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Need to enable widgets appropriately, so user can't press stuff that they aren't meant to.
    /////////////////////////////////////////////////////////////////////////////////////////////
    MIDASOrientation orientation = this->m_SingleViewWidgets[selectedViewer]->GetOrientation();
    MIDASView view = this->m_SingleViewWidgets[selectedViewer]->GetView();

    m_MIDASSlidersWidget->SetBlockSignals(true);
    m_MIDASOrientationWidget->SetBlockSignals(true);

    if (view != MIDAS_VIEW_UNKNOWN)
    {
      m_MIDASOrientationWidget->SetToView(view);
    }
    if (orientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      unsigned int minSlice = this->m_SingleViewWidgets[selectedViewer]->GetMinSlice(orientation);
      unsigned int maxSlice = this->m_SingleViewWidgets[selectedViewer]->GetMaxSlice(orientation);
      unsigned int currentSlice = this->m_SingleViewWidgets[selectedViewer]->GetSliceNumber(orientation);

      m_MIDASSlidersWidget->m_SliceSelectionWidget->setMinimum(minSlice);
      m_MIDASSlidersWidget->m_SliceSelectionWidget->setMaximum(maxSlice);
      m_MIDASSlidersWidget->m_SliceSelectionWidget->setValue(currentSlice);
    }

    double minMag = this->m_SingleViewWidgets[selectedViewer]->GetMinMagnification();
    double maxMag = this->m_SingleViewWidgets[selectedViewer]->GetMaxMagnification();
    double currentMag = this->m_SingleViewWidgets[selectedViewer]->GetMagnificationFactor();
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setMinimum((int)minMag);
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setMaximum((int)maxMag);
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setValue((int)currentMag);

    unsigned int minTime = this->m_SingleViewWidgets[selectedViewer]->GetMinTime();
    unsigned int maxTime = this->m_SingleViewWidgets[selectedViewer]->GetMaxTime();
    unsigned int currentTime = this->m_SingleViewWidgets[selectedViewer]->GetTime();
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setMinimum(minTime);
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setMaximum(maxTime);
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setValue(currentTime);

    m_MIDASSlidersWidget->m_SliceSelectionWidget->setEnabled(true);
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setEnabled(true);
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setEnabled(true);

    m_MIDASSlidersWidget->SetBlockSignals(false);
    m_MIDASOrientationWidget->SetBlockSignals(false);

    this->Update2DCursorVisibility();
  }
  this->RequestUpdateAll();
}

void QmitkMIDASMultiViewWidget::OnFocusChanged()
{

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer::ConstPointer baseRenderer = focusManager->GetFocused();

  vtkRenderWindow* focusedRenderWindow = NULL;
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
  this->SwitchWindows(selectedWindow, focusedRenderWindow);
}

void QmitkMIDASMultiViewWidget::OnDropSingleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_SINGLE);
    this->SetThumbnailMode(false);
  }
}

void QmitkMIDASMultiViewWidget::OnDropMultipleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_MULTIPLE);
    this->SetThumbnailMode(false);
  }
}

void QmitkMIDASMultiViewWidget::OnDropThumbnailRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_ALL);
    this->SetThumbnailMode(true);
  }
}

void QmitkMIDASMultiViewWidget::OnDropAccumulateStateChanged(int state)
{
  if (state == Qt::Checked)
  {
    m_VisibilityManager->SetAccumulateWhenDropping(true);
  }
  else
  {
    m_VisibilityManager->SetAccumulateWhenDropping(false);
  }
}

std::vector<unsigned int> QmitkMIDASMultiViewWidget::GetViewerIndexesToUpdate(bool doAllVisible) const
{
  std::vector<unsigned int> result;

  if (doAllVisible)
  {
    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i]->isVisible())
      {
        result.push_back(i);
      }
    }
  }
  else
  {
    int windowIndex = this->GetSelectedWindowIndex();
    result.push_back((unsigned int)windowIndex);
  }
  return result;
}

bool QmitkMIDASMultiViewWidget::MoveAnterior()
{
  return this->MoveAnteriorPosterior(true, 1);
}

bool QmitkMIDASMultiViewWidget::MovePosterior()
{
  return this->MoveAnteriorPosterior(false, 1);
}

bool QmitkMIDASMultiViewWidget::MoveAnteriorPosterior(bool moveAnterior, int slices)
{
  bool actuallyDidSomething = false;
  int selectedWindow = this->GetSelectedWindowIndex();

  if (selectedWindow != -1)
  {
    MIDASOrientation orientation = this->m_SingleViewWidgets[selectedWindow]->GetOrientation();
    unsigned int currentSlice = this->m_SingleViewWidgets[selectedWindow]->GetSliceNumber(orientation);
    unsigned int minSlice = this->m_SingleViewWidgets[selectedWindow]->GetMinSlice(orientation);
    unsigned int maxSlice = this->m_SingleViewWidgets[selectedWindow]->GetMaxSlice(orientation);

    int upDirection = this->m_SingleViewWidgets[selectedWindow]->GetSliceUpDirection(orientation);
    int nextSlice = currentSlice;

    if (moveAnterior)
    {
      nextSlice = currentSlice + slices*upDirection;
    }
    else
    {
      nextSlice = currentSlice - slices*upDirection;
    }

    if (nextSlice >= (int)minSlice && nextSlice <= (int)maxSlice)
    {
      this->SetSelectedWindowSliceNumber(nextSlice);
      actuallyDidSomething = true;
    }
  }
  return actuallyDidSomething;
}

void QmitkMIDASMultiViewWidget::OnSliceNumberChanged(double sliceNumber)
{
  this->SetSelectedWindowSliceNumber((int)sliceNumber);
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowSliceNumber(int sliceNumber)
{
  int selectedWindow = this->GetSelectedWindowIndex();
  MIDASOrientation orientation = this->m_SingleViewWidgets[selectedWindow]->GetOrientation();

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(this->m_MIDASBindWidget->IsGeometryBound());
    for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
    {
      this->m_SingleViewWidgets[viewersToUpdate[i]]->SetSliceNumber(orientation, sliceNumber);
    }
  }
  else
  {
    MITK_WARN << "Found an invalid orientation in widget " << selectedWindow << ", so ignoring request to change to slice " << sliceNumber << std::endl;
  }
}

void QmitkMIDASMultiViewWidget::OnMagnificationFactorChanged(double magnificationFactor)
{
  this->SetSelectedWindowMagnification((int)magnificationFactor);
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowMagnification(int magnificationFactor)
{
  std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(this->m_MIDASBindWidget->IsMagnificationBound());
  for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
  {
    this->m_SingleViewWidgets[viewersToUpdate[i]]->SetMagnificationFactor(magnificationFactor);
  }
}

void QmitkMIDASMultiViewWidget::OnTimeChanged(double timeStep)
{
  this->SetSelectedTimeStep((int)timeStep);
}

void QmitkMIDASMultiViewWidget::SetSelectedTimeStep(int timeStep)
{
  std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(this->m_DropThumbnailRadioButton->isChecked());
  for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
  {
    this->m_SingleViewWidgets[viewersToUpdate[i]]->SetTime(timeStep);
  }
}

void QmitkMIDASMultiViewWidget::OnOrientationSelected(MIDASView view)
{
  if (view != MIDAS_VIEW_UNKNOWN)
  {
    this->SwitchView(view);

    // Update the focus to the selected window, to trigger things like thumbnail viewer refresh
    // (or indeed anything that's listening to the FocusManager).
    this->UpdateFocusManagerToSelectedViewer();
  }
}

void QmitkMIDASMultiViewWidget::UpdateFocusManagerToSelectedViewer()
{
  int selectedWindow = this->GetSelectedWindowIndex();
  std::vector<QmitkRenderWindow*> windows = this->m_SingleViewWidgets[selectedWindow]->GetSelectedWindows();

  if (windows.size() > 0)
  {
    mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(windows[0]->GetRenderer());
  }
}

bool QmitkMIDASMultiViewWidget::SwitchToAxial()
{
  this->SetSelectedWindowToAxial();

  m_MIDASOrientationWidget->SetBlockSignals(true);
  m_MIDASOrientationWidget->SetToView(MIDAS_VIEW_AXIAL);
  m_MIDASOrientationWidget->SetBlockSignals(false);
  this->UpdateFocusManagerToSelectedViewer();
  return true;
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowToAxial()
{
  this->SwitchView(MIDAS_VIEW_AXIAL);
}

bool QmitkMIDASMultiViewWidget::SwitchToSagittal()
{
  this->SetSelectedWindowToSagittal();

  m_MIDASOrientationWidget->SetBlockSignals(true);
  m_MIDASOrientationWidget->SetToView(MIDAS_VIEW_SAGITTAL);
  m_MIDASOrientationWidget->SetBlockSignals(false);
  this->UpdateFocusManagerToSelectedViewer();
  return true;
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowToSagittal()
{
  this->SwitchView(MIDAS_VIEW_SAGITTAL);
}

bool QmitkMIDASMultiViewWidget::SwitchToCoronal()
{
  this->SetSelectedWindowToCoronal();

  m_MIDASOrientationWidget->SetBlockSignals(true);
  m_MIDASOrientationWidget->SetToView(MIDAS_VIEW_CORONAL);
  m_MIDASOrientationWidget->SetBlockSignals(false);
  this->UpdateFocusManagerToSelectedViewer();
  return true;
}

void QmitkMIDASMultiViewWidget::SetSelectedWindowToCoronal()
{
  this->SwitchView(MIDAS_VIEW_CORONAL);
}

void QmitkMIDASMultiViewWidget::SwitchView(MIDASView view)
{
  int selectedWindow = this->GetSelectedWindowIndex();

  std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(this->m_MIDASBindWidget->IsGeometryBound());
  for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
  {
    unsigned int viewerNumber = viewersToUpdate[i];
    this->m_SingleViewWidgets[viewerNumber]->SetView(view, false);

    if ((int)viewerNumber == selectedWindow)
    {
      if (view == MIDAS_VIEW_AXIAL)
      {
        this->m_SingleViewWidgets[viewerNumber]
              ->SetSelectedWindow(this->m_SingleViewWidgets[viewerNumber]->GetAxialWindow()->GetVtkRenderWindow());
      }
      else if (view == MIDAS_VIEW_SAGITTAL)
      {
        this->m_SingleViewWidgets[viewerNumber]
              ->SetSelectedWindow(this->m_SingleViewWidgets[viewerNumber]->GetSagittalWindow()->GetVtkRenderWindow());
      }
      else if (view == MIDAS_VIEW_CORONAL)
      {
        this->m_SingleViewWidgets[viewerNumber]
              ->SetSelectedWindow(this->m_SingleViewWidgets[viewerNumber]->GetCoronalWindow()->GetVtkRenderWindow());
      }
    }
  }
}

void QmitkMIDASMultiViewWidget::Update2DCursorVisibility()
{
  std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(true);
  for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
  {
    bool globalVisibility = false;
    bool localVisibility = m_Show2DCursors;

    m_SingleViewWidgets[viewersToUpdate[i]]->SetDisplay2DCursorsGlobally(globalVisibility);
    m_SingleViewWidgets[viewersToUpdate[i]]->SetDisplay2DCursorsLocally(localVisibility);
  }

  this->RequestUpdateAll();
}

void QmitkMIDASMultiViewWidget::UpdateBoundGeometry(bool isBoundNow)
{
  int selectedWindow = this->GetSelectedWindowIndex();

  mitk::Geometry3D::Pointer selectedGeometry = m_SingleViewWidgets[selectedWindow]->GetGeometry();
  MIDASOrientation orientation               = m_SingleViewWidgets[selectedWindow]->GetOrientation();
  MIDASView view                             = m_SingleViewWidgets[selectedWindow]->GetView();
  int sliceNumber                            = m_SingleViewWidgets[selectedWindow]->GetSliceNumber(orientation);
  int magnification                          = m_SingleViewWidgets[selectedWindow]->GetMagnificationFactor();
  int timeStepNumber                         = m_SingleViewWidgets[selectedWindow]->GetTime();

  std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(isBoundNow);
  for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
  {
    unsigned int viewerIndex = viewersToUpdate[i];
    m_SingleViewWidgets[viewerIndex]->SetBoundGeometry(selectedGeometry);
    m_SingleViewWidgets[viewerIndex]->SetBoundGeometryActive(isBoundNow);
    m_SingleViewWidgets[viewerIndex]->SetView(view, false);
    m_SingleViewWidgets[viewerIndex]->SetSliceNumber(orientation, sliceNumber);
    m_SingleViewWidgets[viewerIndex]->SetMagnificationFactor(magnification);
    m_SingleViewWidgets[viewerIndex]->SetTime(timeStepNumber);
  } // end for each viewer
}

void QmitkMIDASMultiViewWidget::UpdateBoundMagnification(bool isBoundNow)
{
  if (isBoundNow)
  {
    int selectedWindow = this->GetSelectedWindowIndex();
    int magnification = m_SingleViewWidgets[selectedWindow]->GetMagnificationFactor();

    std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(isBoundNow);
    for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
    {
      unsigned int viewerIndex = viewersToUpdate[i];
      m_SingleViewWidgets[viewerIndex]->SetMagnificationFactor(magnification);
    } // end for each viewer
  }
}

int QmitkMIDASMultiViewWidget::GetSliceNumber() const
{
  return this->m_MIDASSlidersWidget->m_SliceSelectionWidget->value();
}

MIDASOrientation QmitkMIDASMultiViewWidget::GetOrientation() const
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

  if (this->m_MIDASOrientationWidget->m_AxialRadioButton->isChecked())
  {
    orientation = MIDAS_ORIENTATION_AXIAL;
  }
  else if (this->m_MIDASOrientationWidget->m_SagittalRadioButton->isChecked())
  {
    orientation = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (this->m_MIDASOrientationWidget->m_CoronalRadioButton->isChecked())
  {
    orientation = MIDAS_ORIENTATION_CORONAL;
  }

  return orientation;
}


int QmitkMIDASMultiViewWidget::GetSelectedWindowIndex() const
{
  int windowNumber = m_SelectedWindow;
  if (windowNumber < 0)
  {
    windowNumber = 0;
  }
  if (windowNumber >= (int)m_SingleViewWidgets.size())
  {
    // Default back to first viewer.
    windowNumber = 0;
  }

  // Note the following specification.
  assert(windowNumber >= 0);
  assert(windowNumber < (int)m_SingleViewWidgets.size());

  // Return a valid selected window index.
  return windowNumber;
}


QmitkRenderWindow* QmitkMIDASMultiViewWidget::GetActiveRenderWindow() const
{
  // NOTE: This MUST always return not-null.

  QmitkRenderWindow *window = NULL;
  std::vector<QmitkRenderWindow*> selectedWindows;

  int windowNumber = this->GetSelectedWindowIndex();

  selectedWindows = m_SingleViewWidgets[windowNumber]->GetSelectedWindows();
  if (selectedWindows.size() == 0)
  {
    selectedWindows = m_SingleViewWidgets[windowNumber]->GetAllWindows();
  }
  window = selectedWindows[0];

  return window;
}

QHash<QString,QmitkRenderWindow*> QmitkMIDASMultiViewWidget::GetRenderWindows() const
{
  // NOTE: This MUST always return a non-empty map.

  QHash<QString, QmitkRenderWindow*> wnds;

  // See org.mitk.gui.qt.imagenavigator plugin.
  //
  // The assumption is that a QmitkStdMultiWidget has windows called
  // axial, sagittal, coronal, 3d.
  //
  // So, if we take the currently selected widget, and name these render windows
  // accordingly, then the MITK imagenavigator can be used to update it.

  int windowNumber = this->GetSelectedWindowIndex();

  wnds.insert("axial", m_SingleViewWidgets[windowNumber]->GetAxialWindow());
  wnds.insert("sagittal", m_SingleViewWidgets[windowNumber]->GetSagittalWindow());
  wnds.insert("coronal", m_SingleViewWidgets[windowNumber]->GetCoronalWindow());
  wnds.insert("3d", m_SingleViewWidgets[windowNumber]->Get3DWindow());

  for (int i = 0; i < (int)m_SingleViewWidgets.size(); i++)
  {
    if (i != windowNumber)
    {
      QString id = tr(".%1").arg(i);

      wnds.insert("axial" + id, m_SingleViewWidgets[i]->GetAxialWindow());
      wnds.insert("sagittal" + id, m_SingleViewWidgets[i]->GetSagittalWindow());
      wnds.insert("coronal" + id, m_SingleViewWidgets[i]->GetCoronalWindow());
      wnds.insert("3d" + id, m_SingleViewWidgets[i]->Get3DWindow());
    }
  }

  return wnds;
}

QmitkRenderWindow* QmitkMIDASMultiViewWidget::GetRenderWindow(const QString& id) const
{
  QHash<QString,QmitkRenderWindow*> windows = this->GetRenderWindows();
  QHash<QString,QmitkRenderWindow*>::iterator iter = windows.find(id);
  if (iter != windows.end())
  {
    return iter.value();
  }
  else
  {
    return NULL;
  }
}

mitk::Point3D QmitkMIDASMultiViewWidget::GetSelectedPosition(const QString& /*id*/) const
{
  int windowNumber = this->GetSelectedWindowIndex();
  mitk::Point3D position = m_SingleViewWidgets[windowNumber]->GetSelectedPosition();
  return position;
}

void QmitkMIDASMultiViewWidget::SetSelectedPosition(const mitk::Point3D& pos, const QString& /*id*/)
{
  int windowNumber = this->GetSelectedWindowIndex();
  m_SingleViewWidgets[windowNumber]->SetSelectedPosition(pos);
}


void QmitkMIDASMultiViewWidget::Activated()
{
  this->setEnabled(true);
  this->EnableLinkedNavigation(true);
}


void QmitkMIDASMultiViewWidget::Deactivated()
{
  this->setEnabled(false);
  this->EnableLinkedNavigation(false);
}


void QmitkMIDASMultiViewWidget::EnableLinkedNavigation(bool enable)
{
  this->SetNavigationControllerEventListening(enable);
}


bool QmitkMIDASMultiViewWidget::IsLinkedNavigationEnabled() const
{
  return this->GetNavigationControllerEventListening();
}


bool QmitkMIDASMultiViewWidget::GetNavigationControllerEventListening() const
{
  return m_NavigationControllerEventListening;
}


void QmitkMIDASMultiViewWidget::SetNavigationControllerEventListening(bool enabled)
{
  int selectedWindow = this->GetSelectedWindowIndex();
  if (enabled && !this->m_NavigationControllerEventListening)
  {
    m_SingleViewWidgets[selectedWindow]->SetNavigationControllerEventListening(true);
  }
  else if (!enabled && this->m_NavigationControllerEventListening)
  {
    m_SingleViewWidgets[selectedWindow]->SetNavigationControllerEventListening(false);
  }
  this->m_NavigationControllerEventListening = enabled;
}


void QmitkMIDASMultiViewWidget::SetSelectedWindow(unsigned int selectedIndex)
{
  if (selectedIndex >= 0 && selectedIndex < m_SingleViewWidgets.size())
  {
    m_SelectedWindow = selectedIndex;

    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      int nodesInWindow = m_VisibilityManager->GetNodesInWindow(i);

      if (i == selectedIndex && nodesInWindow > 0)
      {
        m_SingleViewWidgets[i]->SetSelected(true);
      }
      else
      {
        m_SingleViewWidgets[i]->SetSelected(false);
      }

      if (  this->m_MIDASBindWidget->AreCursorsBound()
          || (!this->m_MIDASBindWidget->AreCursorsBound() && i == selectedIndex)
          )
      {
        m_SingleViewWidgets[i]->SetNavigationControllerEventListening(true);
      }
      else
      {
        m_SingleViewWidgets[i]->SetNavigationControllerEventListening(false);
      }
    }
    this->Update2DCursorVisibility();
    this->RequestUpdateAll();
  }
  else
  {
    MITK_WARN << "Ignoring request to set the selected window to window number " << selectedIndex << std::endl;
  }
}

void QmitkMIDASMultiViewWidget::OnBindModeSelected(MIDASBindType bind)
{
  bool currentGeometryBound = m_SingleViewWidgets[0]->GetBoundGeometryActive();
  bool requestedGeometryBound = this->m_MIDASBindWidget->IsGeometryBound();
  int selectedWindow = this->GetSelectedWindowIndex();

  if (currentGeometryBound != requestedGeometryBound)
  {
    this->UpdateBoundGeometry(this->m_MIDASBindWidget->IsGeometryBound());
  }

  this->UpdateBoundMagnification(this->m_MIDASBindWidget->IsMagnificationBound());

  if (this->m_MIDASBindWidget->AreCursorsBound())
  {
    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      m_SingleViewWidgets[i]->SetNavigationControllerEventListening(true);
    }
  }
  else
  {
    for (unsigned int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if ((int)i == selectedWindow)
      {
        m_SingleViewWidgets[i]->SetNavigationControllerEventListening(true);
      }
      else
      {
        m_SingleViewWidgets[i]->SetNavigationControllerEventListening(false);
      }
    }
  }

  this->Update2DCursorVisibility();
}

void QmitkMIDASMultiViewWidget::OnPopupOpened(bool opened)
{
  if (!opened)
  {
    std::vector<unsigned int> viewersToUpdate = this->GetViewerIndexesToUpdate(false);
    for (unsigned int i = 0; i < viewersToUpdate.size(); i++)
    {
      unsigned int viewerIndex = viewersToUpdate[i];
      m_SingleViewWidgets[viewerIndex]->repaint();
    } // end for each viewer
  }
}
