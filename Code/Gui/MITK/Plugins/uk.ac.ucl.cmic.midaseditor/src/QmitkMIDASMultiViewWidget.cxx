/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASMultiViewWidget.h"

#include <ctkDoubleSlider.h>
#include <ctkPopupWidget.h>

#include <QButtonGroup>
#include <QCheckBox>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QPalette>
#include <QSize>
#include <QSpacerItem>
#include <QSpinBox>
#include <QStackedLayout>
#include <QToolButton>
#include <QVBoxLayout>

#include <mitkFocusManager.h>
#include <mitkGeometry3D.h>
#include <mitkGlobalInteraction.h>
#include <QmitkRenderWindow.h>
#include <mitkIRenderWindowPart.h>

#include "mitkMIDASOrientationUtils.h"
#include "mitkMIDASViewKeyPressResponder.h"
#include "QmitkMIDASSingleViewWidget.h"

//-----------------------------------------------------------------------------
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
, m_Show2DCursorsCheckBox(NULL)
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
, m_PinButton(NULL)
, m_ControlWidget(NULL)
, m_ControlWidgetLayout(NULL)
, m_PopupWidget(NULL)
, m_ControlsContainerWidget(NULL)
, m_VisibilityManager(visibilityManager)
, m_DataStorage(dataStorage)
, m_RenderingManager(renderingManager)
, m_FocusManagerObserverTag(0)
, m_SelectedViewIndex(0)
, m_DefaultNumberOfRows(defaultNumberOfRows)
, m_DefaultNumberOfColumns(defaultNumberOfColumns)
, m_Show2DCursors(false)
, m_Show3DWindowInOrthoView(false)
, m_RememberViewSettingsPerOrientation(false)
, m_IsThumbnailMode(false)
, m_IsMIDASSegmentationMode(false)
, m_NavigationControllerEventListening(false)
, m_SingleWindowLayout(MIDAS_VIEW_CORONAL)
, m_MultiWindowLayout(MIDAS_VIEW_ORTHO)
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

  m_ControlWidget = new QWidget(this);
  m_ControlWidget->setContentsMargins(0, 0, 0, 0);
  m_ControlWidgetLayout = new QVBoxLayout(m_ControlWidget);
  m_ControlWidgetLayout->setContentsMargins(0, 0, 0, 0);
  m_ControlWidgetLayout->setSpacing(0);
  m_ControlWidget->setLayout(m_ControlWidgetLayout);

  m_PopupWidget = new ctkPopupWidget(m_ControlWidget);
  m_PopupWidget->setOrientation(Qt::Vertical);
  m_PopupWidget->setAnimationEffect(ctkBasePopupWidget::ScrollEffect);
  m_PopupWidget->setHorizontalDirection(Qt::LeftToRight);
  m_PopupWidget->setVerticalDirection(ctkBasePopupWidget::TopToBottom);
  m_PopupWidget->setAutoShow(true);
  m_PopupWidget->setAutoHide(true);
  m_PopupWidget->setEffectDuration(100);
  m_PopupWidget->setContentsMargins(5, 5, 5, 5);
  m_PopupWidget->setLineWidth(0);

  QPalette popupPalette = this->palette();
  QColor windowColor = popupPalette.color(QPalette::Window);
  windowColor.setAlpha(128);
  popupPalette.setColor(QPalette::Window, windowColor);
  m_PopupWidget->setPalette(popupPalette);
  m_PopupWidget->setAttribute(Qt::WA_TranslucentBackground, true);

  int buttonRowHeight = 15;
  m_PinButton = new QToolButton(m_ControlWidget);
  m_PinButton->setContentsMargins(0, 0, 0, 0);
  m_PinButton->setCheckable(true);
  m_PinButton->setAutoRaise(true);
  m_PinButton->setFixedHeight(16);
  QSizePolicy pinButtonSizePolicy;
  pinButtonSizePolicy.setHorizontalPolicy(QSizePolicy::Expanding);
  m_PinButton->setSizePolicy(pinButtonSizePolicy);
  // These two lines ensure that the icon appears on the left on each platform.
  m_PinButton->setText(" ");
  m_PinButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

  QIcon pinButtonIcon;
  pinButtonIcon.addFile(":/PushPinIn.png", QSize(), QIcon::Normal, QIcon::On);
  pinButtonIcon.addFile(":/PushPinOut.png", QSize(), QIcon::Normal, QIcon::Off);
  m_PinButton->setIcon(pinButtonIcon);

  QObject::connect(m_PinButton, SIGNAL(toggled(bool)),
                   this, SLOT(OnPinButtonToggled(bool)));
  m_ControlWidget->installEventFilter(this);

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
  m_LayoutForRenderWindows->setSpacing(0);

  m_LayoutForDropWidgets = new QVBoxLayout();
  m_LayoutForDropWidgets->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForDropWidgets"));
  m_LayoutForDropWidgets->setContentsMargins(0, 0, 0, 0);
  m_LayoutForDropWidgets->setSpacing(0);

  m_LayoutForLayoutWidgets = new QGridLayout();
  m_LayoutForLayoutWidgets->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForLayoutWidgets"));
  m_LayoutForLayoutWidgets->setContentsMargins(0, 0, 0, 0);
  m_LayoutForLayoutWidgets->setVerticalSpacing(0);
  m_LayoutForLayoutWidgets->setHorizontalSpacing(2);

  m_LayoutForTopControls = new QGridLayout();
  m_LayoutForTopControls->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForTopControls"));
  m_LayoutForTopControls->setContentsMargins(0, 0, 0, 0);
  m_LayoutForTopControls->setVerticalSpacing(0);
  m_LayoutForTopControls->setHorizontalSpacing(5);

  m_MIDASSlidersWidget = new QmitkMIDASSlidersWidget(m_ControlsContainerWidget);
  m_MIDASSlidersWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

  m_MIDASOrientationWidget = new QmitkMIDASOrientationWidget(m_ControlsContainerWidget);

  m_Show2DCursorsCheckBox = new QCheckBox(m_ControlsContainerWidget);
  m_Show2DCursorsCheckBox->setText("show cursors");

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

  m_LayoutForTopControls->addWidget(m_MIDASSlidersWidget,     0, 0, 2, 1);
  m_LayoutForTopControls->addLayout(m_LayoutForLayoutWidgets, 0, 1, 2, 1);
  m_LayoutForTopControls->addWidget(m_MIDASOrientationWidget, 0, 2, 1, 1);
  m_LayoutForTopControls->addLayout(m_LayoutForDropWidgets,   0, 3, 2, 1);
  m_LayoutForTopControls->addWidget(m_MIDASBindWidget,        0, 4, 2, 1);
  m_LayoutForTopControls->addWidget(m_Show2DCursorsCheckBox,  1, 2, 1, 1);

  m_LayoutForTopControls->setColumnMinimumWidth(0, 50);
  m_LayoutForTopControls->setColumnStretch(0, 5);
  m_LayoutForTopControls->setColumnStretch(1, 1);
  m_LayoutForTopControls->setColumnStretch(2, 0);
  m_LayoutForTopControls->setColumnStretch(3, 0);
  m_LayoutForTopControls->setColumnStretch(4, 0);

  m_ControlWidgetLayout->addWidget(m_PinButton);

  m_LayoutForGroupingControls->addLayout(m_LayoutForTopControls);
  m_LayoutToPutControlsOnTopOfWindows->addWidget(m_ControlWidget, 0, 0);
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
  connect(m_Show2DCursorsCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnShow2DCursorsCheckBoxToggled(bool)));
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


//-----------------------------------------------------------------------------
QmitkMIDASMultiViewWidget::~QmitkMIDASMultiViewWidget()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }
  this->Deactivated();
}


//-----------------------------------------------------------------------------
QmitkMIDASSingleViewWidget* QmitkMIDASMultiViewWidget::CreateSingleViewWidget()
{
  QmitkMIDASSingleViewWidget *widget = new QmitkMIDASSingleViewWidget(tr("QmitkRenderWindow"),
                                                                      -5, 20,
                                                                      this,
                                                                      m_RenderingManager,
                                                                      m_DataStorage);
  widget->setObjectName(tr("QmitkMIDASSingleViewWidget"));
  widget->setVisible(false);

  widget->SetBackgroundColor(m_BackgroundColour);
  widget->SetShow3DWindowInOrthoView(m_Show3DWindowInOrthoView);
  widget->SetRememberViewSettingsPerOrientation(m_RememberViewSettingsPerOrientation);
  widget->SetDisplayInteractionEnabled(true);

  connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), m_VisibilityManager, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)));
  connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)));
  connect(widget, SIGNAL(CrossPositionChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, int)), this, SLOT(OnCrossPositionChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, int)));
  connect(widget, SIGNAL(CentreChanged(QmitkMIDASSingleViewWidget*, const mitk::Vector3D&)), this, SLOT(OnCentreChanged(QmitkMIDASSingleViewWidget*, const mitk::Vector3D&)));
  connect(widget, SIGNAL(MagnificationFactorChanged(QmitkMIDASSingleViewWidget*, double)), this, SLOT(OnMagnificationFactorChanged(QmitkMIDASSingleViewWidget*, double)));

  return widget;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::RequestUpdateAll()
{
  QList<int> listToUpdate = this->GetViewIndexesToUpdate(true);
  for (int i = 0; i < listToUpdate.size(); i++)
  {
    if (listToUpdate[i] >= 0 && listToUpdate[i] < this->m_SingleViewWidgets.size())
    {
      m_SingleViewWidgets[listToUpdate[i]]->RequestUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetDefaultInterpolationType(MIDASDefaultInterpolationType interpolationType)
{
  m_VisibilityManager->SetDefaultInterpolationType(interpolationType);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetDefaultViewType(MIDASView midasView)
{
  m_VisibilityManager->SetDefaultViewType(midasView);
  if (::IsSingleWindowLayout(midasView))
  {
    this->SetDefaultSingleWindowLayout(midasView);
  }
  else
  {
    this->SetDefaultMultiWindowLayout(midasView);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetDefaultSingleWindowLayout(MIDASView midasView)
{
  m_SingleWindowLayout = midasView;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetDefaultMultiWindowLayout(MIDASView midasView)
{
  m_MultiWindowLayout = midasView;
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShowDropTypeWidgets(bool visible)
{
  m_DropSingleRadioButton->setVisible(visible);
  m_DropMultipleRadioButton->setVisible(visible);
  m_DropThumbnailRadioButton->setVisible(visible);
  m_DropAccumulateCheckBox->setVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShowLayoutButtons(bool visible)
{
  m_1x1LayoutButton->setVisible(visible);
  m_1x2LayoutButton->setVisible(visible);
  m_1x3LayoutButton->setVisible(visible);
  m_2x2LayoutButton->setVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShowMagnificationSlider(bool visible)
{
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->setVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShow2DCursors(bool visible)
{
  m_Show2DCursors = visible;

  bool wasBlocked = m_Show2DCursorsCheckBox->blockSignals(true);
  m_Show2DCursorsCheckBox->setChecked(visible);
  m_Show2DCursorsCheckBox->blockSignals(wasBlocked);

  this->Update2DCursorVisibility();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::GetShow2DCursors() const
{
  return m_Show2DCursors;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::GetShow3DWindowInOrthoView() const
{
  return m_Show3DWindowInOrthoView;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShow3DWindowInOrthoView(bool enabled)
{
  m_Show3DWindowInOrthoView = enabled;
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetShow3DWindowInOrthoView(enabled);
  }
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetRememberViewSettingsPerOrientation(bool rememberViewSettingsPerOrientation)
{
  m_RememberViewSettingsPerOrientation = rememberViewSettingsPerOrientation;
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetRememberViewSettingsPerOrientation(rememberViewSettingsPerOrientation);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::EnableSliderWidgets(bool enabled)
{
  m_MIDASSlidersWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::EnableOrientationWidgets(bool enabled)
{
  m_MIDASOrientationWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::EnableBindWidgets(bool enabled)
{
  m_MIDASBindWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::EnableDropTypeWidgets(bool enabled)
{
  m_LayoutForDropWidgets->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::EnableLayoutWidgets(bool enabled)
{
//  m_LayoutForLayoutWidgets->setEnabled(enabled);
  m_1x1LayoutButton->setEnabled(enabled);
  m_1x2LayoutButton->setEnabled(enabled);
  m_1x3LayoutButton->setEnabled(enabled);
  m_2x2LayoutButton->setEnabled(enabled);
  m_RowsLabel->setEnabled(enabled);
  m_RowsSpinBox->setEnabled(enabled);
  m_ColumnsLabel->setEnabled(enabled);
  m_ColumnsSpinBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::EnableWidgets(bool enabled)
{
  this->EnableDropTypeWidgets(enabled);
  this->EnableSliderWidgets(enabled);
  this->EnableLayoutWidgets(enabled);
  this->EnableOrientationWidgets(enabled);
  this->EnableBindWidgets(enabled);
  m_Show2DCursorsCheckBox->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
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
    m_Show2DCursorsCheckBox->setEnabled(false);
    this->SetLayoutSize(m_MaxRows, m_MaxCols, true);
  }
  else
  {
    this->EnableSliderWidgets(true);
    this->EnableLayoutWidgets(true);
    this->EnableOrientationWidgets(true);
    this->EnableBindWidgets(true);
    m_Show2DCursorsCheckBox->setEnabled(true);
    this->SetLayoutSize(m_NumberOfRowsInNonThumbnailMode, m_NumberOfColumnsInNonThumbnailMode, false);
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::GetThumbnailMode() const
{
  return this->m_IsThumbnailMode;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetMIDASSegmentationMode(bool enabled)
{
  this->m_IsMIDASSegmentationMode = enabled;

  if (enabled)
  {
    this->m_NumberOfRowsBeforeSegmentationMode = m_RowsSpinBox->value();
    this->m_NumberOfColumnsBeforeSegmentationMode = m_ColumnsSpinBox->value();
    this->EnableLayoutWidgets(false);
    this->EnableBindWidgets(false);
    m_Show2DCursorsCheckBox->setEnabled(false);
    this->SetLayoutSize(1, 1, false);
    this->SetSelectedViewIndex(0);
    this->UpdateFocusManagerToSelectedView();
  }
  else
  {
    this->EnableLayoutWidgets(true);
    this->EnableBindWidgets(true);
    m_Show2DCursorsCheckBox->setEnabled(true);
    this->SetLayoutSize(m_NumberOfRowsBeforeSegmentationMode, m_NumberOfColumnsBeforeSegmentationMode, false);
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::GetMIDASSegmentationMode() const
{
  return this->m_IsMIDASSegmentationMode;
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetBackgroundColour(QColor backgroundColour)
{
  m_BackgroundColour = backgroundColour;

  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetBackgroundColor(m_BackgroundColour);
  }

  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetLayoutSize(int numberOfRows, int numberOfColumns, bool isThumbnailMode)
{
  // Work out required number of widgets, and hence if we need to create any new ones.
  int requiredNumberOfViews = numberOfRows * numberOfColumns;
  int currentNumberOfViews = m_SingleViewWidgets.size();

  // If we have the right number of widgets, there is nothing to do, so early exit.
  if (requiredNumberOfViews == currentNumberOfViews)
  {
    return;
  }

  /////////////////////////////////////////
  // Start: Rebuild the number of widgets.
  // NOTE:  The order of widgets in
  //        m_SingleViewWidgets and
  //        m_VisibilityManager must match.
  /////////////////////////////////////////

  if (requiredNumberOfViews > currentNumberOfViews)
  {
    // create some more widgets
    int additionalViews = requiredNumberOfViews - m_SingleViewWidgets.size();
    for (int i = 0; i < additionalViews; i++)
    {
      QmitkMIDASSingleViewWidget *view = this->CreateSingleViewWidget();
      view->hide();

      this->m_SingleViewWidgets.push_back(view);
      this->m_VisibilityManager->RegisterWidget(view);
      this->m_VisibilityManager->SetAllNodeVisibilityForWindow(currentNumberOfViews + i, false);
    }
  }
  else if (requiredNumberOfViews < currentNumberOfViews)
  {
    // destroy surplus widgets
    this->m_VisibilityManager->DeRegisterWidgets(requiredNumberOfViews, m_SingleViewWidgets.size() - 1);

    for (int i = requiredNumberOfViews; i < m_SingleViewWidgets.size(); i++)
    {
      delete m_SingleViewWidgets[i];
    }

    m_SingleViewWidgets.erase(m_SingleViewWidgets.begin() + requiredNumberOfViews,
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
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
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

  int widgetCounter = 0;
  for (int r = 0; r < numberOfRows; r++)
  {
    for (int c = 0; c < numberOfColumns; c++)
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
  int selectedViewIndex = this->GetSelectedViewIndex();
  if (this->GetRowFromIndex(selectedViewIndex) >= numberOfRows || this->GetColumnFromIndex(selectedViewIndex) >= numberOfColumns)
  {
    selectedViewIndex = 0;
  }
  // Pass NULL for the selected vtkRenderWindow, to make sure that new windows don't look selected
  this->SwitchWindows(selectedViewIndex, NULL);

  // Now the number of views has changed, we need to make sure they are all in synch with all the right properties.
  this->Update2DCursorVisibility();
  this->SetShow3DWindowInOrthoView(this->m_Show3DWindowInOrthoView);

  // Make sure that if we are bound, we re-synch the geometry, or magnification.
  if (this->m_MIDASBindWidget->IsGeometryBound())
  {
    this->UpdateBoundGeometry(true);
  }
  if (this->m_MIDASBindWidget->IsMagnificationBound())
  {
    this->UpdateBoundMagnification();
  }
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetRowFromIndex(int i) const
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


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetColumnFromIndex(int i) const
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


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetIndexFromRowAndColumn(int r, int c) const
{
  return r*m_MaxCols + c;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::On1x1ButtonPressed()
{
  this->SetLayoutSize(1,1, false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::On1x2ButtonPressed()
{
  this->SetLayoutSize(1,2, false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::On1x3ButtonPressed()
{
  this->SetLayoutSize(1,3, false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::On2x2ButtonPressed()
{
  this->SetLayoutSize(2,2, false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnRowsSliderValueChanged(int r)
{
  this->SetLayoutSize(r, m_ColumnsSpinBox->value(), false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnColumnsSliderValueChanged(int c)
{
  this->SetLayoutSize(m_RowsSpinBox->value(), c, false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnCrossPositionChanged(QmitkMIDASSingleViewWidget *view, QmitkRenderWindow* renderWindow, int sliceNumber)
{
  // If the view is not found, we do not do anything.
  if (std::find(m_SingleViewWidgets.begin(), m_SingleViewWidgets.end(), view) == m_SingleViewWidgets.end())
  {
    return;
  }

  std::vector<QmitkRenderWindow*> renderWindows = view->GetVisibleRenderWindows();
  if (renderWindows.size() == 1 &&
      renderWindow == renderWindows[0] &&
      sliceNumber != m_MIDASSlidersWidget->m_SliceSelectionWidget->value())
  {
    // This should only be used to update the sliceNumber on the GUI, so must not trigger a further update.
    bool wasBlocked = m_MIDASSlidersWidget->m_SliceSelectionWidget->blockSignals(true);
    m_MIDASSlidersWidget->m_SliceSelectionWidget->setValue(sliceNumber);
    m_MIDASSlidersWidget->m_SliceSelectionWidget->blockSignals(wasBlocked);
  }
  mitk::Point3D crossPosition = view->GetCrossPosition();
  if (m_MIDASBindWidget->AreCursorsBound())
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != view)
      {
        m_SingleViewWidgets[i]->SetCrossPosition(crossPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnCentreChanged(QmitkMIDASSingleViewWidget *widget, const mitk::Vector3D& centre)
{
  if (this->m_MIDASBindWidget->IsMagnificationBound())
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != widget)
      {
        m_SingleViewWidgets[i]->SetCentre(centre);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnMagnificationFactorChanged(QmitkMIDASSingleViewWidget *widget, double magnificationFactor)
{
  bool wasBlocked = m_MIDASSlidersWidget->m_MagnificationFactorWidget->blockSignals(true);
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->setValue(magnificationFactor);
  m_MIDASSlidersWidget->m_MagnificationFactorWidget->blockSignals(wasBlocked);

  if (this->m_MIDASBindWidget->IsMagnificationBound())
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != widget)
      {
        m_SingleViewWidgets[i]->SetMagnificationFactor(magnificationFactor);
      }
    }
  }
  m_PreviousMagnificationFactor = magnificationFactor;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnNodesDropped(QmitkRenderWindow *renderWindow, std::vector<mitk::DataNode*> nodes)
{
  // See also QmitkMIDASMultiViewVisibilityManager::OnNodesDropped which should trigger first.
  if (!this->m_DropThumbnailRadioButton->isChecked())
  {
    this->EnableWidgets(true);
  }

  QmitkMIDASSingleViewWidget* selectedView = NULL;

  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    QmitkMIDASSingleViewWidget* view = m_SingleViewWidgets[i];
    if (view->ContainsRenderWindow(renderWindow))
    {
      selectedView = view;
      MIDASOrientation orientation = selectedView->GetOrientation();
      //  MIDASView midasView = selectedView->GetMIDASView();
      switch (orientation)
      {
      case MIDAS_ORIENTATION_AXIAL:
        renderWindow = selectedView->GetAxialWindow();
        break;
      case MIDAS_ORIENTATION_SAGITTAL:
        renderWindow = selectedView->GetSagittalWindow();
        break;
      case MIDAS_ORIENTATION_CORONAL:
        renderWindow = selectedView->GetCoronalWindow();
        break;
      case MIDAS_ORIENTATION_UNKNOWN:
        renderWindow = selectedView->Get3DWindow();
        break;
      }
      break;
    }
  }

  // This does not trigger OnFocusChanged() the very first time, as when creating the editor, the first widget already has focus.
  mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(renderWindow->GetRenderer());

  double magnificationFactor = selectedView->GetMagnificationFactor();

  m_MIDASSlidersWidget->m_MagnificationFactorWidget->setValue(magnificationFactor);

  MIDASView midasView = selectedView->GetView();
  m_MIDASOrientationWidget->SetView(midasView);

  this->Update2DCursorVisibility();
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SwitchWindows(int selectedViewIndex, QmitkRenderWindow *selectedRenderWindow)
{
  if (selectedViewIndex >= 0 && selectedViewIndex < m_SingleViewWidgets.size())
  {
    QmitkMIDASSingleViewWidget* selectedView = this->m_SingleViewWidgets[selectedViewIndex];

    // This, to turn off borders on all other windows.
    this->SetSelectedViewIndex(selectedViewIndex);

    // This to specifically set the border round one sub-pane for if its an ortho-view.
    if (selectedRenderWindow != NULL)
    {
      int numberOfNodes = m_VisibilityManager->GetNodesInWindow(selectedViewIndex);
      if (numberOfNodes > 0)
      {
        selectedView->SetSelectedRenderWindow(selectedRenderWindow);
      }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Need to enable widgets appropriately, so user can't press stuff that they aren't meant to.
    /////////////////////////////////////////////////////////////////////////////////////////////
    MIDASOrientation orientation = selectedView->GetOrientation();
    MIDASView midasView = selectedView->GetView();

    bool slidersWidgetWasBlocked = m_MIDASSlidersWidget->BlockSignals(true);
    bool orientationWidgetWasBlocked = m_MIDASOrientationWidget->blockSignals(true);

    if (midasView != MIDAS_VIEW_UNKNOWN)
    {
      m_MIDASOrientationWidget->SetView(midasView);
    }
    if (orientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      unsigned int minSlice = selectedView->GetMinSlice(orientation);
      unsigned int maxSlice = selectedView->GetMaxSlice(orientation);
      unsigned int currentSlice = selectedView->GetSliceNumber(orientation);

      m_MIDASSlidersWidget->m_SliceSelectionWidget->setMinimum(minSlice);
      m_MIDASSlidersWidget->m_SliceSelectionWidget->setMaximum(maxSlice);
      m_MIDASSlidersWidget->m_SliceSelectionWidget->setValue(currentSlice);
    }

    double minMag = std::ceil(selectedView->GetMinMagnification());
    double maxMag = std::floor(selectedView->GetMaxMagnification());
    double currentMag = selectedView->GetMagnificationFactor();
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setMinimum(minMag);
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setMaximum(maxMag);
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setValue(currentMag);

    unsigned int minTime = selectedView->GetMinTime();
    unsigned int maxTime = selectedView->GetMaxTime();
    unsigned int currentTime = selectedView->GetTime();
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setMinimum(minTime);
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setMaximum(maxTime);
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setValue(currentTime);

    m_MIDASSlidersWidget->m_SliceSelectionWidget->setEnabled(true);
    m_MIDASSlidersWidget->m_TimeSelectionWidget->setEnabled(true);
    m_MIDASSlidersWidget->m_MagnificationFactorWidget->setEnabled(true);

    m_MIDASSlidersWidget->BlockSignals(slidersWidgetWasBlocked);
    m_MIDASOrientationWidget->blockSignals(orientationWidgetWasBlocked);

    this->Update2DCursorVisibility();
  }
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetFocus()
{
  /*
  if (m_SelectedViewIndex != -1)
  {
    m_SingleViewWidgets[m_SelectedViewIndex]->setFocus();
  }
  else
  {
    m_SingleViewWidgets[0]->setFocus();
  }
  */
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer::ConstPointer baseRenderer = focusManager->GetFocused();

  int selectedViewIndex = -1;
  vtkRenderWindow* focusedVtkRenderWindow = NULL;
  QmitkRenderWindow* focusedRenderWindow = NULL;

  if (baseRenderer.IsNotNull())
  {
    focusedVtkRenderWindow = baseRenderer->GetRenderWindow();
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      QmitkRenderWindow* renderWindow = m_SingleViewWidgets[i]->GetRenderWindow(focusedVtkRenderWindow);
      if (renderWindow != NULL)
      {
        selectedViewIndex = i;
        focusedRenderWindow = renderWindow;
        break;
      }
    }
  }
  this->SwitchWindows(selectedViewIndex, focusedRenderWindow);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnDropSingleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_SINGLE);
    this->SetThumbnailMode(false);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnDropMultipleRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_MULTIPLE);
    this->SetThumbnailMode(false);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnDropThumbnailRadioButtonToggled(bool toggled)
{
  if (toggled)
  {
    m_VisibilityManager->ClearAllWindows();
    m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_ALL);
    this->SetThumbnailMode(true);
  }
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
QList<int> QmitkMIDASMultiViewWidget::GetViewIndexesToUpdate(bool doAllVisible) const
{
  QList<int> result;

  if (doAllVisible)
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i]->isVisible())
      {
        result.push_back(i);
      }
    }
  }
  else
  {
    int windowIndex = this->GetSelectedViewIndex();
    result.push_back(windowIndex);
  }
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::MoveAnterior()
{
  return this->MoveAnteriorPosterior(true, 1);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::MovePosterior()
{
  return this->MoveAnteriorPosterior(false, 1);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::MoveAnteriorPosterior(bool moveAnterior, int slices)
{
  bool actuallyDidSomething = false;
  int selectedViewIndex = this->GetSelectedViewIndex();

  if (selectedViewIndex != -1)
  {
    QmitkMIDASSingleViewWidget* selectedView = this->m_SingleViewWidgets[selectedViewIndex];

    MIDASOrientation orientation = selectedView->GetOrientation();
    unsigned int currentSlice = selectedView->GetSliceNumber(orientation);
    unsigned int minSlice = selectedView->GetMinSlice(orientation);
    unsigned int maxSlice = selectedView->GetMaxSlice(orientation);

    int upDirection = selectedView->GetSliceUpDirection(orientation);
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


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnSliceNumberChanged(double sliceNumber)
{
  this->SetSelectedWindowSliceNumber(static_cast<int>(sliceNumber));
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowSliceNumber(int sliceNumber)
{
  int selectedViewIndex = this->GetSelectedViewIndex();
  MIDASOrientation orientation = this->m_SingleViewWidgets[selectedViewIndex]->GetOrientation();

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(this->m_MIDASBindWidget->IsGeometryBound());
    for (int i = 0; i < viewsToUpdate.size(); i++)
    {
      this->m_SingleViewWidgets[viewsToUpdate[i]]->SetSliceNumber(orientation, sliceNumber);
    }
  }
  else
  {
    MITK_WARN << "Found an invalid orientation in view widget " << selectedViewIndex << ", so ignoring request to change to slice " << sliceNumber << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnMagnificationFactorChanged(double magnificationFactor)
{
  double roundedMagnificationFactor = std::floor(magnificationFactor);

  // If we are between two integers, we raise a new event:
  if (magnificationFactor != roundedMagnificationFactor)
  {
    double newMagnificationFactor = roundedMagnificationFactor;
    // If the value has decreased, we have to increase the rounded value.
    if (magnificationFactor < m_PreviousMagnificationFactor)
    {
      newMagnificationFactor += 1.0;
    }

    this->m_MIDASSlidersWidget->m_MagnificationFactorWidget->setValue(newMagnificationFactor);
  }
  else
  {
    this->SetSelectedWindowMagnification(magnificationFactor);
    m_PreviousMagnificationFactor = magnificationFactor;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowMagnification(double magnificationFactor)
{
  QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(this->m_MIDASBindWidget->IsMagnificationBound());
  for (int i = 0; i < viewsToUpdate.size(); i++)
  {
    this->m_SingleViewWidgets[viewsToUpdate[i]]->SetMagnificationFactor(magnificationFactor);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnTimeChanged(double timeStep)
{
  this->SetSelectedTimeStep(static_cast<int>(timeStep));
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedTimeStep(int timeStep)
{
  QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(this->m_DropThumbnailRadioButton->isChecked());
  for (int i = 0; i < viewsToUpdate.size(); i++)
  {
    this->m_SingleViewWidgets[viewsToUpdate[i]]->SetTime(timeStep);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnOrientationSelected(MIDASView midasView)
{
  if (midasView != MIDAS_VIEW_UNKNOWN)
  {
    this->SwitchMIDASView(midasView);

    // Update the focus to the selected window, to trigger things like thumbnail viewer refresh
    // (or indeed anything that's listening to the FocusManager).
    this->UpdateFocusManagerToSelectedView();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnShow2DCursorsCheckBoxToggled(bool checked)
{
  this->SetShow2DCursors(checked);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::UpdateFocusManagerToSelectedView()
{
  int selectedViewIndex = this->GetSelectedViewIndex();
  std::vector<QmitkRenderWindow*> renderWindows = this->m_SingleViewWidgets[selectedViewIndex]->GetVisibleRenderWindows();

  if (renderWindows.size() > 0)
  {
    mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(renderWindows[0]->GetRenderer());
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchToAxial()
{
  this->SetSelectedWindowToAxial();

  bool wasBlocked = m_MIDASOrientationWidget->blockSignals(true);
  m_MIDASOrientationWidget->SetView(MIDAS_VIEW_AXIAL);
  m_MIDASOrientationWidget->blockSignals(wasBlocked);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowToAxial()
{
  this->SwitchMIDASView(MIDAS_VIEW_AXIAL);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchToSagittal()
{
  this->SetSelectedWindowToSagittal();

  bool wasBlocked = m_MIDASOrientationWidget->blockSignals(true);
  m_MIDASOrientationWidget->SetView(MIDAS_VIEW_SAGITTAL);
  m_MIDASOrientationWidget->blockSignals(wasBlocked);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowToSagittal()
{
  this->SwitchMIDASView(MIDAS_VIEW_SAGITTAL);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchToCoronal()
{
  this->SetSelectedWindowToCoronal();

  bool wasBlocked = m_MIDASOrientationWidget->blockSignals(true);
  m_MIDASOrientationWidget->SetView(MIDAS_VIEW_CORONAL);
  m_MIDASOrientationWidget->blockSignals(wasBlocked);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowToCoronal()
{
  this->SwitchMIDASView(MIDAS_VIEW_CORONAL);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchTo3D()
{
  this->SetSelectedWindowTo3D();

  bool wasBlocked = m_MIDASOrientationWidget->blockSignals(true);
  m_MIDASOrientationWidget->SetView(MIDAS_VIEW_3D);
  m_MIDASOrientationWidget->blockSignals(wasBlocked);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowTo3D()
{
  this->SwitchMIDASView(MIDAS_VIEW_3D);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::ToggleMultiWindowLayout()
{
  QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[this->GetSelectedViewIndex()];
  MIDASView currentMidasView = selectedView->GetView();
  MIDASView nextMidasView;

  if (::IsSingleWindowLayout(currentMidasView))
  {
    nextMidasView = m_MultiWindowLayout;
  }
  else
  {
    switch (selectedView->GetOrientation())
    {
    case MIDAS_ORIENTATION_AXIAL:
      nextMidasView = MIDAS_VIEW_AXIAL;
      break;
    case MIDAS_ORIENTATION_SAGITTAL:
      nextMidasView = MIDAS_VIEW_SAGITTAL;
      break;
    case MIDAS_ORIENTATION_CORONAL:
      nextMidasView = MIDAS_VIEW_CORONAL;
      break;
    case MIDAS_ORIENTATION_UNKNOWN:
      nextMidasView = MIDAS_VIEW_3D;
      break;
    default:
      nextMidasView = MIDAS_VIEW_CORONAL;
    }
  }

  // Note that we do not block the signals here, so this->SwitchMIDASView(nextMidasView) will
  // be called.
  m_MIDASOrientationWidget->SetView(nextMidasView);

  return true;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SwitchMIDASView(MIDASView midasView)
{
  int selectedViewIndex = this->GetSelectedViewIndex();

  QList<int> viewIndexesToUpdate = this->GetViewIndexesToUpdate(this->m_MIDASBindWidget->IsGeometryBound());
  for (int i = 0; i < viewIndexesToUpdate.size(); i++)
  {
    int viewIndexToUpdate = viewIndexesToUpdate[i];
    QmitkMIDASSingleViewWidget* viewToUpdate = this->m_SingleViewWidgets[viewIndexToUpdate];
    viewToUpdate->SetView(midasView, false);

    if (viewIndexToUpdate == selectedViewIndex)
    {
      if (midasView == MIDAS_VIEW_AXIAL)
      {
        viewToUpdate->SetSelectedRenderWindow(viewToUpdate->GetAxialWindow());
      }
      else if (midasView == MIDAS_VIEW_SAGITTAL)
      {
        viewToUpdate->SetSelectedRenderWindow(viewToUpdate->GetSagittalWindow());
      }
      else if (midasView == MIDAS_VIEW_CORONAL)
      {
        viewToUpdate->SetSelectedRenderWindow(viewToUpdate->GetCoronalWindow());
      }
    }
  }

  if (::IsSingleWindowLayout(midasView))
  {
    m_SingleWindowLayout = midasView;
  }
  else
  {
    m_MultiWindowLayout = midasView;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::Update2DCursorVisibility()
{
  QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(true);
  for (int i = 0; i < viewsToUpdate.size(); i++)
  {
    bool globalVisibility = false;
    bool localVisibility = m_Show2DCursors;

    m_SingleViewWidgets[viewsToUpdate[i]]->SetDisplay2DCursorsGlobally(globalVisibility);
    m_SingleViewWidgets[viewsToUpdate[i]]->SetDisplay2DCursorsLocally(localVisibility);
  }

  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::UpdateBoundGeometry(bool isBoundNow)
{
  int selectedViewIndex = this->GetSelectedViewIndex();
  QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];

  mitk::Geometry3D::Pointer selectedGeometry = selectedView->GetGeometry();
  MIDASOrientation orientation               = selectedView->GetOrientation();
  MIDASView midasView                        = selectedView->GetView();
  int sliceNumber                            = selectedView->GetSliceNumber(orientation);
  int magnification                          = selectedView->GetMagnificationFactor();
  int timeStepNumber                         = selectedView->GetTime();

  QList<int> viewIndexesToUpdate = this->GetViewIndexesToUpdate(isBoundNow);
  for (int i = 0; i < viewIndexesToUpdate.size(); i++)
  {
    int viewIndexToUpdate = viewIndexesToUpdate[i];
    QmitkMIDASSingleViewWidget* viewToUpdate = m_SingleViewWidgets[viewIndexToUpdate];
    viewToUpdate->SetBoundGeometry(selectedGeometry);
    viewToUpdate->SetBoundGeometryActive(isBoundNow);
    viewToUpdate->SetView(midasView, false);
    viewToUpdate->SetSliceNumber(orientation, sliceNumber);
    viewToUpdate->SetMagnificationFactor(magnification);
    viewToUpdate->SetTime(timeStepNumber);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::UpdateBoundMagnification()
{
//  int selectedViewIndex = this->GetSelectedViewIndex();
//  int magnification = m_SingleViewWidgets[selectedViewIndex]->GetMagnificationFactor();

//  QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(isBoundNow);
//  for (int i = 0; i < viewsToUpdate.size(); i++)
//  {
//    int viewIndex = viewsToUpdate[i];
//    m_SingleViewWidgets[viewIndex]->SetMagnificationFactor(magnification);
//  }
  int selectedViewIndex = this->GetSelectedViewIndex();
  QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];
  mitk::Vector3D centre = selectedView->GetCentre();
  double magnification = selectedView->GetMagnificationFactor();
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (i != selectedViewIndex)
    {
      m_SingleViewWidgets[i]->SetMagnificationFactor(magnification);
      m_SingleViewWidgets[i]->SetCentre(centre);
    }
  }
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetSliceNumber() const
{
  return this->m_MIDASSlidersWidget->m_SliceSelectionWidget->value();
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASMultiViewWidget::GetOrientation() const
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

  if (this->m_MIDASOrientationWidget->m_AxialWindowRadioButton->isChecked())
  {
    orientation = MIDAS_ORIENTATION_AXIAL;
  }
  else if (this->m_MIDASOrientationWidget->m_SagittalWindowRadioButton->isChecked())
  {
    orientation = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (this->m_MIDASOrientationWidget->m_CoronalWindowRadioButton->isChecked())
  {
    orientation = MIDAS_ORIENTATION_CORONAL;
  }
  else if (this->m_MIDASOrientationWidget->m_3DWindowRadioButton->isChecked())
  {
    orientation = MIDAS_ORIENTATION_UNKNOWN;
  }

  return orientation;
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetSelectedViewIndex() const
{
  int selectedViewIndex = m_SelectedViewIndex;
  if (selectedViewIndex < 0 || selectedViewIndex >= m_SingleViewWidgets.size())
  {
    // Default back to first view.
    selectedViewIndex = 0;
  }

  // Note the following specification.
  assert(selectedViewIndex >= 0);
  assert(selectedViewIndex < m_SingleViewWidgets.size());

  // Return a valid selected window index.
  return selectedViewIndex;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASMultiViewWidget::GetSelectedRenderWindow() const
{
  // NOTE: This MUST always return not-null.
  QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[this->GetSelectedViewIndex()];
  return selectedView->GetSelectedRenderWindow();
}


//-----------------------------------------------------------------------------
QHash<QString,QmitkRenderWindow*> QmitkMIDASMultiViewWidget::GetRenderWindows() const
{
  // NOTE: This MUST always return a non-empty map.

  QHash<QString, QmitkRenderWindow*> renderWindows;

  // See org.mitk.gui.qt.imagenavigator plugin.
  //
  // The assumption is that a QmitkStdMultiWidget has windows called
  // axial, sagittal, coronal, 3d.
  //
  // So, if we take the currently selected widget, and name these render windows
  // accordingly, then the MITK imagenavigator can be used to update it.

  int selectedViewIndex = this->GetSelectedViewIndex();
  QmitkMIDASSingleViewWidget* view = m_SingleViewWidgets[selectedViewIndex];

  renderWindows.insert("axial", view->GetAxialWindow());
  renderWindows.insert("sagittal", view->GetSagittalWindow());
  renderWindows.insert("coronal", view->GetCoronalWindow());
  renderWindows.insert("3d", view->Get3DWindow());

  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (i != selectedViewIndex)
    {
      QString id = tr(".%1").arg(i);

      view = m_SingleViewWidgets[i];
      renderWindows.insert("axial" + id, view->GetAxialWindow());
      renderWindows.insert("sagittal" + id, view->GetSagittalWindow());
      renderWindows.insert("coronal" + id, view->GetCoronalWindow());
      renderWindows.insert("3d" + id, view->Get3DWindow());
    }
  }

  return renderWindows;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkMIDASMultiViewWidget::GetRenderWindow(const QString& id) const
{
  QHash<QString,QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  QHash<QString,QmitkRenderWindow*>::iterator iter = renderWindows.find(id);
  if (iter != renderWindows.end())
  {
    return iter.value();
  }
  else
  {
    return NULL;
  }
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkMIDASMultiViewWidget::GetCrossPosition(const QString& id) const
{
  if (id.isNull())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    return m_SingleViewWidgets[selectedViewIndex]->GetCrossPosition();
  }
  else
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(id);
    if (renderWindow)
    {
      for (int i = 0; i < m_SingleViewWidgets.size(); ++i)
      {
        if (m_SingleViewWidgets[i]->ContainsRenderWindow(renderWindow))
        {
          return m_SingleViewWidgets[i]->GetCrossPosition();
        }
      }
    }
  }
  mitk::Point3D fallBackValue;
  fallBackValue.Fill(0.0);
  return fallBackValue;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetCrossPosition(const mitk::Point3D& crossPosition, const QString& id)
{
  if (id.isNull())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    m_SingleViewWidgets[selectedViewIndex]->SetCrossPosition(crossPosition);
  }
  else
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(id);
    if (renderWindow)
    {
      for (int i = 0; i < m_SingleViewWidgets.size(); ++i)
      {
        if (m_SingleViewWidgets[i]->ContainsRenderWindow(renderWindow))
        {
          m_SingleViewWidgets[i]->SetCrossPosition(crossPosition);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::Activated()
{
//  this->setEnabled(true);
  this->EnableLinkedNavigation(true);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::Deactivated()
{
//  this->setEnabled(false);
  this->EnableLinkedNavigation(false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::EnableLinkedNavigation(bool enable)
{
  this->SetNavigationControllerEventListening(enable);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::IsLinkedNavigationEnabled() const
{
  return this->GetNavigationControllerEventListening();
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::GetNavigationControllerEventListening() const
{
  return m_NavigationControllerEventListening;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetNavigationControllerEventListening(bool enabled)
{
  int selectedViewIndex = this->GetSelectedViewIndex();
  if (enabled && !this->m_NavigationControllerEventListening)
  {
    m_SingleViewWidgets[selectedViewIndex]->SetNavigationControllerEventListening(true);
  }
  else if (!enabled && this->m_NavigationControllerEventListening)
  {
    m_SingleViewWidgets[selectedViewIndex]->SetNavigationControllerEventListening(false);
  }
  this->m_NavigationControllerEventListening = enabled;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedViewIndex(int selectedViewIndex)
{
  if (selectedViewIndex >= 0 && selectedViewIndex < m_SingleViewWidgets.size())
  {
    m_SelectedViewIndex = selectedViewIndex;

    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      int nodesInWindow = m_VisibilityManager->GetNodesInWindow(i);

      if (i == selectedViewIndex && nodesInWindow > 0)
      {
        m_SingleViewWidgets[i]->SetSelected(true);
      }
      else
      {
        m_SingleViewWidgets[i]->SetSelected(false);
      }

      if (i == selectedViewIndex)
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
    MITK_WARN << "Ignoring request to set the selected window to window number " << selectedViewIndex << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnBindModeSelected(MIDASBindType bind)
{
  bool currentGeometryBound = m_SingleViewWidgets[0]->GetBoundGeometryActive();
  bool requestedGeometryBound = this->m_MIDASBindWidget->IsGeometryBound();

  if (currentGeometryBound != requestedGeometryBound)
  {
    this->UpdateBoundGeometry(this->m_MIDASBindWidget->IsGeometryBound());
  }

  if (this->m_MIDASBindWidget->AreCursorsBound())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];
    mitk::Point3D crossPosition = selectedView->GetCrossPosition();
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (i != selectedViewIndex)
      {
        m_SingleViewWidgets[i]->SetCrossPosition(crossPosition);
      }
    }
  }

  if (this->m_MIDASBindWidget->IsMagnificationBound())
  {
    //  this->UpdateBoundMagnification();

    int selectedViewIndex = this->GetSelectedViewIndex();
    QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];
    mitk::Vector3D centre = selectedView->GetCentre();
    double magnification = selectedView->GetMagnificationFactor();
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (i != selectedViewIndex)
      {
        m_SingleViewWidgets[i]->SetMagnificationFactor(magnification);
        m_SingleViewWidgets[i]->SetCentre(centre);
      }
    }
  }

  this->Update2DCursorVisibility();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnPopupOpened(bool opened)
{
  if (!opened)
  {
    QList<int> viewIndexesToUpdate = this->GetViewIndexesToUpdate(false);
    for (int i = 0; i < viewIndexesToUpdate.size(); i++)
    {
      int viewIndexToUpdate = viewIndexesToUpdate[i];
      m_SingleViewWidgets[viewIndexToUpdate]->repaint();
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSliceSelectTracking(bool isTracking)
{
  m_MIDASSlidersWidget->SetSliceTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetMagnificationSelectTracking(bool isTracking)
{
  m_MIDASSlidersWidget->SetMagnificationTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetTimeSelectTracking(bool isTracking)
{
  m_MIDASSlidersWidget->SetTimeTracking(isTracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnPinButtonToggled(bool checked)
{
  if (checked)
  {
    m_PopupWidget->pinPopup(true);
  }
  else
  {
    m_PopupWidget->setAutoHide(true);
  }
}


//---------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::eventFilter(QObject* object, QEvent* event)
{
  if (object == this->m_ControlWidget && event->type() == QEvent::Enter)
  {
    this->m_PopupWidget->showPopup();
  }
  return this->QObject::eventFilter(object, event);
}
