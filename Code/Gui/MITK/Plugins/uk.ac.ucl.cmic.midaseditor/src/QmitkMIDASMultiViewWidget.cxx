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

#include <mitkMIDASOrientationUtils.h>
#include <mitkMIDASViewKeyPressResponder.h>
#include <QmitkMIDASSingleViewWidget.h>

#include "QmitkMIDASMultiViewWidgetControlPanel.h"
#include "ui_QmitkMIDASMultiViewWidgetControlPanel.h"

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
, m_LayoutForRenderWindows(NULL)
, m_PinButton(NULL)
, m_PopupWidget(NULL)
, m_VisibilityManager(visibilityManager)
, m_DataStorage(dataStorage)
, m_RenderingManager(renderingManager)
, m_FocusManagerObserverTag(0)
, m_SelectedViewIndex(0)
, m_DefaultViewRows(defaultNumberOfRows)
, m_DefaultViewColumns(defaultNumberOfColumns)
, m_Show2DCursors(false)
, m_Show3DWindowInOrthoView(false)
, m_RememberSettingsPerLayout(false)
, m_IsThumbnailMode(false)
, m_IsMIDASSegmentationMode(false)
, m_NavigationControllerEventListening(false)
, m_Magnification(0.0)
, m_SingleWindowLayout(MIDAS_LAYOUT_CORONAL)
, m_MultiWindowLayout(MIDAS_LAYOUT_ORTHO)
, m_ControlPanel(0)
, m_ViewKeyPressStateMachine(0)
{
  assert(visibilityManager);

  this->setFocusPolicy(Qt::StrongFocus);

  /************************************
   * Create stuff.
   ************************************/

  m_TopLevelLayout = new QGridLayout(this);
  m_TopLevelLayout->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_TopLevelLayout"));
  m_TopLevelLayout->setContentsMargins(0, 0, 0, 0);
  m_TopLevelLayout->setSpacing(0);

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForRenderWindows"));
  m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
  m_LayoutForRenderWindows->setSpacing(0);

  QWidget* pinButtonWidget = new QWidget(this);
  pinButtonWidget->setContentsMargins(0, 0, 0, 0);
  QVBoxLayout* pinButtonWidgetLayout = new QVBoxLayout(pinButtonWidget);
  pinButtonWidgetLayout->setContentsMargins(0, 0, 0, 0);
  pinButtonWidgetLayout->setSpacing(0);
  pinButtonWidget->setLayout(pinButtonWidgetLayout);

  m_PopupWidget = new ctkPopupWidget(pinButtonWidget);
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
  m_PinButton = new QToolButton(this);
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
  m_PinButton->installEventFilter(this);

  m_ControlPanel = this->CreateControlPanel(m_PopupWidget);

  pinButtonWidgetLayout->addWidget(m_PinButton);

  m_TopLevelLayout->addWidget(pinButtonWidget, 0, 0);
  m_TopLevelLayout->setRowMinimumHeight(0, buttonRowHeight);
  m_TopLevelLayout->addLayout(m_LayoutForRenderWindows, 1, 0);

  /************************************
   * Now initialise stuff.
   ************************************/

  // Create/Connect the state machine
  m_ViewKeyPressStateMachine = mitk::MIDASViewKeyPressStateMachine::New("MIDASKeyPressStateMachine", this);
  mitk::GlobalInteraction::GetInstance()->AddListener(m_ViewKeyPressStateMachine);

  // Default to dropping into single window.
  m_ControlPanel->SetDropType(MIDAS_DROP_TYPE_SINGLE);
  m_VisibilityManager->SetDropType(MIDAS_DROP_TYPE_SINGLE);

  // We have the default rows and columns passed in via constructor args, in initialise list.
  m_ControlPanel->SetViewNumber(m_DefaultViewRows, m_DefaultViewColumns);
  this->SetViewNumber(m_DefaultViewRows, m_DefaultViewColumns, false);

  // Connect Qt Signals to make it all hang together.
  connect(m_ControlPanel, SIGNAL(SliceIndexChanged(int)), this, SLOT(OnSliceIndexChanged(int)));
  connect(m_ControlPanel, SIGNAL(TimeStepChanged(int)), this, SLOT(OnTimeStepChanged(int)));
  connect(m_ControlPanel, SIGNAL(MagnificationChanged(double)), this, SLOT(OnMagnificationChanged(double)));

  connect(m_ControlPanel, SIGNAL(LayoutChanged(MIDASLayout)), this, SLOT(OnLayoutChanged(MIDASLayout)));
  connect(m_ControlPanel, SIGNAL(CursorVisibilityChanged(bool)), this, SLOT(OnShow2DCursorsCheckBoxToggled(bool)));

  connect(m_ControlPanel, SIGNAL(ViewNumberChanged(int, int)), this, SLOT(OnViewNumberChanged(int, int)));

  connect(m_ControlPanel, SIGNAL(ViewBindingTypeChanged()), this, SLOT(OnViewBindingTypeChanged()));

  connect(m_ControlPanel, SIGNAL(DropTypeChanged(MIDASDropType)), this, SLOT(OnDropTypeChanged(MIDASDropType)));
  connect(m_ControlPanel, SIGNAL(DropAccumulateChanged(bool)), this, SLOT(OnDropAccumulateChanged(bool)));

  connect(m_PopupWidget, SIGNAL(popupOpened(bool)), this, SLOT(OnPopupOpened(bool)));

  // We listen to FocusManager to detect when things have changed focus, and hence to highlight the "current window".
  itk::SimpleMemberCommand<QmitkMIDASMultiViewWidget>::Pointer onFocusChangedCommand =
    itk::SimpleMemberCommand<QmitkMIDASMultiViewWidget>::New();
  onFocusChangedCommand->SetCallbackFunction( this, &QmitkMIDASMultiViewWidget::OnFocusChanged );

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
}


//-----------------------------------------------------------------------------
QmitkMIDASMultiViewWidgetControlPanel* QmitkMIDASMultiViewWidget::CreateControlPanel(QWidget* parent)
{
  QmitkMIDASMultiViewWidgetControlPanel* controlPanel = new QmitkMIDASMultiViewWidgetControlPanel(parent);

  controlPanel->SetMaxViewNumber(m_MaxViewRows, m_MaxViewColumns);

  QHBoxLayout* controlPanelLayout = new QHBoxLayout(parent);
  controlPanelLayout->setContentsMargins(0, 0, 0, 0);
  controlPanelLayout->setSpacing(0);
  controlPanelLayout->addWidget(controlPanel);

  return controlPanel;
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
  QmitkMIDASSingleViewWidget* widget = new QmitkMIDASSingleViewWidget(tr("QmitkRenderWindow"),
                                                                      -5, 20,
                                                                      this,
                                                                      m_RenderingManager,
                                                                      m_DataStorage);
  widget->setObjectName(tr("QmitkMIDASSingleViewWidget"));
  widget->setVisible(false);

  widget->SetBackgroundColor(m_BackgroundColour);
  widget->SetShow3DWindowInOrthoView(m_Show3DWindowInOrthoView);
  widget->SetRememberSettingsPerLayout(m_RememberSettingsPerLayout);
  widget->SetDisplayInteractionsEnabled(true);
  widget->SetPanningBound(true);
  widget->SetZoomingBound(true);

  connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), m_VisibilityManager, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)));
  connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)));
  connect(widget, SIGNAL(SelectedPositionChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, int)), this, SLOT(OnSelectedPositionChanged(QmitkMIDASSingleViewWidget*, QmitkRenderWindow*, int)));
  connect(widget, SIGNAL(CursorPositionChanged(QmitkMIDASSingleViewWidget*, const mitk::Vector3D&)), this, SLOT(OnCursorPositionChanged(QmitkMIDASSingleViewWidget*, const mitk::Vector3D&)));
  connect(widget, SIGNAL(MagnificationChanged(QmitkMIDASSingleViewWidget*, double)), this, SLOT(OnMagnificationChanged(QmitkMIDASSingleViewWidget*, double)));

  return widget;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::RequestUpdateAll()
{
  QList<int> listToUpdate = this->GetViewIndexesToUpdate(true);
  for (int i = 0; i < listToUpdate.size(); i++)
  {
    if (listToUpdate[i] >= 0 && listToUpdate[i] < m_SingleViewWidgets.size())
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
void QmitkMIDASMultiViewWidget::SetDefaultLayout(MIDASLayout layout)
{
  m_VisibilityManager->SetDefaultLayout(layout);
  if (::IsSingleWindowLayout(layout))
  {
    this->SetDefaultSingleWindowLayout(layout);
  }
  else
  {
    this->SetDefaultMultiWindowLayout(layout);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetDefaultSingleWindowLayout(MIDASLayout layout)
{
  m_SingleWindowLayout = layout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetDefaultMultiWindowLayout(MIDASLayout layout)
{
  m_MultiWindowLayout = layout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShowDropTypeWidgets(bool visible)
{
  m_ControlPanel->SetDropTypeControlsVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetViewNumberControlsVisible(bool visible)
{
  m_ControlPanel->SetViewNumberControlsVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShowMagnificationSlider(bool visible)
{
  m_ControlPanel->SetMagnificationControlsVisible(visible);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetDropType(MIDASDropType dropType)
{
  m_ControlPanel->SetDropType(dropType);

  m_VisibilityManager->ClearAllWindows();
  m_VisibilityManager->SetDropType(dropType);
  this->SetThumbnailMode(dropType == MIDAS_DROP_TYPE_ALL);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetShow2DCursors(bool visible)
{
  m_Show2DCursors = visible;

  m_ControlPanel->SetCursorVisible(visible);

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
void QmitkMIDASMultiViewWidget::SetRememberSettingsPerLayout(bool rememberSettingsPerLayout)
{
  m_RememberSettingsPerLayout = rememberSettingsPerLayout;
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetRememberSettingsPerLayout(rememberSettingsPerLayout);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetThumbnailMode(bool enabled)
{
  m_IsThumbnailMode = enabled;

  if (enabled)
  {
    m_ViewRowsInNonThumbnailMode = m_ControlPanel->GetViewRows();
    m_ViewColumnsInNonThumbnailMode = m_ControlPanel->GetViewColumns();
    m_ControlPanel->SetSliceAndMagnificationControlsEnabled(false);
    m_ControlPanel->SetLayoutControlsEnabled(false);
    m_ControlPanel->SetViewNumber(m_MaxViewRows, m_MaxViewColumns);
    m_ControlPanel->SetViewNumberControlsEnabled(false);
    m_ControlPanel->SetViewBindingControlsEnabled(false);
    this->SetViewNumber(m_MaxViewRows, m_MaxViewColumns, true);
  }
  else
  {
    m_ControlPanel->SetSliceAndMagnificationControlsEnabled(m_NavigationControllerEventListening);
    m_ControlPanel->SetLayoutControlsEnabled(m_NavigationControllerEventListening);
    m_ControlPanel->SetViewNumberControlsEnabled(true);
    m_ControlPanel->SetViewNumber(m_ViewRowsInNonThumbnailMode, m_ViewColumnsInNonThumbnailMode);
    this->SetViewNumber(m_ViewRowsInNonThumbnailMode, m_ViewColumnsInNonThumbnailMode, false);
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::GetThumbnailMode() const
{
  return m_IsThumbnailMode;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetMIDASSegmentationMode(bool enabled)
{
  m_IsMIDASSegmentationMode = enabled;

  if (enabled)
  {
    m_ViewRowsBeforeSegmentationMode = m_ControlPanel->GetViewRows();
    m_ViewColumnsBeforeSegmentationMode = m_ControlPanel->GetViewColumns();
    m_ControlPanel->SetViewNumberControlsEnabled(false);
    m_ControlPanel->SetViewBindingControlsEnabled(false);
    this->SetViewNumber(1, 1, false);
    this->SetSelectedViewIndex(0);
    this->UpdateFocusManagerToSelectedView();
  }
  else
  {
    m_ControlPanel->SetViewNumberControlsEnabled(true);
    m_ControlPanel->SetViewBindingControlsEnabled(true);
    this->SetViewNumber(m_ViewRowsBeforeSegmentationMode, m_ViewColumnsBeforeSegmentationMode, false);
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::GetMIDASSegmentationMode() const
{
  return m_IsMIDASSegmentationMode;
}


//-----------------------------------------------------------------------------
MIDASLayout QmitkMIDASMultiViewWidget::GetDefaultLayoutForSegmentation() const
{
  assert(m_VisibilityManager);

  MIDASLayout layout = m_VisibilityManager->GetDefaultLayout();

  if (   layout != MIDAS_LAYOUT_AXIAL
      && layout != MIDAS_LAYOUT_SAGITTAL
      && layout != MIDAS_LAYOUT_CORONAL
     )
  {
    layout = MIDAS_LAYOUT_CORONAL;
  }

  return layout;
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
void QmitkMIDASMultiViewWidget::SetViewNumber(int viewRows, int viewColumns, bool isThumbnailMode)
{
  // Work out required number of widgets, and hence if we need to create any new ones.
  int requiredNumberOfViews = viewRows * viewColumns;
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
      QmitkMIDASSingleViewWidget* view = this->CreateSingleViewWidget();
      view->hide();

      std::vector<QmitkRenderWindow*> renderWindows = view->GetRenderWindows();
      for (unsigned j = 0; j < renderWindows.size(); ++j)
      {
        m_ViewKeyPressStateMachine->AddRenderer(renderWindows[j]->GetRenderer());
      }
      m_SingleViewWidgets.push_back(view);
      m_VisibilityManager->RegisterWidget(view);
      m_VisibilityManager->SetAllNodeVisibilityForWindow(currentNumberOfViews + i, false);
    }
  }
  else if (requiredNumberOfViews < currentNumberOfViews)
  {
    // destroy surplus widgets
    m_VisibilityManager->DeRegisterWidgets(requiredNumberOfViews, m_SingleViewWidgets.size() - 1);

    for (int i = requiredNumberOfViews; i < m_SingleViewWidgets.size(); i++)
    {
      std::vector<QmitkRenderWindow*> renderWindows = m_SingleViewWidgets[i]->GetRenderWindows();
      for (unsigned j = 0; j < renderWindows.size(); ++j)
      {
        m_ViewKeyPressStateMachine->RemoveRenderer(renderWindows[j]->GetRenderer());
      }
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
    m_ViewRowsInNonThumbnailMode = m_ControlPanel->GetViewRows();
    m_ViewColumnsInNonThumbnailMode = m_ControlPanel->GetViewColumns();
  }
  else
  {
    // otherwise we remember the "next" (the number we are being asked for in this method call) number of rows and columns.
    m_ViewRowsInNonThumbnailMode = viewRows;
    m_ViewColumnsInNonThumbnailMode = viewColumns;
  }

  // Make all current widgets inVisible, as we are going to destroy layout.
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->hide();
  }

  // Put all widgets in the grid.
  // Prior experience suggests we always need a new grid,
  // because otherwise widgets don't appear to remove properly.

  m_TopLevelLayout->removeItem(m_LayoutForRenderWindows);
  delete m_LayoutForRenderWindows;

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("QmitkMIDASMultiViewWidget::m_LayoutForRenderWindows"));
  m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
  m_LayoutForRenderWindows->setVerticalSpacing(0);
  m_LayoutForRenderWindows->setHorizontalSpacing(0);

  m_TopLevelLayout->addLayout(m_LayoutForRenderWindows, 1, 0);

  int widgetCounter = 0;
  for (int r = 0; r < viewRows; r++)
  {
    for (int c = 0; c < viewColumns; c++)
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
  m_ControlPanel->SetViewNumber(viewRows, viewColumns);

  // Test the current m_Selected window, and reset to 0 if it now points to an invisible window.
  int selectedViewIndex = this->GetSelectedViewIndex();
  QmitkRenderWindow* selectedRenderWindow = this->GetSelectedRenderWindow();
  if (this->GetRowFromIndex(selectedViewIndex) >= viewRows || this->GetColumnFromIndex(selectedViewIndex) >= viewColumns)
  {
    selectedViewIndex = 0;
    selectedRenderWindow = m_SingleViewWidgets[selectedViewIndex]->GetSelectedRenderWindow();
  }
  this->SetSelectedRenderWindow(selectedViewIndex, selectedRenderWindow);

  // Now the number of views has changed, we need to make sure they are all in synch with all the right properties.
  this->Update2DCursorVisibility();
  this->SetShow3DWindowInOrthoView(m_Show3DWindowInOrthoView);

  // Make sure that if we are bound, we re-synch the geometry, or magnification.
  if (m_ControlPanel->AreViewGeometriesBound())
  {
    this->UpdateBoundGeometry(true);
  }
  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    this->UpdateBoundMagnification();
  }
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetRowFromIndex(int i) const
{
  if (i < 0 || i >= m_MaxViewRows * m_MaxViewColumns)
  {
    return 0;
  }
  else
  {
    return i / m_MaxViewColumns; // Note, intentionally integer division
  }
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetColumnFromIndex(int i) const
{
  if (i < 0 || i >= m_MaxViewRows * m_MaxViewColumns)
  {
    return 0;
  }
  else
  {
    return i % m_MaxViewColumns; // Note, intentionally modulus.
  }
}


//-----------------------------------------------------------------------------
int QmitkMIDASMultiViewWidget::GetIndexFromRowAndColumn(int r, int c) const
{
  return r * m_MaxViewColumns + c;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnViewNumberChanged(int rows, int columns)
{
  this->SetViewNumber(rows, columns, false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnSelectedPositionChanged(QmitkMIDASSingleViewWidget* view, QmitkRenderWindow* renderWindow, int sliceIndex)
{
  // If the view is not found, we do not do anything.
  if (std::find(m_SingleViewWidgets.begin(), m_SingleViewWidgets.end(), view) == m_SingleViewWidgets.end())
  {
    return;
  }

  std::vector<QmitkRenderWindow*> renderWindows = view->GetVisibleRenderWindows();
  if (renderWindows.size() == 1 &&
      renderWindow == renderWindows[0] &&
      sliceIndex != m_ControlPanel->GetSliceIndex())
  {
    m_ControlPanel->SetSliceIndex(sliceIndex);
  }

  mitk::Point3D selectedPosition = view->GetSelectedPosition();
  mitk::Vector3D cursorPosition = view->GetCursorPosition();

  if (m_ControlPanel->AreViewCursorsBound())
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != view)
      {
        m_SingleViewWidgets[i]->SetSelectedPosition(selectedPosition);
        m_SingleViewWidgets[i]->SetCursorPosition(cursorPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnCursorPositionChanged(QmitkMIDASSingleViewWidget* widget, const mitk::Vector3D& cursorPosition)
{
  if (m_ControlPanel->AreViewCursorsBound())
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != widget)
      {
        m_SingleViewWidgets[i]->SetCursorPosition(cursorPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnMagnificationChanged(QmitkMIDASSingleViewWidget* widget, double magnification)
{
  m_ControlPanel->SetMagnification(magnification);

  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != widget)
      {
        m_SingleViewWidgets[i]->SetMagnification(magnification);
      }
    }
  }
  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnNodesDropped(QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes)
{
  // See also QmitkMIDASMultiViewVisibilityManager::OnNodesDropped which should trigger first.
  if (m_ControlPanel->GetDropType() != MIDAS_DROP_TYPE_ALL)
  {
    m_ControlPanel->SetSliceAndMagnificationControlsEnabled(true);
    m_ControlPanel->SetLayoutControlsEnabled(true);
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

  double magnification = selectedView->GetMagnification();

  m_ControlPanel->SetMagnification(magnification);
  this->OnMagnificationChanged(magnification);

  MIDASLayout layout = selectedView->GetLayout();
  m_ControlPanel->SetLayout(layout);
  this->OnLayoutChanged(layout);

  this->Update2DCursorVisibility();
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedRenderWindow(int selectedViewIndex, QmitkRenderWindow* selectedRenderWindow)
{
  if (selectedViewIndex >= 0 && selectedViewIndex < m_SingleViewWidgets.size())
  {
    QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];

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
    MIDASLayout layout = selectedView->GetLayout();

    if (layout != MIDAS_LAYOUT_UNKNOWN)
    {
      m_ControlPanel->SetLayout(layout);
    }

    if (orientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      unsigned int maxSliceIndex = selectedView->GetMaxSliceIndex(orientation);
      unsigned int sliceIndex = selectedView->GetSliceIndex(orientation);
      m_ControlPanel->SetMaxSliceIndex(maxSliceIndex);
      m_ControlPanel->SetSliceIndex(sliceIndex);
    }

    unsigned int maxTimeStep = selectedView->GetMaxTimeStep();
    unsigned int timeStep = selectedView->GetTimeStep();
    m_ControlPanel->SetMaxTimeStep(maxTimeStep);
    m_ControlPanel->SetTimeStep(timeStep);

    double minMagnification = std::ceil(selectedView->GetMinMagnification());
    double maxMagnification = std::floor(selectedView->GetMaxMagnification());
    double magnification = selectedView->GetMagnification();
    m_ControlPanel->SetMinMagnification(minMagnification);
    m_ControlPanel->SetMaxMagnification(maxMagnification);
    m_ControlPanel->SetMagnification(magnification);

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
  mitk::BaseRenderer* renderer = focusManager->GetFocused();

  int selectedViewIndex = -1;
  vtkRenderWindow* focusedVtkRenderWindow = NULL;
  QmitkRenderWindow* focusedRenderWindow = NULL;

  if (renderer)
  {
    focusedVtkRenderWindow = renderer->GetRenderWindow();
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
  this->SetSelectedRenderWindow(selectedViewIndex, focusedRenderWindow);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnDropTypeChanged(MIDASDropType dropType)
{
  m_VisibilityManager->ClearAllWindows();
  m_VisibilityManager->SetDropType(dropType);
  this->SetThumbnailMode(dropType == MIDAS_DROP_TYPE_ALL);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnDropAccumulateChanged(bool checked)
{
  m_VisibilityManager->SetAccumulateWhenDropping(checked);
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
  return this->MoveAnteriorPosterior(1);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::MovePosterior()
{
  return this->MoveAnteriorPosterior(-1);
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::MoveAnteriorPosterior(int slices)
{
  bool actuallyDidSomething = false;
  int selectedViewIndex = this->GetSelectedViewIndex();

  if (selectedViewIndex != -1)
  {
    QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];

    MIDASOrientation orientation = selectedView->GetOrientation();
    if (orientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      unsigned int sliceIndex = selectedView->GetSliceIndex(orientation);
      int upDirection = selectedView->GetSliceUpDirection(orientation);

      int nextSliceIndex = sliceIndex + slices * upDirection;

      unsigned int maxSliceIndex = selectedView->GetMaxSliceIndex(orientation);

      if (nextSliceIndex >= 0 && nextSliceIndex <= static_cast<int>(maxSliceIndex))
      {
        this->SetSelectedWindowSliceIndex(nextSliceIndex);
        actuallyDidSomething = true;
      }
    }
  }
  return actuallyDidSomething;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnSliceIndexChanged(int sliceIndex)
{
  this->SetSelectedWindowSliceIndex(sliceIndex);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowSliceIndex(int sliceIndex)
{
  int selectedViewIndex = this->GetSelectedViewIndex();
  MIDASOrientation orientation = m_SingleViewWidgets[selectedViewIndex]->GetOrientation();

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(m_ControlPanel->AreViewGeometriesBound());
    for (int i = 0; i < viewsToUpdate.size(); i++)
    {
      m_SingleViewWidgets[viewsToUpdate[i]]->SetSliceIndex(orientation, sliceIndex);
    }
  }
  else
  {
    MITK_WARN << "Found an invalid orientation in view widget " << selectedViewIndex << ", so ignoring request to change to slice " << sliceIndex << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnMagnificationChanged(double magnification)
{
  double roundedMagnification = std::floor(magnification);

  // If we are between two integers, we raise a new event:
  if (magnification != roundedMagnification)
  {
    // If the value has decreased, we have to increase the rounded value.
    if (magnification < m_Magnification)
    {
      roundedMagnification += 1.0;
    }

    magnification = roundedMagnification;
    m_ControlPanel->SetMagnification(magnification);
  }

  this->SetSelectedWindowMagnification(magnification);
  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedWindowMagnification(double magnification)
{
  QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(m_ControlPanel->AreViewMagnificationsBound());
  for (int i = 0; i < viewsToUpdate.size(); i++)
  {
    m_SingleViewWidgets[viewsToUpdate[i]]->SetMagnification(magnification);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnTimeStepChanged(int timeStep)
{
  this->SetSelectedTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedTimeStep(int timeStep)
{
  bool dropType = m_ControlPanel->GetDropType();
  QList<int> viewsToUpdate = this->GetViewIndexesToUpdate(dropType == MIDAS_DROP_TYPE_ALL);
  for (int i = 0; i < viewsToUpdate.size(); i++)
  {
    m_SingleViewWidgets[viewsToUpdate[i]]->SetTimeStep(timeStep);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::OnLayoutChanged(MIDASLayout layout)
{
  if (layout != MIDAS_LAYOUT_UNKNOWN)
  {
    this->SetLayout(layout);

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
  std::vector<QmitkRenderWindow*> renderWindows = m_SingleViewWidgets[selectedViewIndex]->GetVisibleRenderWindows();

  if (renderWindows.size() > 0)
  {
    mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(renderWindows[0]->GetRenderer());
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchToAxial()
{
  this->SetLayout(MIDAS_LAYOUT_AXIAL);
  m_ControlPanel->SetLayout(MIDAS_LAYOUT_AXIAL);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchToSagittal()
{
  this->SetLayout(MIDAS_LAYOUT_SAGITTAL);
  m_ControlPanel->SetLayout(MIDAS_LAYOUT_SAGITTAL);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchToCoronal()
{
  this->SetLayout(MIDAS_LAYOUT_CORONAL);
  m_ControlPanel->SetLayout(MIDAS_LAYOUT_CORONAL);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::SwitchTo3D()
{
  this->SetLayout(MIDAS_LAYOUT_3D);
  m_ControlPanel->SetLayout(MIDAS_LAYOUT_3D);
  this->UpdateFocusManagerToSelectedView();
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkMIDASMultiViewWidget::ToggleMultiWindowLayout()
{
  QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[this->GetSelectedViewIndex()];
  MIDASLayout currentLayout = selectedView->GetLayout();
  MIDASLayout nextLayout;

  if (::IsSingleWindowLayout(currentLayout))
  {
    nextLayout = m_MultiWindowLayout;
  }
  else
  {
    switch (selectedView->GetOrientation())
    {
    case MIDAS_ORIENTATION_AXIAL:
      nextLayout = MIDAS_LAYOUT_AXIAL;
      break;
    case MIDAS_ORIENTATION_SAGITTAL:
      nextLayout = MIDAS_LAYOUT_SAGITTAL;
      break;
    case MIDAS_ORIENTATION_CORONAL:
      nextLayout = MIDAS_LAYOUT_CORONAL;
      break;
    case MIDAS_ORIENTATION_UNKNOWN:
      nextLayout = MIDAS_LAYOUT_3D;
      break;
    default:
      nextLayout = MIDAS_LAYOUT_CORONAL;
    }
  }

  // Note that we do not block the signals here, so this->SetLayout(nextLayout) will
  // be called.
  m_ControlPanel->SetLayout(nextLayout);
  this->OnLayoutChanged(nextLayout);

  return true;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetLayout(MIDASLayout layout)
{
  int selectedViewIndex = this->GetSelectedViewIndex();

  QList<int> viewIndexesToUpdate = this->GetViewIndexesToUpdate(m_ControlPanel->AreViewLayoutsBound());
  for (int i = 0; i < viewIndexesToUpdate.size(); i++)
  {
    int viewIndexToUpdate = viewIndexesToUpdate[i];
    QmitkMIDASSingleViewWidget* viewToUpdate = m_SingleViewWidgets[viewIndexToUpdate];
    viewToUpdate->SetLayout(layout, false);

    if (viewIndexToUpdate == selectedViewIndex)
    {
      if (layout == MIDAS_LAYOUT_AXIAL)
      {
        viewToUpdate->SetSelectedRenderWindow(viewToUpdate->GetAxialWindow());
      }
      else if (layout == MIDAS_LAYOUT_SAGITTAL)
      {
        viewToUpdate->SetSelectedRenderWindow(viewToUpdate->GetSagittalWindow());
      }
      else if (layout == MIDAS_LAYOUT_CORONAL)
      {
        viewToUpdate->SetSelectedRenderWindow(viewToUpdate->GetCoronalWindow());
      }
    }
  }

  if (::IsSingleWindowLayout(layout))
  {
    m_SingleWindowLayout = layout;
  }
  else
  {
    m_MultiWindowLayout = layout;
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
  MIDASOrientation orientation = selectedView->GetOrientation();
  int sliceIndex = selectedView->GetSliceIndex(orientation);
  int timeStep = selectedView->GetTimeStep();
  double magnification = selectedView->GetMagnification();

  QList<int> viewIndexesToUpdate = this->GetViewIndexesToUpdate(isBoundNow);
  for (int i = 0; i < viewIndexesToUpdate.size(); i++)
  {
    int viewIndexToUpdate = viewIndexesToUpdate[i];
    QmitkMIDASSingleViewWidget* viewToUpdate = m_SingleViewWidgets[viewIndexToUpdate];
    viewToUpdate->SetBoundGeometry(selectedGeometry);
    viewToUpdate->SetBoundGeometryActive(isBoundNow);
    viewToUpdate->SetMagnification(magnification);
    viewToUpdate->SetSliceIndex(orientation, sliceIndex);
    viewToUpdate->SetTimeStep(timeStep);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::UpdateBoundMagnification()
{
  int selectedViewIndex = this->GetSelectedViewIndex();
  QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];
  double magnification = selectedView->GetMagnification();
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (i != selectedViewIndex)
    {
      m_SingleViewWidgets[i]->SetMagnification(magnification);
    }
  }
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASMultiViewWidget::GetOrientation() const
{
  MIDASOrientation orientation;

  switch (m_ControlPanel->GetLayout())
  {
  case MIDAS_LAYOUT_AXIAL:
    orientation = MIDAS_ORIENTATION_AXIAL;
    break;
  case MIDAS_LAYOUT_SAGITTAL:
    orientation = MIDAS_ORIENTATION_SAGITTAL;
    break;
  case MIDAS_LAYOUT_CORONAL:
    orientation = MIDAS_ORIENTATION_CORONAL;
    break;
  default:
    orientation = MIDAS_ORIENTATION_UNKNOWN;
    break;
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
mitk::Point3D QmitkMIDASMultiViewWidget::GetSelectedPosition(const QString& id) const
{
  if (id.isNull())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    return m_SingleViewWidgets[selectedViewIndex]->GetSelectedPosition();
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
          return m_SingleViewWidgets[i]->GetSelectedPosition();
        }
      }
    }
  }
  mitk::Point3D fallBackValue;
  fallBackValue.Fill(0.0);
  return fallBackValue;
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition, const QString& id)
{
  if (id.isNull())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    m_SingleViewWidgets[selectedViewIndex]->SetSelectedPosition(selectedPosition);
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
          m_SingleViewWidgets[i]->SetSelectedPosition(selectedPosition);
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
  if (enabled && !m_NavigationControllerEventListening)
  {
    m_SingleViewWidgets[selectedViewIndex]->SetNavigationControllerEventListening(true);
  }
  else if (!enabled && m_NavigationControllerEventListening)
  {
    m_SingleViewWidgets[selectedViewIndex]->SetNavigationControllerEventListening(false);
  }
  m_NavigationControllerEventListening = enabled;
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
void QmitkMIDASMultiViewWidget::OnViewBindingTypeChanged()
{
  bool currentGeometryBound = m_SingleViewWidgets[0]->GetBoundGeometryActive();
  bool requestedGeometryBound = m_ControlPanel->AreViewGeometriesBound();

  if (currentGeometryBound != requestedGeometryBound)
  {
    this->UpdateBoundGeometry(requestedGeometryBound);
  }

  if (m_ControlPanel->AreViewLayoutsBound())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];
    MIDASLayout layout = selectedView->GetLayout();
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (i != selectedViewIndex)
      {
        m_SingleViewWidgets[i]->SetLayout(layout, false);
      }
    }
  }

  if (m_ControlPanel->AreViewCursorsBound())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];
    mitk::Point3D selectedPosition = selectedView->GetSelectedPosition();
    mitk::Vector3D cursorPosition = selectedView->GetCursorPosition();
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (i != selectedViewIndex)
      {
        m_SingleViewWidgets[i]->SetSelectedPosition(selectedPosition);
        m_SingleViewWidgets[i]->SetCursorPosition(cursorPosition);
      }
    }
  }

  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    int selectedViewIndex = this->GetSelectedViewIndex();
    QmitkMIDASSingleViewWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];
    double magnification = selectedView->GetMagnification();
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (i != selectedViewIndex)
      {
        m_SingleViewWidgets[i]->SetMagnification(magnification);
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
void QmitkMIDASMultiViewWidget::SetSliceIndexTracking(bool tracking)
{
  m_ControlPanel->SetSliceIndexTracking(tracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetTimeStepTracking(bool tracking)
{
  m_ControlPanel->SetTimeStepTracking(tracking);
}


//-----------------------------------------------------------------------------
void QmitkMIDASMultiViewWidget::SetMagnificationTracking(bool tracking)
{
  m_ControlPanel->SetMagnificationTracking(tracking);
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
  if (object == m_PinButton && event->type() == QEvent::Enter)
  {
    m_PopupWidget->showPopup();
  }
  return this->QObject::eventFilter(object, event);
}
