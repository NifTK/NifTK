/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerWidget.h"

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

#include <niftkSingleViewerWidget.h>

#include "niftkMultiViewerControls.h"

//-----------------------------------------------------------------------------
niftkMultiViewerWidget::niftkMultiViewerWidget(
    niftkMultiViewerVisibilityManager* visibilityManager,
    mitk::RenderingManager* renderingManager,
    mitk::DataStorage::Pointer dataStorage,
    int defaultViewerRows,
    int defaultViewerColumns,
    QWidget* parent, Qt::WindowFlags f)
: QWidget(parent, f)
, m_TopLevelLayout(NULL)
, m_LayoutForRenderWindows(NULL)
, m_PinButton(NULL)
, m_PopupWidget(NULL)
, m_VisibilityManager(visibilityManager)
, m_DataStorage(dataStorage)
, m_RenderingManager(renderingManager)
, m_FocusManagerObserverTag(0)
, m_SelectedViewerIndex(0)
, m_DefaultViewerRows(defaultViewerRows)
, m_DefaultViewerColumns(defaultViewerColumns)
, m_Show3DWindowIn2x2WindowLayout(false)
, m_CursorDefaultVisibility(true)
, m_RememberSettingsPerWindowLayout(false)
, m_IsThumbnailMode(false)
, m_SegmentationModeEnabled(false)
, m_LinkedNavigation(false)
, m_Magnification(0.0)
, m_SingleWindowLayout(WINDOW_LAYOUT_CORONAL)
, m_MultiWindowLayout(WINDOW_LAYOUT_ORTHO)
, m_ControlPanel(0)
{
  assert(visibilityManager);

  this->setFocusPolicy(Qt::StrongFocus);

  /************************************
   * Create stuff.
   ************************************/

  m_TopLevelLayout = new QGridLayout(this);
  m_TopLevelLayout->setObjectName(QString::fromUtf8("niftkMultiViewerWidget::m_TopLevelLayout"));
  m_TopLevelLayout->setContentsMargins(0, 0, 0, 0);
  m_TopLevelLayout->setSpacing(0);

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("niftkMultiViewerWidget::m_LayoutForRenderWindows"));
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
  m_PopupWidget->setContentsMargins(5, 5, 5, 1);
  m_PopupWidget->setLineWidth(0);

#ifdef __APPLE__
  QPalette popupPalette = this->palette();
  QColor windowColor = popupPalette.color(QPalette::Window);
  windowColor.setAlpha(64);
  popupPalette.setColor(QPalette::Window, windowColor);
  m_PopupWidget->setPalette(popupPalette);
#else
  QPalette popupPalette = this->palette();
  QColor windowColor = popupPalette.color(QPalette::Window);
  windowColor.setAlpha(128);
  popupPalette.setColor(QPalette::Window, windowColor);
  m_PopupWidget->setPalette(popupPalette);
  m_PopupWidget->setAttribute(Qt::WA_TranslucentBackground, true);
#endif

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
  pinButtonIcon.addFile(":Icons/PushPinIn.png", QSize(), QIcon::Normal, QIcon::On);
  pinButtonIcon.addFile(":Icons/PushPinOut.png", QSize(), QIcon::Normal, QIcon::Off);
  m_PinButton->setIcon(pinButtonIcon);

  this->connect(m_PinButton, SIGNAL(toggled(bool)), SLOT(OnPinButtonToggled(bool)));
  m_PinButton->installEventFilter(this);

  m_ControlPanel = this->CreateControlPanel(m_PopupWidget);

  pinButtonWidgetLayout->addWidget(m_PinButton);

  m_TopLevelLayout->addWidget(pinButtonWidget, 0, 0);
  m_TopLevelLayout->setRowMinimumHeight(0, buttonRowHeight);
  m_TopLevelLayout->addLayout(m_LayoutForRenderWindows, 1, 0);

  /************************************
   * Now initialise stuff.
   ************************************/

  m_ControlPanel->SetDirectionAnnotationsVisible(true);

  // Default to dropping into single window.
  m_ControlPanel->SetDropType(DNDDISPLAY_DROP_SINGLE);
  m_VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);

  // We have the default rows and columns passed in via constructor args, in initialise list.
  m_ControlPanel->SetViewerNumber(m_DefaultViewerRows, m_DefaultViewerColumns);
  this->SetViewerNumber(m_DefaultViewerRows, m_DefaultViewerColumns, false);

  // Connect Qt Signals to make it all hang together.
  this->connect(m_ControlPanel, SIGNAL(SelectedSliceChanged(int)), SLOT(OnSelectedSliceChanged(int)));
  this->connect(m_ControlPanel, SIGNAL(TimeStepChanged(int)), SLOT(OnTimeStepChanged(int)));
  this->connect(m_ControlPanel, SIGNAL(MagnificationChanged(double)), SLOT(OnMagnificationChanged(double)));

  this->connect(m_ControlPanel, SIGNAL(ShowCursorChanged(bool)), SLOT(OnCursorVisibilityChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ShowDirectionAnnotationsChanged(bool)), SLOT(OnShowDirectionAnnotationsChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(Show3DWindowChanged(bool)), SLOT(OnShow3DWindowChanged(bool)));

  this->connect(m_ControlPanel, SIGNAL(WindowLayoutChanged(WindowLayout)), SLOT(OnWindowLayoutChanged(WindowLayout)));
  this->connect(m_ControlPanel, SIGNAL(WindowCursorBindingChanged(bool)), SLOT(OnWindowCursorBindingChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(WindowMagnificationBindingChanged(bool)), SLOT(OnWindowMagnificationBindingChanged(bool)));

  this->connect(m_ControlPanel, SIGNAL(ViewerNumberChanged(int, int)), SLOT(OnViewerNumberChanged(int, int)));

  this->connect(m_ControlPanel, SIGNAL(ViewerPositionBindingChanged(bool)), SLOT(OnViewerPositionBindingChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerCursorBindingChanged(bool)), SLOT(OnViewerCursorBindingChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerMagnificationBindingChanged(bool)), SLOT(OnViewerMagnificationBindingChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerWindowLayoutBindingChanged(bool)), SLOT(OnViewerWindowLayoutBindingChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerGeometryBindingChanged(bool)), SLOT(OnViewerGeometryBindingChanged(bool)));

  this->connect(m_ControlPanel, SIGNAL(DropTypeChanged(DnDDisplayDropType)), SLOT(OnDropTypeChanged(DnDDisplayDropType)));
  this->connect(m_ControlPanel, SIGNAL(DropAccumulateChanged(bool)), SLOT(OnDropAccumulateChanged(bool)));

  this->connect(m_PopupWidget, SIGNAL(popupOpened(bool)), SLOT(OnPopupOpened(bool)));

  // We listen to FocusManager to detect when things have changed focus, and hence to highlight the "current window".
  itk::SimpleMemberCommand<niftkMultiViewerWidget>::Pointer onFocusChangedCommand =
    itk::SimpleMemberCommand<niftkMultiViewerWidget>::New();
  onFocusChangedCommand->SetCallbackFunction( this, &niftkMultiViewerWidget::OnFocusChanged );

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
}


//-----------------------------------------------------------------------------
niftkMultiViewerControls* niftkMultiViewerWidget::CreateControlPanel(QWidget* parent)
{
  niftkMultiViewerControls* controlPanel = new niftkMultiViewerControls(parent);

  controlPanel->SetMaxViewerNumber(m_MaxViewerRows, m_MaxViewerColumns);

  controlPanel->SetWindowCursorsBound(true);
  controlPanel->SetWindowMagnificationsBound(true);

  return controlPanel;
}


//-----------------------------------------------------------------------------
niftkMultiViewerWidget::~niftkMultiViewerWidget()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }
  this->EnableLinkedNavigation(false);
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidget* niftkMultiViewerWidget::CreateViewer()
{
  niftkSingleViewerWidget* viewer = new niftkSingleViewerWidget(this, m_RenderingManager);
  viewer->SetDataStorage(m_DataStorage);
  viewer->setObjectName(tr("niftkSingleViewerWidget"));
  viewer->setVisible(false);

  viewer->SetBackgroundColour(m_BackgroundColour);
  viewer->SetShow3DWindowIn2x2WindowLayout(m_Show3DWindowIn2x2WindowLayout);
  viewer->SetRememberSettingsPerWindowLayout(m_RememberSettingsPerWindowLayout);
  viewer->SetDisplayInteractionsEnabled(true);
  viewer->SetCursorPositionBinding(true);
  viewer->SetScaleFactorBinding(true);
  viewer->SetDefaultSingleWindowLayout(m_SingleWindowLayout);
  viewer->SetDefaultMultiWindowLayout(m_MultiWindowLayout);

  m_VisibilityManager->connect(viewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(viewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(viewer, SIGNAL(SelectedPositionChanged(niftkSingleViewerWidget*, const mitk::Point3D&)), SLOT(OnSelectedPositionChanged(niftkSingleViewerWidget*, const mitk::Point3D&)));
  this->connect(viewer, SIGNAL(SelectedTimeStepChanged(niftkSingleViewerWidget*, int)), SLOT(OnSelectedTimeStepChanged(niftkSingleViewerWidget*, int)));
  this->connect(viewer, SIGNAL(CursorPositionChanged(niftkSingleViewerWidget*, MIDASOrientation, const mitk::Vector2D&)), SLOT(OnCursorPositionChanged(niftkSingleViewerWidget*, MIDASOrientation, const mitk::Vector2D&)));
  this->connect(viewer, SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, MIDASOrientation, double)), SLOT(OnScaleFactorChanged(niftkSingleViewerWidget*, MIDASOrientation, double)));
  this->connect(viewer, SIGNAL(WindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout)), SLOT(OnWindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout)));
  this->connect(viewer, SIGNAL(CursorPositionBindingChanged(niftkSingleViewerWidget*, bool)), SLOT(OnCursorPositionBindingChanged(niftkSingleViewerWidget*, bool)));
  this->connect(viewer, SIGNAL(ScaleFactorBindingChanged(niftkSingleViewerWidget*, bool)), SLOT(OnScaleFactorBindingChanged(niftkSingleViewerWidget*, bool)));
  this->connect(viewer, SIGNAL(CursorVisibilityChanged(niftkSingleViewerWidget*, bool)), SLOT(OnCursorVisibilityChanged(niftkSingleViewerWidget*, bool)));

  return viewer;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::RequestUpdateAll()
{
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    if (viewer->isVisible())
    {
      viewer->RequestUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetInterpolationType(DnDDisplayInterpolationType interpolationType)
{
  m_VisibilityManager->SetInterpolationType(interpolationType);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetDefaultWindowLayout(WindowLayout windowLayout)
{
  m_VisibilityManager->SetDefaultWindowLayout(windowLayout);
  if (::IsSingleWindowLayout(windowLayout))
  {
    this->SetDefaultSingleWindowLayout(windowLayout);
  }
  else
  {
    this->SetDefaultMultiWindowLayout(windowLayout);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetDefaultSingleWindowLayout(WindowLayout windowLayout)
{
  m_SingleWindowLayout = windowLayout;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetDefaultMultiWindowLayout(WindowLayout windowLayout)
{
  m_MultiWindowLayout = windowLayout;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetShowMagnificationSlider(bool visible)
{
  m_ControlPanel->SetMagnificationControlsVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::AreShowOptionsVisible() const
{
  return m_ControlPanel->AreShowOptionsVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetShowOptionsVisible(bool visible)
{
  m_ControlPanel->SetShowOptionsVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::AreWindowLayoutControlsVisible() const
{
  return m_ControlPanel->AreWindowLayoutControlsVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetWindowLayoutControlsVisible(bool visible)
{
  m_ControlPanel->SetWindowLayoutControlsVisible(visible);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::AreViewerNumberControlsVisible() const
{
  return m_ControlPanel->AreViewerNumberControlsVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetViewerNumberControlsVisible(bool visible)
{
  m_ControlPanel->SetViewerNumberControlsVisible(visible);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetShowDropTypeControls(bool visible)
{
  m_ControlPanel->SetDropTypeControlsVisible(visible);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetDropType(DnDDisplayDropType dropType)
{
  if (dropType != m_ControlPanel->GetDropType())
  {
    m_ControlPanel->SetDropType(dropType);

    m_VisibilityManager->ClearAllViewers();
    m_VisibilityManager->SetDropType(dropType);
    this->SetThumbnailMode(dropType == DNDDISPLAY_DROP_ALL);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::IsCursorVisible() const
{
  return m_ControlPanel->IsCursorVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetCursorVisible(bool visible)
{
  m_ControlPanel->SetCursorVisible(visible);

  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  selectedViewer->SetCursorVisible(visible);

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        otherViewer->SetCursorVisible(visible);
      }
    }
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::GetCursorDefaultVisibility() const
{
  return m_CursorDefaultVisibility;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetCursorDefaultVisibility(bool visible)
{
  m_CursorDefaultVisibility = visible;
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::AreDirectionAnnotationsVisible() const
{
  return m_ControlPanel->AreDirectionAnnotationsVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetDirectionAnnotationsVisible(bool visible)
{
  m_ControlPanel->SetDirectionAnnotationsVisible(visible);
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetDirectionAnnotationsVisible(visible);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::GetShow3DWindowIn2x2WindowLayout() const
{
  return m_Show3DWindowIn2x2WindowLayout;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetShow3DWindowIn2x2WindowLayout(bool visible)
{
  m_Show3DWindowIn2x2WindowLayout = visible;
  m_ControlPanel->Set3DWindowVisible(visible);
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetShow3DWindowIn2x2WindowLayout(visible);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetRememberSettingsPerWindowLayout(bool rememberSettingsPerWindowLayout)
{
  m_RememberSettingsPerWindowLayout = rememberSettingsPerWindowLayout;
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerWindowLayout);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetThumbnailMode(bool enabled)
{
  m_IsThumbnailMode = enabled;

  if (enabled)
  {
    m_ViewerRowsInNonThumbnailMode = m_ControlPanel->GetViewerRows();
    m_ViewerColumnsInNonThumbnailMode = m_ControlPanel->GetViewerColumns();
    m_ControlPanel->SetSingleViewerControlsEnabled(false);
    m_ControlPanel->SetViewerNumber(m_MaxViewerRows, m_MaxViewerColumns);
    m_ControlPanel->SetMultiViewerControlsEnabled(false);
    this->SetViewerNumber(m_MaxViewerRows, m_MaxViewerColumns, true);
  }
  else
  {
    m_ControlPanel->SetSingleViewerControlsEnabled(m_LinkedNavigation);
    m_ControlPanel->SetMultiViewerControlsEnabled(true);
    m_ControlPanel->SetViewerNumber(m_ViewerRowsInNonThumbnailMode, m_ViewerColumnsInNonThumbnailMode);
    this->SetViewerNumber(m_ViewerRowsInNonThumbnailMode, m_ViewerColumnsInNonThumbnailMode, false);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::GetThumbnailMode() const
{
  return m_IsThumbnailMode;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSegmentationModeEnabled(bool enabled)
{
  m_SegmentationModeEnabled = enabled;

  if (enabled)
  {
    m_ViewerRowsBeforeSegmentationMode = m_ControlPanel->GetViewerRows();
    m_ViewerColumnsBeforeSegmentationMode = m_ControlPanel->GetViewerColumns();
    m_ControlPanel->SetMultiViewerControlsEnabled(false);
    this->SetViewerNumber(1, 1, false);
    this->SetSelectedViewerByIndex(0);
    this->UpdateFocusManagerToSelectedViewer();
  }
  else
  {
    m_ControlPanel->SetMultiViewerControlsEnabled(true);
    this->SetViewerNumber(m_ViewerRowsBeforeSegmentationMode, m_ViewerColumnsBeforeSegmentationMode, false);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::IsSegmentationModeEnabled() const
{
  return m_SegmentationModeEnabled;
}


//-----------------------------------------------------------------------------
WindowLayout niftkMultiViewerWidget::GetDefaultWindowLayoutForSegmentation() const
{
  assert(m_VisibilityManager);

  WindowLayout windowLayout = m_VisibilityManager->GetDefaultWindowLayout();

  if (   windowLayout != WINDOW_LAYOUT_AXIAL
      && windowLayout != WINDOW_LAYOUT_SAGITTAL
      && windowLayout != WINDOW_LAYOUT_CORONAL
     )
  {
    windowLayout = WINDOW_LAYOUT_CORONAL;
  }

  return windowLayout;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetBackgroundColour(QColor backgroundColour)
{
  m_BackgroundColour = backgroundColour;

  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetBackgroundColour(m_BackgroundColour);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetViewerNumber(int viewerRows, int viewerColumns, bool isThumbnailMode)
{
  // Work out required number of viewers, and hence if we need to create any new ones.
  int requiredNumberOfViewers = viewerRows * viewerColumns;
  int currentNumberOfViewers = m_Viewers.size();

  // If we have the right number of viewers, there is nothing to do, so early exit.
  if (viewerRows == m_MaxViewerRows && viewerColumns == m_MaxViewerColumns)
  {
    return;
  }

  int numberOfSurvivingViewers = std::min(currentNumberOfViewers, requiredNumberOfViewers);
  std::vector<mitk::Point3D> selectedPositionInSurvivingViewers(numberOfSurvivingViewers);
  std::vector<std::vector<mitk::Vector2D> > cursorPositionsInSurvivingViewers(numberOfSurvivingViewers);
  std::vector<std::vector<double> > scaleFactorsInSurvivingViewers(numberOfSurvivingViewers);

  for (int i = 0; i < numberOfSurvivingViewers; ++i)
  {
    selectedPositionInSurvivingViewers[i] = m_Viewers[i]->GetSelectedPosition();
    cursorPositionsInSurvivingViewers[i] = m_Viewers[i]->GetCursorPositions();
    scaleFactorsInSurvivingViewers[i] = m_Viewers[i]->GetScaleFactors();
  }

  // Make all current viewers inVisible, as we are going to destroy layout.
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->hide();
  }

  /////////////////////////////////////////
  // Start: Rebuild the number of viewers.
  // NOTE:  The order of viewers in
  //        m_Viewers and
  //        m_VisibilityManager must match.
  /////////////////////////////////////////

  if (requiredNumberOfViewers > currentNumberOfViewers)
  {
    // create some more viewers
    int additionalViewers = requiredNumberOfViewers - m_Viewers.size();
    for (int i = 0; i < additionalViewers; i++)
    {
      niftkSingleViewerWidget* viewer = this->CreateViewer();
      viewer->hide();

      m_Viewers.push_back(viewer);
      m_VisibilityManager->RegisterViewer(viewer);
      m_VisibilityManager->SetAllNodeVisibilityForViewer(currentNumberOfViewers + i, false);
    }
  }
  else if (requiredNumberOfViewers < currentNumberOfViewers)
  {
    // destroy surplus viewers
    m_VisibilityManager->DeRegisterViewers(requiredNumberOfViewers, m_Viewers.size() - 1);

    for (int i = requiredNumberOfViewers; i < m_Viewers.size(); i++)
    {
      delete m_Viewers[i];
    }

    m_Viewers.erase(m_Viewers.begin() + requiredNumberOfViewers, m_Viewers.end());
  }

  // We need to remember the "previous" number of rows and columns, so when we switch out
  // of thumbnail mode, we know how many rows and columns to revert to.
  if (isThumbnailMode)
  {
    m_ViewerRowsInNonThumbnailMode = m_ControlPanel->GetViewerRows();
    m_ViewerColumnsInNonThumbnailMode = m_ControlPanel->GetViewerColumns();
  }
  else
  {
    // otherwise we remember the "next" (the number we are being asked for in this method call) number of rows and columns.
    m_ViewerRowsInNonThumbnailMode = viewerRows;
    m_ViewerColumnsInNonThumbnailMode = viewerColumns;
  }

  // Make all current viewers inVisible, as we are going to destroy layout.
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->hide();
  }

  // Put all viewers in the grid.
  // Prior experience suggests we always need a new grid,
  // because otherwise viewers don't appear to remove properly.

  m_TopLevelLayout->removeItem(m_LayoutForRenderWindows);
  delete m_LayoutForRenderWindows;

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("niftkMultiViewerWidget::m_LayoutForRenderWindows"));
  m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
  m_LayoutForRenderWindows->setVerticalSpacing(0);
  m_LayoutForRenderWindows->setHorizontalSpacing(0);

  m_TopLevelLayout->addLayout(m_LayoutForRenderWindows, 1, 0);

  int viewerIndex = 0;
  for (int row = 0; row < viewerRows; row++)
  {
    for (int column = 0; column < viewerColumns; column++)
    {
      m_LayoutForRenderWindows->addWidget(m_Viewers[viewerIndex], row, column);
      m_Viewers[viewerIndex]->show();
      m_Viewers[viewerIndex]->setEnabled(true);
      viewerIndex++;
    }
  }

  for (int i = 0; i < numberOfSurvivingViewers; ++i)
  {
    bool signalsWereBlocked = m_Viewers[i]->blockSignals(true);
    m_Viewers[i]->SetSelectedPosition(selectedPositionInSurvivingViewers[i]);
    m_Viewers[i]->SetCursorPositions(cursorPositionsInSurvivingViewers[i]);
    m_Viewers[i]->SetScaleFactors(scaleFactorsInSurvivingViewers[i]);
    m_Viewers[i]->blockSignals(signalsWereBlocked);
  }

  ////////////////////////////////////////
  // End: Rebuild the number of viewers.
  ////////////////////////////////////////

  // Update row/column of viewers without triggering another layout size change.
  m_ControlPanel->SetViewerNumber(viewerRows, viewerColumns);

  // Test the current m_Selected window, and reset to 0 if it now points to an invisible window.
  int selectedViewerIndex = this->GetSelectedViewerIndex();
  niftkSingleViewerWidget* selectedViewer;
  QmitkRenderWindow* selectedRenderWindow = this->GetSelectedRenderWindow();
  if (this->GetViewerRowFromIndex(selectedViewerIndex) >= viewerRows || this->GetViewerColumnFromIndex(selectedViewerIndex) >= viewerColumns)
  {
    selectedViewerIndex = 0;
    selectedViewer = m_Viewers[selectedViewerIndex];
    selectedRenderWindow = selectedViewer->GetSelectedRenderWindow();
  }
  else
  {
    selectedViewer = m_Viewers[selectedViewerIndex];
  }
  this->SetSelectedRenderWindow(selectedViewerIndex, selectedRenderWindow);

  // Now the number of viewers has changed, we need to make sure they are all in synch with all the right properties.
  this->OnCursorVisibilityChanged(selectedViewer, selectedViewer->IsCursorVisible());
  this->SetShow3DWindowIn2x2WindowLayout(m_Show3DWindowIn2x2WindowLayout);

  if (m_ControlPanel->AreViewerGeometriesBound())
  {
    niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    mitk::TimeGeometry* geometry = selectedViewer->GetGeometry();

    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetBoundGeometry(geometry);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    MIDASOrientation orientation = selectedViewer->GetOrientation();
    double magnification = selectedViewer->GetMagnification(orientation);
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetMagnification(orientation, magnification);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidget::GetViewerRowFromIndex(int i) const
{
  if (i < 0 || i >= m_MaxViewerRows * m_MaxViewerColumns)
  {
    return 0;
  }
  else
  {
    return i / m_MaxViewerColumns; // Note, intentionally integer division
  }
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidget::GetViewerColumnFromIndex(int i) const
{
  if (i < 0 || i >= m_MaxViewerRows * m_MaxViewerColumns)
  {
    return 0;
  }
  else
  {
    return i % m_MaxViewerColumns; // Note, intentionally modulus.
  }
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidget::GetViewerIndexFromRowAndColumn(int r, int c) const
{
  return r * m_MaxViewerColumns + c;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerNumberChanged(int rows, int columns)
{
  this->SetViewerNumber(rows, columns, false);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnSelectedPositionChanged(niftkSingleViewerWidget* viewer, const mitk::Point3D& selectedPosition)
{
  // If the viewer is not found, we do not do anything.
  if (std::find(m_Viewers.begin(), m_Viewers.end(), viewer) == m_Viewers.end())
  {
    return;
  }

  MIDASOrientation orientation = viewer->GetOrientation();
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    m_ControlPanel->SetSelectedSlice(viewer->GetSelectedSlice(orientation));
  }

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetSelectedPosition(selectedPosition);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnSelectedTimeStepChanged(niftkSingleViewerWidget* viewer, int timeStep)
{
  // If the viewer is not found, we do not do anything.
  if (std::find(m_Viewers.begin(), m_Viewers.end(), viewer) == m_Viewers.end())
  {
    return;
  }

  m_ControlPanel->SetTimeStep(timeStep);

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetTimeStep(timeStep);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnCursorPositionChanged(niftkSingleViewerWidget* viewer, MIDASOrientation orientation, const mitk::Vector2D& cursorPosition)
{
  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetCursorPosition(orientation, cursorPosition);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnScaleFactorChanged(niftkSingleViewerWidget* viewer, MIDASOrientation orientation, double scaleFactor)
{
  double magnification = viewer->GetMagnification(orientation);
  m_ControlPanel->SetMagnification(magnification);

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetScaleFactor(orientation, scaleFactor);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnCursorPositionBindingChanged(niftkSingleViewerWidget* viewer, bool bound)
{
  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetCursorPositionBinding(bound);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnScaleFactorBindingChanged(niftkSingleViewerWidget* viewer, bool bound)
{
  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetScaleFactorBinding(bound);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnNodesDropped(niftkSingleViewerWidget* dropOntoViewer, QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes)
{
  // See also niftkMultiViewerVisibilityManager::OnNodesDropped which should trigger first.
  if (m_ControlPanel->GetDropType() != DNDDISPLAY_DROP_ALL)
  {
    m_ControlPanel->SetSingleViewerControlsEnabled(true);
  }

  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  // This does not trigger OnFocusChanged() the very first time, as when creating the editor, the first viewer already has focus.
  mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(renderWindow->GetRenderer());

  if (m_ControlPanel->AreViewerWindowLayoutsBound())
  {
    bool signalsWereBlocked = dropOntoViewer->blockSignals(true);
    dropOntoViewer->SetWindowLayout(selectedViewer->GetWindowLayout());
    dropOntoViewer->blockSignals(signalsWereBlocked);
  }

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    const mitk::Point3D& selectedPosition = selectedViewer->GetSelectedPosition();
    bool signalsWereBlocked = dropOntoViewer->blockSignals(true);
    dropOntoViewer->SetSelectedPosition(selectedPosition);
    dropOntoViewer->blockSignals(signalsWereBlocked);
  }

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    const std::vector<mitk::Vector2D>& cursorPositions = selectedViewer->GetCursorPositions();
    bool signalsWereBlocked = dropOntoViewer->blockSignals(true);
    dropOntoViewer->SetCursorPositions(cursorPositions);
    dropOntoViewer->blockSignals(signalsWereBlocked);
  }

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    double scaleFactor = selectedViewer->GetScaleFactor(selectedViewer->GetOrientation());
    bool signalsWereBlocked = dropOntoViewer->blockSignals(true);
    dropOntoViewer->SetScaleFactor(dropOntoViewer->GetOrientation(), scaleFactor);
    dropOntoViewer->blockSignals(signalsWereBlocked);
  }

//  m_ControlPanel->SetMagnification(magnification);
//  m_ControlPanel->SetMagnification(scaleFactor);
//  this->OnMagnificationChanged(magnification);
//  this->OnMagnificationChanged(scaleFactor);
//  this->SetMagnification(magnification);

  bool cursorVisibility;
  if (m_ControlPanel->AreViewerCursorsBound())
  {
    cursorVisibility = m_ControlPanel->IsCursorVisible();
  }
  else
  {
    cursorVisibility = m_CursorDefaultVisibility;
    m_ControlPanel->SetCursorVisible(cursorVisibility);
  }
  dropOntoViewer->SetCursorVisible(cursorVisibility);

  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSelectedRenderWindow(int selectedViewerIndex, QmitkRenderWindow* selectedRenderWindow)
{
  if (selectedViewerIndex >= 0 && selectedViewerIndex < m_Viewers.size())
  {
    niftkSingleViewerWidget* selectedViewer = m_Viewers[selectedViewerIndex];

    // This, to turn off borders on all other windows.
    this->SetSelectedViewerByIndex(selectedViewerIndex);

    // This to specifically set the border round one sub-pane for if its in 2x2 window layout.
    if (selectedRenderWindow != NULL)
    {
      int numberOfNodes = m_VisibilityManager->GetNodesInViewer(selectedViewerIndex);
      if (numberOfNodes > 0)
      {
        bool signalsWereBlocked = selectedViewer->blockSignals(true);
        selectedViewer->SetSelectedRenderWindow(selectedRenderWindow);
        selectedViewer->blockSignals(signalsWereBlocked);
      }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // Need to enable widgets appropriately, so user can't press stuff that they aren't meant to.
    /////////////////////////////////////////////////////////////////////////////////////////////
    MIDASOrientation orientation = selectedViewer->GetOrientation();
    WindowLayout windowLayout = selectedViewer->GetWindowLayout();

    if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      m_ControlPanel->SetWindowLayout(windowLayout);
    }

    if (orientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      int maxSlice = selectedViewer->GetMaxSlice(orientation);
      int selectedSlice = selectedViewer->GetSelectedSlice(orientation);
      m_ControlPanel->SetMaxSlice(maxSlice);
      m_ControlPanel->SetSelectedSlice(selectedSlice);
    }

    int maxTimeStep = selectedViewer->GetMaxTimeStep();
    int timeStep = selectedViewer->GetTimeStep();
    m_ControlPanel->SetMaxTimeStep(maxTimeStep);
    m_ControlPanel->SetTimeStep(timeStep);

    if (orientation != MIDAS_ORIENTATION_UNKNOWN)
    {
      m_ControlPanel->SetMagnificationControlsEnabled(true);
      double minMagnification = std::ceil(selectedViewer->GetMinMagnification());
      double maxMagnification = std::floor(selectedViewer->GetMaxMagnification());
      double magnification = selectedViewer->GetMagnification(orientation);
      m_ControlPanel->SetMinMagnification(minMagnification);
      m_ControlPanel->SetMaxMagnification(maxMagnification);
      m_ControlPanel->SetMagnification(magnification);
    }
    else
    {
      m_ControlPanel->SetMagnificationControlsEnabled(false);
    }

    this->OnCursorVisibilityChanged(selectedViewer, selectedViewer->IsCursorVisible());
  }
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetFocus()
{
  this->GetSelectedViewer()->SetFocus();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  int selectedViewerIndex = -1;
  QmitkRenderWindow* focusedRenderWindow = NULL;

  if (focusedRenderer)
  {
    for (int i = 0; i < m_Viewers.size(); i++)
    {
      const std::vector<QmitkRenderWindow*>& viewerRenderWindows = m_Viewers[i]->GetRenderWindows();
      for (int j = 0; j < viewerRenderWindows.size(); ++j)
      {
        if (focusedRenderer == viewerRenderWindows[j]->GetRenderer())
        {
          selectedViewerIndex = i;
          focusedRenderWindow = viewerRenderWindows[j];
          break;
        }
      }
      if (focusedRenderWindow)
      {
        break;
      }
    }
  }

  this->SetSelectedRenderWindow(selectedViewerIndex, focusedRenderWindow);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnDropTypeChanged(DnDDisplayDropType dropType)
{
  m_VisibilityManager->ClearAllViewers();
  m_VisibilityManager->SetDropType(dropType);
  this->SetThumbnailMode(dropType == DNDDISPLAY_DROP_ALL);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnDropAccumulateChanged(bool checked)
{
  m_VisibilityManager->SetAccumulateWhenDropping(checked);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnSelectedSliceChanged(int selectedSlice)
{
  MIDASOrientation orientation = this->GetSelectedViewer()->GetOrientation();

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    bool signalsWereBlocked = selectedViewer->blockSignals(true);
    selectedViewer->SetSelectedSlice(orientation, selectedSlice);
    selectedViewer->blockSignals(signalsWereBlocked);

    if (m_ControlPanel->AreViewerPositionsBound())
    {
      foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
      {
        if (otherViewer != selectedViewer && otherViewer->isVisible())
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetSelectedSlice(orientation, selectedSlice);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }
  }
  else
  {
    MITK_WARN << "Found an invalid orientation in viewer " << this->GetSelectedViewerIndex() << ", so ignoring request to change to slice " << selectedSlice << std::endl;
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnMagnificationChanged(double magnification)
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

  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  MIDASOrientation orientation = selectedViewer->GetOrientation();

  bool signalsWereBlocked = selectedViewer->blockSignals(true);
  selectedViewer->SetMagnification(orientation, magnification);
  selectedViewer->blockSignals(signalsWereBlocked);

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetMagnification(orientation, magnification);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }

  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnTimeStepChanged(int timeStep)
{
  this->SetSelectedTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSelectedTimeStep(int timeStep)
{
  DnDDisplayDropType dropType = m_ControlPanel->GetDropType();

  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  bool signalsWereBlocked = selectedViewer->blockSignals(true);
  selectedViewer->SetTimeStep(timeStep);
  selectedViewer->blockSignals(signalsWereBlocked);

  if (dropType == DNDDISPLAY_DROP_ALL)
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        signalsWereBlocked = selectedViewer->blockSignals(true);
        otherViewer->SetTimeStep(timeStep);
        selectedViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowLayoutChanged(WindowLayout windowLayout)
{
  if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    this->SetWindowLayout(windowLayout);

    // Update the focus to the selected window, to trigger things like thumbnail viewer refresh
    // (or indeed anything that's listening to the FocusManager).
    this->UpdateFocusManagerToSelectedViewer();
  }

}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowLayoutChanged(niftkSingleViewerWidget* selectedViewer, WindowLayout windowLayout)
{
  m_ControlPanel->SetWindowLayout(windowLayout);
  m_ControlPanel->SetWindowCursorsBound(selectedViewer->GetCursorPositionBinding());
  m_ControlPanel->SetWindowMagnificationsBound(selectedViewer->GetScaleFactorBinding());
  this->UpdateFocusManagerToSelectedViewer();

  if (m_ControlPanel->AreViewerWindowLayoutsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetWindowLayout(windowLayout, m_ControlPanel->AreViewerCursorsBound(), m_ControlPanel->AreViewerMagnificationsBound());
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    const mitk::Point3D& selectedPosition = selectedViewer->GetSelectedPosition();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetSelectedPosition(selectedPosition);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    const std::vector<mitk::Vector2D>& cursorPositions = selectedViewer->GetCursorPositions();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetCursorPositions(cursorPositions);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    std::vector<double> scaleFactors = selectedViewer->GetScaleFactors();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetScaleFactors(scaleFactors);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnCursorVisibilityChanged(niftkSingleViewerWidget* viewer, bool visible)
{
  m_ControlPanel->SetCursorVisible(visible);

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetCursorVisible(visible);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnGeometryChanged(niftkSingleViewerWidget* /*selectedViewer*/, mitk::TimeGeometry* geometry)
{
  if (m_ControlPanel->AreViewerGeometriesBound())
  {
    foreach (niftkSingleViewerWidget* viewer, m_Viewers)
    {
      bool signalsWereBlocked = viewer->blockSignals(true);
      viewer->SetBoundGeometry(geometry);
      viewer->blockSignals(signalsWereBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowCursorBindingChanged(bool bound)
{
  /// If the cursor positions are bound across the viewers then the binding property
  /// across the windows of the viewers can be controlled just together. That is, it
  /// is either set for each or none of them.
  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (niftkSingleViewerWidget* viewer, m_Viewers)
    {
      if (viewer->isVisible())
      {
        bool signalsWereBlocked = viewer->blockSignals(true);
        viewer->SetCursorPositionBinding(bound);
        viewer->blockSignals(signalsWereBlocked);
      }
    }
  }
  else
  {
    this->GetSelectedViewer()->SetCursorPositionBinding(bound);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowMagnificationBindingChanged(bool bound)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  bool signalsWereBlocked = selectedViewer->blockSignals(true);
  selectedViewer->SetScaleFactorBinding(bound);
  selectedViewer->blockSignals(signalsWereBlocked);

  /// If the scale factors are bound across the viewers then the binding property
  /// across the windows of the viewers can be controlled just together. That is, it
  /// is either set for each or none of them.
  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      const std::vector<double>& scaleFactors = selectedViewer->GetScaleFactors();
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetScaleFactorBinding(bound);
        otherViewer->SetScaleFactors(scaleFactors);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnCursorVisibilityChanged(bool visible)
{
  this->SetCursorVisible(visible);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnShowDirectionAnnotationsChanged(bool visible)
{
  this->SetDirectionAnnotationsVisible(visible);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnShow3DWindowChanged(bool visible)
{
  this->SetShow3DWindowIn2x2WindowLayout(visible);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::UpdateFocusManagerToSelectedViewer()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  if (QmitkRenderWindow* selectedRenderWindow = this->GetSelectedRenderWindow())
  {
    mitk::BaseRenderer* selectedRenderer = selectedRenderWindow->GetRenderer();
    if (selectedRenderer != focusedRenderer)
    {
      focusManager->SetFocused(selectedRenderer);
    }
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::ToggleCursorVisibility()
{
  this->SetCursorVisible(!this->IsCursorVisible());

  return true;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetWindowLayout(WindowLayout windowLayout)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  bool signalsWereBlocked = selectedViewer->blockSignals(true);
  selectedViewer->SetWindowLayout(windowLayout);
  selectedViewer->blockSignals(signalsWereBlocked);

  m_ControlPanel->SetWindowLayout(windowLayout);
  m_ControlPanel->SetWindowCursorsBound(selectedViewer->GetCursorPositionBinding());
  m_ControlPanel->SetWindowMagnificationsBound(selectedViewer->GetScaleFactorBinding());

  if (m_ControlPanel->AreViewerWindowLayoutsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetWindowLayout(windowLayout);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }

  MIDASOrientation orientation = selectedViewer->GetOrientation();

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    mitk::Vector2D cursorPosition = selectedViewer->GetCursorPosition(orientation);
    const std::vector<mitk::Vector2D>& cursorPositions = selectedViewer->GetCursorPositions();

    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        if (m_ControlPanel->AreWindowCursorsBound())
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetCursorPositions(cursorPositions);
          otherViewer->blockSignals(signalsWereBlocked);
        }
        else
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetCursorPosition(orientation, cursorPosition);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }
  }

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    double scaleFactor = selectedViewer->GetScaleFactor(orientation);
    const std::vector<double>& scaleFactors = selectedViewer->GetScaleFactors();

    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        if (m_ControlPanel->AreWindowMagnificationsBound())
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetScaleFactors(scaleFactors);
          otherViewer->blockSignals(signalsWereBlocked);
        }
        else
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetScaleFactor(orientation, scaleFactor);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }
  }

  if (::IsSingleWindowLayout(windowLayout))
  {
    m_SingleWindowLayout = windowLayout;
  }
  else
  {
    m_MultiWindowLayout = windowLayout;
  }
}


//-----------------------------------------------------------------------------
MIDASOrientation niftkMultiViewerWidget::GetOrientation() const
{
  MIDASOrientation orientation;

  switch (m_ControlPanel->GetWindowLayout())
  {
  case WINDOW_LAYOUT_AXIAL:
    orientation = MIDAS_ORIENTATION_AXIAL;
    break;
  case WINDOW_LAYOUT_SAGITTAL:
    orientation = MIDAS_ORIENTATION_SAGITTAL;
    break;
  case WINDOW_LAYOUT_CORONAL:
    orientation = MIDAS_ORIENTATION_CORONAL;
    break;
  default:
    orientation = MIDAS_ORIENTATION_UNKNOWN;
    break;
  }

  return orientation;
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidget::GetSelectedViewerIndex() const
{
  int selectedViewerIndex = m_SelectedViewerIndex;
  if (selectedViewerIndex < 0 || selectedViewerIndex >= m_Viewers.size())
  {
    // Default back to first viewer.
    selectedViewerIndex = 0;
  }

  // Note the following specification.
  assert(selectedViewerIndex >= 0);
  assert(selectedViewerIndex < m_Viewers.size());

  // Return a valid selected viewer index.
  return selectedViewerIndex;
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidget* niftkMultiViewerWidget::GetSelectedViewer() const
{
  int selectedViewerIndex = m_SelectedViewerIndex;
  if (selectedViewerIndex < 0 || selectedViewerIndex >= m_Viewers.size())
  {
    // Default back to first viewer.
    selectedViewerIndex = 0;
  }

  return m_Viewers[selectedViewerIndex];
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkMultiViewerWidget::GetSelectedRenderWindow() const
{
  // NOTE: This MUST always return not-null.
  return this->GetSelectedViewer()->GetSelectedRenderWindow();
}


//-----------------------------------------------------------------------------
QHash<QString,QmitkRenderWindow*> niftkMultiViewerWidget::GetRenderWindows() const
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

  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  renderWindows.insert("axial", selectedViewer->GetAxialWindow());
  renderWindows.insert("sagittal", selectedViewer->GetSagittalWindow());
  renderWindows.insert("coronal", selectedViewer->GetCoronalWindow());
  renderWindows.insert("3d", selectedViewer->Get3DWindow());

  int i = 0;
  foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
  {
    if (otherViewer != selectedViewer)
    {
      QString id = tr(".%1").arg(i);

      renderWindows.insert("axial" + id, otherViewer->GetAxialWindow());
      renderWindows.insert("sagittal" + id, otherViewer->GetSagittalWindow());
      renderWindows.insert("coronal" + id, otherViewer->GetCoronalWindow());
      renderWindows.insert("3d" + id, otherViewer->Get3DWindow());
      ++i;
    }
  }

  return renderWindows;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkMultiViewerWidget::GetRenderWindow(const QString& id) const
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
mitk::Point3D niftkMultiViewerWidget::GetSelectedPosition(const QString& id) const
{
  if (id.isNull())
  {
    return this->GetSelectedViewer()->GetSelectedPosition();
  }
  else
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(id);
    if (renderWindow)
    {
      foreach (niftkSingleViewerWidget* viewer, m_Viewers)
      {
        if (viewer->ContainsRenderWindow(renderWindow))
        {
          return viewer->GetSelectedPosition();
        }
      }
    }
  }
  mitk::Point3D fallBackValue;
  fallBackValue.Fill(0.0);
  return fallBackValue;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition, const QString& id)
{
  if (id.isNull())
  {
    this->GetSelectedViewer()->SetSelectedPosition(selectedPosition);
  }
  else
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(id);
    if (renderWindow)
    {
      foreach (niftkSingleViewerWidget* viewer, m_Viewers)
      {
        if (viewer->ContainsRenderWindow(renderWindow))
        {
          viewer->SetSelectedPosition(selectedPosition);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::EnableLinkedNavigation(bool enabled)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  if (enabled && !m_LinkedNavigation)
  {
    selectedViewer->EnableLinkedNavigation(true);
  }
  else if (!enabled && m_LinkedNavigation)
  {
    selectedViewer->EnableLinkedNavigation(false);
  }
  m_LinkedNavigation = enabled;
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::IsLinkedNavigationEnabled() const
{
  return m_LinkedNavigation;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSelectedViewerByIndex(int selectedViewerIndex)
{
  if (selectedViewerIndex >= 0 && selectedViewerIndex < m_Viewers.size())
  {
    m_SelectedViewerIndex = selectedViewerIndex;
    niftkSingleViewerWidget* selectedViewer = m_Viewers[selectedViewerIndex];

    for (int i = 0; i < m_Viewers.size(); i++)
    {
      niftkSingleViewerWidget* viewer = m_Viewers[i];

      int nodesInWindow = m_VisibilityManager->GetNodesInViewer(i);

      if (viewer == selectedViewer)
      {
        viewer->SetSelected(nodesInWindow > 0);
        viewer->EnableLinkedNavigation(true);
      }
      else
      {
        viewer->SetSelected(false);
        viewer->EnableLinkedNavigation(false);
      }
    }

    m_ControlPanel->SetWindowCursorsBound(selectedViewer->GetCursorPositionBinding());
    m_ControlPanel->SetWindowMagnificationsBound(selectedViewer->GetScaleFactorBinding());

    this->OnCursorVisibilityChanged(selectedViewer, selectedViewer->IsCursorVisible());
    this->RequestUpdateAll();
  }
  else
  {
    MITK_WARN << "Ignoring request to set the selected window to window number " << selectedViewerIndex << std::endl;
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerPositionBindingChanged(bool bound)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (bound)
  {
    const mitk::Point3D& selectedPosition = selectedViewer->GetSelectedPosition();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetSelectedPosition(selectedPosition);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerCursorBindingChanged(bool bound)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  bool windowCursorPositionsBound = selectedViewer->GetCursorPositionBinding();

  if (bound)
  {
    MIDASOrientation orientation = selectedViewer->GetOrientation();
    mitk::Vector2D cursorPosition = selectedViewer->GetCursorPosition(orientation);
    const std::vector<mitk::Vector2D>& cursorPositions = selectedViewer->GetCursorPositions();

    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetCursorPositionBinding(windowCursorPositionsBound);
        otherViewer->blockSignals(signalsWereBlocked);
        if (windowCursorPositionsBound)
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetCursorPositions(cursorPositions);
          otherViewer->blockSignals(signalsWereBlocked);
        }
        else
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetCursorPosition(orientation, cursorPosition);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerMagnificationBindingChanged(bool bound)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  bool windowScaleFactorsBound = selectedViewer->GetScaleFactorBinding();

  if (bound)
  {
    MIDASOrientation orientation = selectedViewer->GetOrientation();
    double scaleFactor = selectedViewer->GetScaleFactor(orientation);
    std::vector<double> scaleFactors = selectedViewer->GetScaleFactors();

    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetScaleFactorBinding(windowScaleFactorsBound);
        otherViewer->blockSignals(signalsWereBlocked);
        if (windowScaleFactorsBound)
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetScaleFactors(scaleFactors);
          otherViewer->blockSignals(signalsWereBlocked);
        }
        else
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetScaleFactor(orientation, scaleFactor);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerWindowLayoutBindingChanged(bool bound)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (bound)
  {
    WindowLayout windowLayout = selectedViewer->GetWindowLayout();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetWindowLayout(windowLayout);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerGeometryBindingChanged(bool bound)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (bound)
  {
    mitk::TimeGeometry* geometry = selectedViewer->GetGeometry();

    foreach (niftkSingleViewerWidget* viewer, m_Viewers)
    {
      bool signalsWereBlocked = viewer->blockSignals(true);
      viewer->SetBoundGeometry(geometry);
      viewer->SetBoundGeometryActive(true);
      viewer->blockSignals(signalsWereBlocked);
    }
  }
  else
  {
    foreach (niftkSingleViewerWidget* viewer, m_Viewers)
    {
      bool signalsWereBlocked = viewer->blockSignals(true);
      viewer->SetBoundGeometryActive(false);
      viewer->blockSignals(signalsWereBlocked);
    }
  }

  this->OnCursorVisibilityChanged(selectedViewer, selectedViewer->IsCursorVisible());
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnPopupOpened(bool opened)
{
  if (!opened)
  {
    this->GetSelectedViewer()->repaint();
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSliceTracking(bool tracking)
{
  m_ControlPanel->SetSliceTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetTimeStepTracking(bool tracking)
{
  m_ControlPanel->SetTimeStepTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetMagnificationTracking(bool tracking)
{
  m_ControlPanel->SetMagnificationTracking(tracking);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnPinButtonToggled(bool checked)
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
bool niftkMultiViewerWidget::eventFilter(QObject* object, QEvent* event)
{
  if (object == m_PinButton && event->type() == QEvent::Enter)
  {
    m_PopupWidget->showPopup();
  }
  return this->QObject::eventFilter(object, event);
}
