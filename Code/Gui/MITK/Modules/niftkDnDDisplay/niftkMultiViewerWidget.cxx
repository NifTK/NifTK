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
, m_Show2DCursors(false)
, m_Show3DWindowIn2x2WindowLayout(false)
, m_RememberSettingsPerLayout(false)
, m_IsThumbnailMode(false)
, m_SegmentationModeEnabled(false)
, m_NavigationControllerEventListening(false)
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
  this->connect(m_ControlPanel, SIGNAL(SliceIndexChanged(int)), SLOT(OnSliceIndexChanged(int)));
  this->connect(m_ControlPanel, SIGNAL(TimeStepChanged(int)), SLOT(OnTimeStepChanged(int)));
  this->connect(m_ControlPanel, SIGNAL(MagnificationChanged(double)), SLOT(OnMagnificationChanged(double)));

  this->connect(m_ControlPanel, SIGNAL(ShowCursorChanged(bool)), SLOT(OnShowCursorChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ShowDirectionAnnotationsChanged(bool)), SLOT(OnShowDirectionAnnotationsChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(Show3DWindowChanged(bool)), SLOT(OnShow3DWindowChanged(bool)));

  this->connect(m_ControlPanel, SIGNAL(WindowLayoutChanged(WindowLayout)), SLOT(OnWindowLayoutChanged(WindowLayout)));
  this->connect(m_ControlPanel, SIGNAL(WindowCursorBindingChanged(bool)), SLOT(OnWindowCursorBindingChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(WindowMagnificationBindingChanged(bool)), SLOT(OnWindowMagnificationBindingChanged(bool)));

  this->connect(m_ControlPanel, SIGNAL(ViewerNumberChanged(int, int)), SLOT(OnViewerNumberChanged(int, int)));

  this->connect(m_ControlPanel, SIGNAL(ViewerPositionBindingChanged(bool)), SLOT(OnViewerPositionBindingChanged()));
  this->connect(m_ControlPanel, SIGNAL(ViewerCursorBindingChanged(bool)), SLOT(OnViewerCursorBindingChanged()));
  this->connect(m_ControlPanel, SIGNAL(ViewerMagnificationBindingChanged(bool)), SLOT(OnViewerMagnificationBindingChanged()));
  this->connect(m_ControlPanel, SIGNAL(ViewerLayoutBindingChanged(bool)), SLOT(OnViewerLayoutBindingChanged()));
  this->connect(m_ControlPanel, SIGNAL(ViewerGeometryBindingChanged(bool)), SLOT(OnViewerGeometryBindingChanged()));

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
  this->Deactivated();
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidget* niftkMultiViewerWidget::CreateViewer()
{
  niftkSingleViewerWidget* viewer = new niftkSingleViewerWidget(tr("QmitkRenderWindow"),
                                                                      -5, 20,
                                                                      this,
                                                                      m_RenderingManager,
                                                                      m_DataStorage);
  viewer->setObjectName(tr("niftkSingleViewerWidget"));
  viewer->setVisible(false);

  viewer->SetBackgroundColor(m_BackgroundColour);
  viewer->SetShow3DWindowIn2x2WindowLayout(m_Show3DWindowIn2x2WindowLayout);
  viewer->SetRememberSettingsPerWindowLayout(m_RememberSettingsPerLayout);
  viewer->SetDisplayInteractionsEnabled(true);
  viewer->SetCursorPositionsBound(true);
  viewer->SetScaleFactorsBound(true);
  viewer->SetDefaultSingleWindowLayout(m_SingleWindowLayout);
  viewer->SetDefaultMultiWindowLayout(m_MultiWindowLayout);

  m_VisibilityManager->connect(viewer, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(viewer, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  this->connect(viewer, SIGNAL(SelectedPositionChanged(niftkSingleViewerWidget*, QmitkRenderWindow*, int)), SLOT(OnSelectedPositionChanged(niftkSingleViewerWidget*, QmitkRenderWindow*, int)));
  this->connect(viewer, SIGNAL(CursorPositionChanged(niftkSingleViewerWidget*, const mitk::Vector3D&)), SLOT(OnCursorPositionChanged(niftkSingleViewerWidget*, const mitk::Vector3D&)));
  this->connect(viewer, SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, double)), SLOT(OnScaleFactorChanged(niftkSingleViewerWidget*, double)));
  this->connect(viewer, SIGNAL(WindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout)), SLOT(OnWindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout)));

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
bool niftkMultiViewerWidget::GetShow2DCursors() const
{
  return m_Show2DCursors;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetShow2DCursors(bool visible)
{
  m_Show2DCursors = visible;

  m_ControlPanel->SetCursorVisible(visible);

  this->Update2DCursorVisibility();
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
  this->RequestUpdateAll();
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
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetRememberSettingsPerWindowLayout(bool rememberSettingsPerLayout)
{
  m_RememberSettingsPerLayout = rememberSettingsPerLayout;
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerLayout);
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
    m_ControlPanel->SetSingleViewerControlsEnabled(m_NavigationControllerEventListening);
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

  WindowLayout layout = m_VisibilityManager->GetDefaultWindowLayout();

  if (   layout != WINDOW_LAYOUT_AXIAL
      && layout != WINDOW_LAYOUT_SAGITTAL
      && layout != WINDOW_LAYOUT_CORONAL
     )
  {
    layout = WINDOW_LAYOUT_CORONAL;
  }

  return layout;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetBackgroundColour(QColor backgroundColour)
{
  m_BackgroundColour = backgroundColour;

  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetBackgroundColor(m_BackgroundColour);
  }

  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetViewerNumber(int viewerRows, int viewerColumns, bool isThumbnailMode)
{
  // Work out required number of viewers, and hence if we need to create any new ones.
  int requiredNumberOfViewers = viewerRows * viewerColumns;
  int currentNumberOfViewers = m_Viewers.size();

  // If we have the right number of viewers, there is nothing to do, so early exit.
  if (requiredNumberOfViewers == currentNumberOfViewers)
  {
    return;
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

  int viewerCounter = 0;
  for (int row = 0; row < viewerRows; row++)
  {
    for (int column = 0; column < viewerColumns; column++)
    {
      m_LayoutForRenderWindows->addWidget(m_Viewers[viewerCounter], row, column);
      m_Viewers[viewerCounter]->show();
      m_Viewers[viewerCounter]->setEnabled(true);
      viewerCounter++;
    }
  }

  ////////////////////////////////////////
  // End: Rebuild the number of viewers.
  ////////////////////////////////////////

  // Update row/column of viewers without triggering another layout size change.
  m_ControlPanel->SetViewerNumber(viewerRows, viewerColumns);

  // Test the current m_Selected window, and reset to 0 if it now points to an invisible window.
  int selectedViewerIndex = this->GetSelectedViewerIndex();
  QmitkRenderWindow* selectedRenderWindow = this->GetSelectedRenderWindow();
  if (this->GetViewerRowFromIndex(selectedViewerIndex) >= viewerRows || this->GetViewerColumnFromIndex(selectedViewerIndex) >= viewerColumns)
  {
    selectedViewerIndex = 0;
    selectedRenderWindow = m_Viewers[selectedViewerIndex]->GetSelectedRenderWindow();
  }
  this->SetSelectedRenderWindow(selectedViewerIndex, selectedRenderWindow);

  // Now the number of viewers has changed, we need to make sure they are all in synch with all the right properties.
  this->Update2DCursorVisibility();
  this->SetShow3DWindowIn2x2WindowLayout(m_Show3DWindowIn2x2WindowLayout);

  if (m_ControlPanel->AreViewerGeometriesBound())
  {
    niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    mitk::TimeGeometry* geometry = selectedViewer->GetGeometry();

    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        otherViewer->SetBoundGeometry(geometry);
      }
    }
  }

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    this->UpdateBoundMagnification();
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
void niftkMultiViewerWidget::OnSelectedPositionChanged(niftkSingleViewerWidget* viewer, QmitkRenderWindow* renderWindow, int sliceIndex)
{
  // If the viewer is not found, we do not do anything.
  if (std::find(m_Viewers.begin(), m_Viewers.end(), viewer) == m_Viewers.end())
  {
    return;
  }

  m_ControlPanel->SetSliceIndex(viewer->GetSliceIndex(viewer->GetOrientation()));

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    mitk::Point3D selectedPosition = viewer->GetSelectedPosition();
    for (int i = 0; i < m_Viewers.size(); i++)
    {
      if (m_Viewers[i] != viewer)
      {
        m_Viewers[i]->SetSelectedPosition(selectedPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnCursorPositionChanged(niftkSingleViewerWidget* viewer, const mitk::Vector3D& cursorPosition)
{
  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        otherViewer->SetCursorPosition(cursorPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnScaleFactorChanged(niftkSingleViewerWidget* viewer, double scaleFactor)
{
  double magnification = viewer->GetMagnification();
  m_ControlPanel->SetMagnification(magnification);

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != viewer)
      {
        otherViewer->SetScaleFactor(scaleFactor);
      }
    }
  }
  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnNodesDropped(QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes)
{
  mitk::DataStorage::SetOfObjects::Pointer nodeSet = mitk::DataStorage::SetOfObjects::New();
  for (unsigned i = 0; i < nodes.size(); ++i)
  {
    nodeSet->InsertElement(i, nodes[i]);
  }
  // calculate bounding geometry of these nodes
  mitk::TimeGeometry::Pointer bounds = m_DataStorage->ComputeBoundingGeometry3D(nodeSet);
  // initialize the viewers to the bounding geometry
  m_RenderingManager->InitializeViews(bounds);

  // See also niftkMultiViewerVisibilityManager::OnNodesDropped which should trigger first.
  if (m_ControlPanel->GetDropType() != DNDDISPLAY_DROP_ALL)
  {
    m_ControlPanel->SetSingleViewerControlsEnabled(true);
  }

  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  niftkSingleViewerWidget* dropOntoViewer = 0;

  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    if (viewer->ContainsRenderWindow(renderWindow))
    {
      dropOntoViewer = viewer;
      MIDASOrientation orientation = dropOntoViewer->GetOrientation();
      switch (orientation)
      {
      case MIDAS_ORIENTATION_AXIAL:
        renderWindow = dropOntoViewer->GetAxialWindow();
        break;
      case MIDAS_ORIENTATION_SAGITTAL:
        renderWindow = dropOntoViewer->GetSagittalWindow();
        break;
      case MIDAS_ORIENTATION_CORONAL:
        renderWindow = dropOntoViewer->GetCoronalWindow();
        break;
      case MIDAS_ORIENTATION_UNKNOWN:
        renderWindow = dropOntoViewer->Get3DWindow();
        break;
      }
      break;
    }
  }

  // This does not trigger OnFocusChanged() the very first time, as when creating the editor, the first viewer already has focus.
  mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(renderWindow->GetRenderer());

  double magnification = selectedViewer->GetMagnification();

  WindowLayout layout = selectedViewer->GetWindowLayout();
  if (m_ControlPanel->AreViewerWindowLayoutsBound())
  {
    dropOntoViewer->SetWindowLayout(layout);
  }

  //  double scaleFactor = selectedViewer->GetScaleFactor();
  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    dropOntoViewer->SetMagnification(magnification);
  }

//  m_ControlPanel->SetMagnification(magnification);
//  m_ControlPanel->SetMagnification(scaleFactor);
//  this->OnMagnificationChanged(magnification);
//  this->OnMagnificationChanged(scaleFactor);
//  this->SetMagnification(magnification);

  this->Update2DCursorVisibility();
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
        selectedViewer->SetSelectedRenderWindow(selectedRenderWindow);
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
      unsigned int maxSliceIndex = selectedViewer->GetMaxSliceIndex(orientation);
      unsigned int sliceIndex = selectedViewer->GetSliceIndex(orientation);
      m_ControlPanel->SetMaxSliceIndex(maxSliceIndex);
      m_ControlPanel->SetSliceIndex(sliceIndex);
    }

    unsigned int maxTimeStep = selectedViewer->GetMaxTimeStep();
    unsigned int timeStep = selectedViewer->GetTimeStep();
    m_ControlPanel->SetMaxTimeStep(maxTimeStep);
    m_ControlPanel->SetTimeStep(timeStep);

    double minMagnification = std::ceil(selectedViewer->GetMinMagnification());
    double maxMagnification = std::floor(selectedViewer->GetMaxMagnification());
    double magnification = selectedViewer->GetMagnification();
    m_ControlPanel->SetMinMagnification(minMagnification);
    m_ControlPanel->SetMaxMagnification(maxMagnification);
    m_ControlPanel->SetMagnification(magnification);

    this->Update2DCursorVisibility();
  }
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetFocus()
{
  this->GetSelectedViewer()->setFocus();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* renderer = focusManager->GetFocused();

  int selectedViewerIndex = -1;
  vtkRenderWindow* focusedVtkRenderWindow = NULL;
  QmitkRenderWindow* focusedRenderWindow = NULL;

  if (renderer)
  {
    focusedVtkRenderWindow = renderer->GetRenderWindow();
    for (int i = 0; i < m_Viewers.size(); i++)
    {
      QmitkRenderWindow* renderWindow = m_Viewers[i]->GetRenderWindow(focusedVtkRenderWindow);
      if (renderWindow != NULL)
      {
        selectedViewerIndex = i;
        focusedRenderWindow = renderWindow;
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
void niftkMultiViewerWidget::OnSliceIndexChanged(int sliceIndex)
{
  this->SetSelectedWindowSliceIndex(sliceIndex);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSelectedWindowSliceIndex(int sliceIndex)
{
  MIDASOrientation orientation = this->GetSelectedViewer()->GetOrientation();

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    selectedViewer->SetSliceIndex(orientation, sliceIndex);

    if (m_ControlPanel->AreViewerPositionsBound())
    {
      foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
      {
        if (otherViewer != selectedViewer && otherViewer->isVisible())
        {
          otherViewer->SetSliceIndex(orientation, sliceIndex);
        }
      }
    }
  }
  else
  {
    MITK_WARN << "Found an invalid orientation in viewer " << this->GetSelectedViewerIndex() << ", so ignoring request to change to slice " << sliceIndex << std::endl;
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
  selectedViewer->SetMagnification(magnification);

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        otherViewer->SetMagnification(magnification);
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
  selectedViewer->SetTimeStep(timeStep);

  if (dropType == DNDDISPLAY_DROP_ALL)
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        otherViewer->SetTimeStep(timeStep);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowLayoutChanged(WindowLayout windowLayout)
{
  if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    this->SetLayout(windowLayout);

    // Update the focus to the selected window, to trigger things like thumbnail viewer refresh
    // (or indeed anything that's listening to the FocusManager).
    this->UpdateFocusManagerToSelectedViewer();
  }

}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowLayoutChanged(niftkSingleViewerWidget* selectedViewer, WindowLayout windowLayout)
{
  m_ControlPanel->SetWindowLayout(windowLayout);
  this->UpdateFocusManagerToSelectedViewer();

  if (m_ControlPanel->AreViewerWindowLayoutsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        otherViewer->SetWindowLayout(windowLayout);
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
      viewer->SetBoundGeometry(geometry);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowCursorBindingChanged(bool bound)
{
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    if (viewer->isVisible())
    {
      viewer->SetCursorPositionsBound(bound);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowMagnificationBindingChanged(bool bound)
{
  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    if (viewer->isVisible())
    {
      viewer->SetScaleFactorsBound(bound);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnShowCursorChanged(bool visible)
{
  this->SetShow2DCursors(visible);
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
bool niftkMultiViewerWidget::ToggleCursor()
{
  this->SetShow2DCursors(!this->GetShow2DCursors());

  return true;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetLayout(WindowLayout windowLayout)
{
  m_ControlPanel->SetWindowLayout(windowLayout);

  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  selectedViewer->SetWindowLayout(windowLayout);

  if (windowLayout == WINDOW_LAYOUT_AXIAL)
  {
    selectedViewer->SetSelectedRenderWindow(selectedViewer->GetAxialWindow());
  }
  else if (windowLayout == WINDOW_LAYOUT_SAGITTAL)
  {
    selectedViewer->SetSelectedRenderWindow(selectedViewer->GetSagittalWindow());
  }
  else if (windowLayout == WINDOW_LAYOUT_CORONAL)
  {
    selectedViewer->SetSelectedRenderWindow(selectedViewer->GetCoronalWindow());
  }

  if (m_ControlPanel->AreViewerWindowLayoutsBound())
  {
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        otherViewer->SetWindowLayout(windowLayout);
      }
    }
  }

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    double magnification = selectedViewer->GetMagnification();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        otherViewer->SetMagnification(magnification);
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
void niftkMultiViewerWidget::Update2DCursorVisibility()
{
  bool globalVisibility = false;
  bool localVisibility = m_Show2DCursors;

  foreach (niftkSingleViewerWidget* viewer, m_Viewers)
  {
    if (viewer->isVisible())
    {
      viewer->SetDisplay2DCursorsGlobally(globalVisibility);
      viewer->SetDisplay2DCursorsLocally(localVisibility);
    }
  }

  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::UpdateBoundMagnification()
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  double magnification = selectedViewer->GetMagnification();
  foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
  {
    if (otherViewer != selectedViewer)
    {
      otherViewer->SetMagnification(magnification);
    }
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
void niftkMultiViewerWidget::Activated()
{
//  this->setEnabled(true);
  this->EnableLinkedNavigation(true);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::Deactivated()
{
//  this->setEnabled(false);
  this->EnableLinkedNavigation(false);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::EnableLinkedNavigation(bool enable)
{
  this->SetNavigationControllerEventListening(enable);
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::IsLinkedNavigationEnabled() const
{
  return this->GetNavigationControllerEventListening();
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::GetNavigationControllerEventListening() const
{
  return m_NavigationControllerEventListening;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetNavigationControllerEventListening(bool enabled)
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  if (enabled && !m_NavigationControllerEventListening)
  {
    selectedViewer->SetNavigationControllerEventListening(true);
  }
  else if (!enabled && m_NavigationControllerEventListening)
  {
    selectedViewer->SetNavigationControllerEventListening(false);
  }
  m_NavigationControllerEventListening = enabled;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSelectedViewerByIndex(int selectedViewerIndex)
{
  if (selectedViewerIndex >= 0 && selectedViewerIndex < m_Viewers.size())
  {
    m_SelectedViewerIndex = selectedViewerIndex;

    for (int i = 0; i < m_Viewers.size(); i++)
    {
      int nodesInWindow = m_VisibilityManager->GetNodesInViewer(i);

      if (i == selectedViewerIndex && nodesInWindow > 0)
      {
        m_Viewers[i]->SetSelected(true);
      }
      else
      {
        m_Viewers[i]->SetSelected(false);
      }

      if (i == selectedViewerIndex)
      {
        m_Viewers[i]->SetNavigationControllerEventListening(true);
      }
      else
      {
        m_Viewers[i]->SetNavigationControllerEventListening(false);
      }
    }
    this->Update2DCursorVisibility();
    this->RequestUpdateAll();
  }
  else
  {
    MITK_WARN << "Ignoring request to set the selected window to window number " << selectedViewerIndex << std::endl;
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerPositionBindingChanged()
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    mitk::Point3D selectedPosition = selectedViewer->GetSelectedPosition();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        otherViewer->SetSelectedPosition(selectedPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerCursorBindingChanged()
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    mitk::Vector3D cursorPosition = selectedViewer->GetCursorPosition();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        otherViewer->SetCursorPosition(cursorPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerLayoutBindingChanged()
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (m_ControlPanel->AreViewerWindowLayoutsBound())
  {
    WindowLayout windowLayout = selectedViewer->GetWindowLayout();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        otherViewer->SetWindowLayout(windowLayout);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerMagnificationBindingChanged()
{
  niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    double magnification = selectedViewer->GetMagnification();
    foreach (niftkSingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer)
      {
        otherViewer->SetMagnification(magnification);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewerGeometryBindingChanged()
{
  if (m_ControlPanel->AreViewerGeometriesBound())
  {
    niftkSingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    mitk::TimeGeometry* geometry = selectedViewer->GetGeometry();

    foreach (niftkSingleViewerWidget* viewer, m_Viewers)
    {
      viewer->SetBoundGeometry(geometry);
      viewer->SetBoundGeometryActive(true);
    }
  }
  else
  {
    foreach (niftkSingleViewerWidget* viewer, m_Viewers)
    {
      viewer->SetBoundGeometryActive(false);
    }
  }
//  this->Update2DCursorVisibility();
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
void niftkMultiViewerWidget::SetSliceIndexTracking(bool tracking)
{
  m_ControlPanel->SetSliceIndexTracking(tracking);
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
