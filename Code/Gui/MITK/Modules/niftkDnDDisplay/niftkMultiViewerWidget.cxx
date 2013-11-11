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
    int defaultNumberOfRows,
    int defaultNumberOfColumns,
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
, m_SelectedViewIndex(0)
, m_DefaultViewRows(defaultNumberOfRows)
, m_DefaultViewColumns(defaultNumberOfColumns)
, m_Show2DCursors(false)
, m_Show3DWindowInOrthoView(false)
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

  m_ControlPanel->SetDirectionAnnotationsVisible(true);

  // Default to dropping into single window.
  m_ControlPanel->SetDropType(DNDDISPLAY_DROP_SINGLE);
  m_VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);

  // We have the default rows and columns passed in via constructor args, in initialise list.
  m_ControlPanel->SetViewNumber(m_DefaultViewRows, m_DefaultViewColumns);
  this->SetViewNumber(m_DefaultViewRows, m_DefaultViewColumns, false);

  // Connect Qt Signals to make it all hang together.
  QObject::connect(m_ControlPanel, SIGNAL(SliceIndexChanged(int)), this, SLOT(OnSliceIndexChanged(int)));
  QObject::connect(m_ControlPanel, SIGNAL(TimeStepChanged(int)), this, SLOT(OnTimeStepChanged(int)));
  QObject::connect(m_ControlPanel, SIGNAL(MagnificationChanged(double)), this, SLOT(OnMagnificationChanged(double)));

  QObject::connect(m_ControlPanel, SIGNAL(ShowCursorChanged(bool)), this, SLOT(OnShowCursorChanged(bool)));
  QObject::connect(m_ControlPanel, SIGNAL(ShowDirectionAnnotationsChanged(bool)), this, SLOT(OnShowDirectionAnnotationsChanged(bool)));
  QObject::connect(m_ControlPanel, SIGNAL(Show3DWindowChanged(bool)), this, SLOT(OnShow3DWindowChanged(bool)));

  QObject::connect(m_ControlPanel, SIGNAL(LayoutChanged(WindowLayout)), this, SLOT(OnLayoutChanged(WindowLayout)));
  QObject::connect(m_ControlPanel, SIGNAL(WindowCursorBindingChanged(bool)), this, SLOT(OnWindowCursorBindingChanged(bool)));
  QObject::connect(m_ControlPanel, SIGNAL(WindowMagnificationBindingChanged(bool)), this, SLOT(OnWindowMagnificationBindingChanged(bool)));

  QObject::connect(m_ControlPanel, SIGNAL(ViewNumberChanged(int, int)), this, SLOT(OnViewNumberChanged(int, int)));

  QObject::connect(m_ControlPanel, SIGNAL(ViewPositionBindingChanged(bool)), this, SLOT(OnViewPositionBindingChanged()));
  QObject::connect(m_ControlPanel, SIGNAL(ViewCursorBindingChanged(bool)), this, SLOT(OnViewCursorBindingChanged()));
  QObject::connect(m_ControlPanel, SIGNAL(ViewMagnificationBindingChanged(bool)), this, SLOT(OnViewMagnificationBindingChanged()));
  QObject::connect(m_ControlPanel, SIGNAL(ViewLayoutBindingChanged(bool)), this, SLOT(OnViewLayoutBindingChanged()));
  QObject::connect(m_ControlPanel, SIGNAL(ViewGeometryBindingChanged(bool)), this, SLOT(OnViewGeometryBindingChanged()));

  QObject::connect(m_ControlPanel, SIGNAL(DropTypeChanged(DnDDisplayDropType)), this, SLOT(OnDropTypeChanged(DnDDisplayDropType)));
  QObject::connect(m_ControlPanel, SIGNAL(DropAccumulateChanged(bool)), this, SLOT(OnDropAccumulateChanged(bool)));

  QObject::connect(m_PopupWidget, SIGNAL(popupOpened(bool)), this, SLOT(OnPopupOpened(bool)));

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

  controlPanel->SetMaxViewNumber(m_MaxViewRows, m_MaxViewColumns);

  controlPanel->SetWindowCursorsBound(true);
  controlPanel->SetWindowMagnificationsBound(true);

  QHBoxLayout* controlPanelLayout = new QHBoxLayout(parent);
  controlPanelLayout->setContentsMargins(0, 0, 0, 0);
  controlPanelLayout->setSpacing(0);
  controlPanelLayout->addWidget(controlPanel);

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
niftkSingleViewerWidget* niftkMultiViewerWidget::CreateSingleViewWidget()
{
  niftkSingleViewerWidget* widget = new niftkSingleViewerWidget(tr("QmitkRenderWindow"),
                                                                      -5, 20,
                                                                      this,
                                                                      m_RenderingManager,
                                                                      m_DataStorage);
  widget->setObjectName(tr("niftkSingleViewerWidget"));
  widget->setVisible(false);

  widget->SetBackgroundColor(m_BackgroundColour);
  widget->SetShow3DWindowInOrthoView(m_Show3DWindowInOrthoView);
  widget->SetRememberSettingsPerLayout(m_RememberSettingsPerLayout);
  widget->SetDisplayInteractionsEnabled(true);
  widget->SetCursorPositionsBound(true);
  widget->SetScaleFactorsBound(true);
  widget->SetDefaultSingleWindowLayout(m_SingleWindowLayout);
  widget->SetDefaultMultiWindowLayout(m_MultiWindowLayout);

  QObject::connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), m_VisibilityManager, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  QObject::connect(widget, SIGNAL(NodesDropped(QmitkRenderWindow*, std::vector<mitk::DataNode*>)), this, SLOT(OnNodesDropped(QmitkRenderWindow*,std::vector<mitk::DataNode*>)), Qt::DirectConnection);
  QObject::connect(widget, SIGNAL(SelectedPositionChanged(niftkSingleViewerWidget*, QmitkRenderWindow*, int)), this, SLOT(OnSelectedPositionChanged(niftkSingleViewerWidget*, QmitkRenderWindow*, int)));
  QObject::connect(widget, SIGNAL(CursorPositionChanged(niftkSingleViewerWidget*, const mitk::Vector3D&)), this, SLOT(OnCursorPositionChanged(niftkSingleViewerWidget*, const mitk::Vector3D&)));
  QObject::connect(widget, SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, double)), this, SLOT(OnScaleFactorChanged(niftkSingleViewerWidget*, double)));
  QObject::connect(widget, SIGNAL(LayoutChanged(niftkSingleViewerWidget*, WindowLayout)), this, SLOT(OnLayoutChanged(niftkSingleViewerWidget*, WindowLayout)));

  return widget;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::RequestUpdateAll()
{
  foreach (niftkSingleViewerWidget* view, m_SingleViewWidgets)
  {
    if (view->isVisible())
    {
      view->RequestUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetInterpolationType(DnDDisplayInterpolationType interpolationType)
{
  m_VisibilityManager->SetInterpolationType(interpolationType);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetDefaultLayout(WindowLayout windowLayout)
{
  m_VisibilityManager->SetDefaultLayout(windowLayout);
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
bool niftkMultiViewerWidget::AreViewNumberControlsVisible() const
{
  return m_ControlPanel->AreViewNumberControlsVisible();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetViewNumberControlsVisible(bool visible)
{
  m_ControlPanel->SetViewNumberControlsVisible(visible);
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

    m_VisibilityManager->ClearAllWindows();
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
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetDirectionAnnotationsVisible(visible);
  }
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::GetShow3DWindowInOrthoView() const
{
  return m_Show3DWindowInOrthoView;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetShow3DWindowInOrthoView(bool visible)
{
  m_Show3DWindowInOrthoView = visible;
  m_ControlPanel->Set3DWindowVisible(visible);
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetShow3DWindowInOrthoView(visible);
  }
  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetRememberSettingsPerLayout(bool rememberSettingsPerLayout)
{
  m_RememberSettingsPerLayout = rememberSettingsPerLayout;
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetRememberSettingsPerLayout(rememberSettingsPerLayout);
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetThumbnailMode(bool enabled)
{
  m_IsThumbnailMode = enabled;

  if (enabled)
  {
    m_ViewRowsInNonThumbnailMode = m_ControlPanel->GetViewRows();
    m_ViewColumnsInNonThumbnailMode = m_ControlPanel->GetViewColumns();
    m_ControlPanel->SetSingleViewControlsEnabled(false);
    m_ControlPanel->SetViewNumber(m_MaxViewRows, m_MaxViewColumns);
    m_ControlPanel->SetMultiViewControlsEnabled(false);
    this->SetViewNumber(m_MaxViewRows, m_MaxViewColumns, true);
  }
  else
  {
    m_ControlPanel->SetSingleViewControlsEnabled(m_NavigationControllerEventListening);
    m_ControlPanel->SetMultiViewControlsEnabled(true);
    m_ControlPanel->SetViewNumber(m_ViewRowsInNonThumbnailMode, m_ViewColumnsInNonThumbnailMode);
    this->SetViewNumber(m_ViewRowsInNonThumbnailMode, m_ViewColumnsInNonThumbnailMode, false);
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
    m_ViewRowsBeforeSegmentationMode = m_ControlPanel->GetViewRows();
    m_ViewColumnsBeforeSegmentationMode = m_ControlPanel->GetViewColumns();
    m_ControlPanel->SetMultiViewControlsEnabled(false);
    this->SetViewNumber(1, 1, false);
    this->SetSelectedViewIndex(0);
    this->UpdateFocusManagerToSelectedView();
  }
  else
  {
    m_ControlPanel->SetMultiViewControlsEnabled(true);
    this->SetViewNumber(m_ViewRowsBeforeSegmentationMode, m_ViewColumnsBeforeSegmentationMode, false);
  }
}


//-----------------------------------------------------------------------------
bool niftkMultiViewerWidget::IsSegmentationModeEnabled() const
{
  return m_SegmentationModeEnabled;
}


//-----------------------------------------------------------------------------
WindowLayout niftkMultiViewerWidget::GetDefaultLayoutForSegmentation() const
{
  assert(m_VisibilityManager);

  WindowLayout layout = m_VisibilityManager->GetDefaultLayout();

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

  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    m_SingleViewWidgets[i]->SetBackgroundColor(m_BackgroundColour);
  }

  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetViewNumber(int viewRows, int viewColumns, bool isThumbnailMode)
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
      niftkSingleViewerWidget* view = this->CreateSingleViewWidget();
      view->hide();

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
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("niftkMultiViewerWidget::m_LayoutForRenderWindows"));
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

  if (m_ControlPanel->AreViewGeometriesBound())
  {
    niftkSingleViewerWidget* selectedView = this->GetSelectedView();
    mitk::TimeGeometry* geometry = selectedView->GetGeometry();

    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView)
      {
        otherView->SetBoundGeometry(geometry);
      }
    }
  }

  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    this->UpdateBoundMagnification();
  }
}


//-----------------------------------------------------------------------------
int niftkMultiViewerWidget::GetRowFromIndex(int i) const
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
int niftkMultiViewerWidget::GetColumnFromIndex(int i) const
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
int niftkMultiViewerWidget::GetIndexFromRowAndColumn(int r, int c) const
{
  return r * m_MaxViewColumns + c;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewNumberChanged(int rows, int columns)
{
  this->SetViewNumber(rows, columns, false);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnSelectedPositionChanged(niftkSingleViewerWidget* view, QmitkRenderWindow* renderWindow, int sliceIndex)
{
  // If the view is not found, we do not do anything.
  if (std::find(m_SingleViewWidgets.begin(), m_SingleViewWidgets.end(), view) == m_SingleViewWidgets.end())
  {
    return;
  }

  m_ControlPanel->SetSliceIndex(view->GetSliceIndex(view->GetOrientation()));

  if (m_ControlPanel->AreViewPositionsBound())
  {
    mitk::Point3D selectedPosition = view->GetSelectedPosition();
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != view)
      {
        m_SingleViewWidgets[i]->SetSelectedPosition(selectedPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnCursorPositionChanged(niftkSingleViewerWidget* widget, const mitk::Vector3D& cursorPosition)
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
void niftkMultiViewerWidget::OnScaleFactorChanged(niftkSingleViewerWidget* view, double scaleFactor)
{
  double magnification = view->GetMagnification();
  m_ControlPanel->SetMagnification(magnification);

  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    for (int i = 0; i < m_SingleViewWidgets.size(); i++)
    {
      if (m_SingleViewWidgets[i] != view)
      {
        m_SingleViewWidgets[i]->SetScaleFactor(scaleFactor);
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
  // initialize the views to the bounding geometry
  m_RenderingManager->InitializeViews(bounds);

  // See also niftkMultiViewerVisibilityManager::OnNodesDropped which should trigger first.
  if (m_ControlPanel->GetDropType() != DNDDISPLAY_DROP_ALL)
  {
    m_ControlPanel->SetSingleViewControlsEnabled(true);
  }

  niftkSingleViewerWidget* selectedView = this->GetSelectedView();

  niftkSingleViewerWidget* dropOntoView = 0;

  foreach (niftkSingleViewerWidget* view, m_SingleViewWidgets)
  {
    if (view->ContainsRenderWindow(renderWindow))
    {
      dropOntoView = view;
      MIDASOrientation orientation = dropOntoView->GetOrientation();
      switch (orientation)
      {
      case MIDAS_ORIENTATION_AXIAL:
        renderWindow = dropOntoView->GetAxialWindow();
        break;
      case MIDAS_ORIENTATION_SAGITTAL:
        renderWindow = dropOntoView->GetSagittalWindow();
        break;
      case MIDAS_ORIENTATION_CORONAL:
        renderWindow = dropOntoView->GetCoronalWindow();
        break;
      case MIDAS_ORIENTATION_UNKNOWN:
        renderWindow = dropOntoView->Get3DWindow();
        break;
      }
      break;
    }
  }

  // This does not trigger OnFocusChanged() the very first time, as when creating the editor, the first widget already has focus.
  mitk::GlobalInteraction::GetInstance()->GetFocusManager()->SetFocused(renderWindow->GetRenderer());

  double magnification = selectedView->GetMagnification();

  WindowLayout layout = selectedView->GetLayout();
  if (m_ControlPanel->AreViewLayoutsBound())
  {
    dropOntoView->SetLayout(layout);
  }

  //  double scaleFactor = selectedView->GetScaleFactor();
  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    dropOntoView->SetMagnification(magnification);
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
void niftkMultiViewerWidget::SetSelectedRenderWindow(int selectedViewIndex, QmitkRenderWindow* selectedRenderWindow)
{
  if (selectedViewIndex >= 0 && selectedViewIndex < m_SingleViewWidgets.size())
  {
    niftkSingleViewerWidget* selectedView = m_SingleViewWidgets[selectedViewIndex];

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
    WindowLayout windowLayout = selectedView->GetLayout();

    if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      m_ControlPanel->SetLayout(windowLayout);
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
void niftkMultiViewerWidget::SetFocus()
{
  this->GetSelectedView()->setFocus();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnFocusChanged()
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
void niftkMultiViewerWidget::OnDropTypeChanged(DnDDisplayDropType dropType)
{
  m_VisibilityManager->ClearAllWindows();
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
  MIDASOrientation orientation = this->GetSelectedView()->GetOrientation();

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    niftkSingleViewerWidget* selectedView = this->GetSelectedView();
    selectedView->SetSliceIndex(orientation, sliceIndex);

    if (m_ControlPanel->AreViewPositionsBound())
    {
      foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
      {
        if (otherView != selectedView && otherView->isVisible())
        {
          otherView->SetSliceIndex(orientation, sliceIndex);
        }
      }
    }
  }
  else
  {
    MITK_WARN << "Found an invalid orientation in view widget " << this->GetSelectedViewIndex() << ", so ignoring request to change to slice " << sliceIndex << std::endl;
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

  niftkSingleViewerWidget* selectedView = this->GetSelectedView();
  selectedView->SetMagnification(magnification);

  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView && otherView->isVisible())
      {
        otherView->SetMagnification(magnification);
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

  niftkSingleViewerWidget* selectedView = this->GetSelectedView();
  selectedView->SetTimeStep(timeStep);

  if (dropType == DNDDISPLAY_DROP_ALL)
  {
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView && otherView->isVisible())
      {
        otherView->SetTimeStep(timeStep);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnLayoutChanged(WindowLayout windowLayout)
{
  if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    this->SetLayout(windowLayout);

    // Update the focus to the selected window, to trigger things like thumbnail viewer refresh
    // (or indeed anything that's listening to the FocusManager).
    this->UpdateFocusManagerToSelectedView();
  }

}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnLayoutChanged(niftkSingleViewerWidget* selectedView, WindowLayout windowLayout)
{
  m_ControlPanel->SetLayout(windowLayout);
  this->UpdateFocusManagerToSelectedView();

  if (m_ControlPanel->AreViewLayoutsBound())
  {
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView && otherView->isVisible())
      {
        otherView->SetLayout(windowLayout);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnGeometryChanged(niftkSingleViewerWidget* /*selectedView*/, mitk::TimeGeometry* geometry)
{
  if (m_ControlPanel->AreViewGeometriesBound())
  {
    foreach (niftkSingleViewerWidget* view, m_SingleViewWidgets)
    {
      view->SetBoundGeometry(geometry);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowCursorBindingChanged(bool bound)
{
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (m_SingleViewWidgets[i]->isVisible())
    {
      m_SingleViewWidgets[i]->SetCursorPositionsBound(bound);
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnWindowMagnificationBindingChanged(bool bound)
{
  for (int i = 0; i < m_SingleViewWidgets.size(); i++)
  {
    if (m_SingleViewWidgets[i]->isVisible())
    {
      m_SingleViewWidgets[i]->SetScaleFactorsBound(bound);
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
  this->SetShow3DWindowInOrthoView(visible);
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::UpdateFocusManagerToSelectedView()
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
  m_ControlPanel->SetLayout(windowLayout);

  niftkSingleViewerWidget* selectedView = this->GetSelectedView();
  selectedView->SetLayout(windowLayout);

  if (windowLayout == WINDOW_LAYOUT_AXIAL)
  {
    selectedView->SetSelectedRenderWindow(selectedView->GetAxialWindow());
  }
  else if (windowLayout == WINDOW_LAYOUT_SAGITTAL)
  {
    selectedView->SetSelectedRenderWindow(selectedView->GetSagittalWindow());
  }
  else if (windowLayout == WINDOW_LAYOUT_CORONAL)
  {
    selectedView->SetSelectedRenderWindow(selectedView->GetCoronalWindow());
  }

  if (m_ControlPanel->AreViewLayoutsBound())
  {
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView && otherView->isVisible())
      {
        otherView->SetLayout(windowLayout);
      }
    }
  }

  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    double magnification = selectedView->GetMagnification();
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView && otherView->isVisible())
      {
        otherView->SetMagnification(magnification);
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

  foreach (niftkSingleViewerWidget* view, m_SingleViewWidgets)
  {
    if (view->isVisible())
    {
      view->SetDisplay2DCursorsGlobally(globalVisibility);
      view->SetDisplay2DCursorsLocally(localVisibility);
    }
  }

  this->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::UpdateBoundMagnification()
{
  niftkSingleViewerWidget* selectedView = this->GetSelectedView();
  double magnification = selectedView->GetMagnification();
  foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
  {
    if (otherView != selectedView)
    {
      otherView->SetMagnification(magnification);
    }
  }
}


//-----------------------------------------------------------------------------
MIDASOrientation niftkMultiViewerWidget::GetOrientation() const
{
  MIDASOrientation orientation;

  switch (m_ControlPanel->GetLayout())
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
int niftkMultiViewerWidget::GetSelectedViewIndex() const
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
niftkSingleViewerWidget* niftkMultiViewerWidget::GetSelectedView() const
{
  int selectedViewIndex = m_SelectedViewIndex;
  if (selectedViewIndex < 0 || selectedViewIndex >= m_SingleViewWidgets.size())
  {
    // Default back to first view.
    selectedViewIndex = 0;
  }

  return m_SingleViewWidgets[selectedViewIndex];
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* niftkMultiViewerWidget::GetSelectedRenderWindow() const
{
  // NOTE: This MUST always return not-null.
  return this->GetSelectedView()->GetSelectedRenderWindow();
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

  niftkSingleViewerWidget* selectedView = this->GetSelectedView();

  renderWindows.insert("axial", selectedView->GetAxialWindow());
  renderWindows.insert("sagittal", selectedView->GetSagittalWindow());
  renderWindows.insert("coronal", selectedView->GetCoronalWindow());
  renderWindows.insert("3d", selectedView->Get3DWindow());

  int i = 0;
  foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
  {
    if (otherView != selectedView)
    {
      QString id = tr(".%1").arg(i);

      renderWindows.insert("axial" + id, otherView->GetAxialWindow());
      renderWindows.insert("sagittal" + id, otherView->GetSagittalWindow());
      renderWindows.insert("coronal" + id, otherView->GetCoronalWindow());
      renderWindows.insert("3d" + id, otherView->Get3DWindow());
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
    return this->GetSelectedView()->GetSelectedPosition();
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
void niftkMultiViewerWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition, const QString& id)
{
  if (id.isNull())
  {
    this->GetSelectedView()->SetSelectedPosition(selectedPosition);
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
  niftkSingleViewerWidget* selectedView = this->GetSelectedView();
  if (enabled && !m_NavigationControllerEventListening)
  {
    selectedView->SetNavigationControllerEventListening(true);
  }
  else if (!enabled && m_NavigationControllerEventListening)
  {
    selectedView->SetNavigationControllerEventListening(false);
  }
  m_NavigationControllerEventListening = enabled;
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::SetSelectedViewIndex(int selectedViewIndex)
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
void niftkMultiViewerWidget::OnViewPositionBindingChanged()
{
  niftkSingleViewerWidget* selectedView = this->GetSelectedView();

  if (m_ControlPanel->AreViewPositionsBound())
  {
    mitk::Point3D selectedPosition = selectedView->GetSelectedPosition();
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView)
      {
        otherView->SetSelectedPosition(selectedPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewCursorBindingChanged()
{
  niftkSingleViewerWidget* selectedView = this->GetSelectedView();

  if (m_ControlPanel->AreViewCursorsBound())
  {
    mitk::Vector3D cursorPosition = selectedView->GetCursorPosition();
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView)
      {
        otherView->SetCursorPosition(cursorPosition);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewLayoutBindingChanged()
{
  niftkSingleViewerWidget* selectedView = this->GetSelectedView();

  if (m_ControlPanel->AreViewLayoutsBound())
  {
    WindowLayout windowLayout = selectedView->GetLayout();
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView)
      {
        otherView->SetLayout(windowLayout);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewMagnificationBindingChanged()
{
  niftkSingleViewerWidget* selectedView = this->GetSelectedView();

  if (m_ControlPanel->AreViewMagnificationsBound())
  {
    double magnification = selectedView->GetMagnification();
    foreach (niftkSingleViewerWidget* otherView, m_SingleViewWidgets)
    {
      if (otherView != selectedView)
      {
        otherView->SetMagnification(magnification);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnViewGeometryBindingChanged()
{
  if (m_ControlPanel->AreViewGeometriesBound())
  {
    niftkSingleViewerWidget* selectedView = this->GetSelectedView();
    mitk::TimeGeometry* geometry = selectedView->GetGeometry();

    foreach (niftkSingleViewerWidget* view, m_SingleViewWidgets)
    {
      view->SetBoundGeometry(geometry);
      view->SetBoundGeometryActive(true);
    }
  }
  else
  {
    foreach (niftkSingleViewerWidget* view, m_SingleViewWidgets)
    {
      view->SetBoundGeometryActive(false);
    }
  }
//  this->Update2DCursorVisibility();
}


//-----------------------------------------------------------------------------
void niftkMultiViewerWidget::OnPopupOpened(bool opened)
{
  if (!opened)
  {
    this->GetSelectedView()->repaint();
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
