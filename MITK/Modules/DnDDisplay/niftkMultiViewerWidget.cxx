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

#include <mitkBaseGeometry.h>
#include <QmitkRenderWindow.h>

#include "niftkSingleViewerWidget.h"
#include "niftkMultiViewerControls.h"


namespace niftk
{

//-----------------------------------------------------------------------------
MultiViewerWidget::MultiViewerWidget(
    mitk::RenderingManager* renderingManager,
    QWidget* parent, Qt::WindowFlags f)
: QWidget(parent, f),
  m_TopLevelLayout(nullptr),
  m_LayoutForRenderWindows(nullptr),
  m_PinButton(nullptr),
  m_PopupWidget(nullptr),
  m_DisplayConvention(DISPLAY_CONVENTION_RADIO),
  m_VisibilityManager(MultiViewerVisibilityManager::New(renderingManager->GetDataStorage())),
  m_RenderingManager(renderingManager),
  m_SelectedViewerIndex(0),
  m_ViewerRows(0),
  m_ViewerColumns(0),
  m_CursorDefaultVisibility(true),
  m_RememberSettingsPerWindowLayout(false),
  m_ThumbnailMode(false),
  m_LinkedNavigationEnabled(false),
  m_Magnification(0.0),
  m_SingleWindowLayout(WINDOW_LAYOUT_CORONAL),
  m_MultiWindowLayout(WINDOW_LAYOUT_ORTHO_NO_3D),
  m_BindingOptions(0),
  m_ControlPanel(nullptr)
{
  this->setFocusPolicy(Qt::StrongFocus);

  /************************************
   * Create stuff.
   ************************************/

  m_TopLevelLayout = new QGridLayout(this);
  m_TopLevelLayout->setObjectName(QString::fromUtf8("MultiViewerWidget::m_TopLevelLayout"));
  m_TopLevelLayout->setContentsMargins(0, 0, 0, 0);
  m_TopLevelLayout->setSpacing(0);

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("MultiViewerWidget::m_LayoutForRenderWindows"));
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
  m_ControlPanel->SetPositionAnnotationVisible(true);
  m_ControlPanel->SetIntensityAnnotationVisible(true);
  m_ControlPanel->SetPropertyAnnotationVisible(true);

  // Default to dropping into single window.
  m_ControlPanel->SetDropType(DNDDISPLAY_DROP_SINGLE);
  m_VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);

  this->SetViewerNumber(1, 1);

  // Connect Qt Signals to make it all hang together.
  this->connect(m_ControlPanel, SIGNAL(SelectedSliceChanged(int)), SLOT(OnSelectedSliceControlChanged(int)));
  this->connect(m_ControlPanel, SIGNAL(TimeStepChanged(int)), SLOT(OnTimeStepControlChanged(int)));
  this->connect(m_ControlPanel, SIGNAL(MagnificationChanged(double)), SLOT(OnMagnificationControlChanged(double)));

  this->connect(m_ControlPanel, SIGNAL(ShowCursorChanged(bool)), SLOT(OnCursorVisibilityControlChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ShowDirectionAnnotationsChanged(bool)), SLOT(OnShowDirectionAnnotationsControlsChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ShowPositionAnnotationChanged(bool)), SLOT(OnShowPositionAnnotationControlsChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ShowIntensityAnnotationChanged(bool)), SLOT(OnShowIntensityAnnotationControlsChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ShowPropertyAnnotationChanged(bool)), SLOT(OnShowPropertyAnnotationControlsChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(PropertiesForAnnotationChanged()), SLOT(OnPropertiesForAnnotationControlsChanged()));

  this->connect(m_ControlPanel, SIGNAL(WindowLayoutChanged(WindowLayout)), SLOT(OnWindowLayoutControlChanged(WindowLayout)));
  this->connect(m_ControlPanel, SIGNAL(WindowCursorBindingChanged(bool)), SLOT(OnWindowCursorBindingControlChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(WindowMagnificationBindingChanged(bool)), SLOT(OnWindowMagnificationBindingControlChanged(bool)));

  this->connect(m_ControlPanel, SIGNAL(ViewerNumberChanged(int, int)), SLOT(OnViewerNumberControlChanged(int, int)));

  this->connect(m_ControlPanel, SIGNAL(ViewerPositionBindingChanged(bool)), SLOT(OnViewerPositionBindingControlChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerCursorBindingChanged(bool)), SLOT(OnViewerCursorBindingControlChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerMagnificationBindingChanged(bool)), SLOT(OnViewerMagnificationBindingControlChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerVisibilityBindingChanged(bool)), SLOT(OnViewerVisibilityBindingControlChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerWindowLayoutBindingChanged(bool)), SLOT(OnViewerWindowLayoutBindingControlChanged(bool)));
  this->connect(m_ControlPanel, SIGNAL(ViewerGeometryBindingChanged(bool)), SLOT(OnViewerGeometryBindingControlChanged(bool)));

  this->connect(m_ControlPanel, SIGNAL(DropTypeChanged(DnDDisplayDropType)), SLOT(OnDropTypeControlChanged(DnDDisplayDropType)));
  this->connect(m_ControlPanel, SIGNAL(DropAccumulateChanged(bool)), SLOT(OnDropAccumulateControlChanged(bool)));

  this->connect(m_PopupWidget, SIGNAL(popupOpened(bool)), SLOT(OnPopupOpened(bool)));

  this->connect(m_VisibilityManager, SIGNAL(VisibilityBindingChanged(bool)), SLOT(OnVisibilityBindingChanged(bool)));
}


//-----------------------------------------------------------------------------
MultiViewerControls* MultiViewerWidget::CreateControlPanel(QWidget* parent)
{
  MultiViewerControls* controlPanel = new MultiViewerControls(parent);

  controlPanel->SetMaxViewerNumber(m_MaxViewerRows, m_MaxViewerColumns);

  controlPanel->SetWindowCursorsBound(true);
  controlPanel->SetWindowMagnificationsBound(true);

  return controlPanel;
}


//-----------------------------------------------------------------------------
MultiViewerWidget::~MultiViewerWidget()
{
  m_VisibilityManager = nullptr;

  this->EnableLinkedNavigation(false);
}


//-----------------------------------------------------------------------------
SingleViewerWidget* MultiViewerWidget::CreateViewer(const QString& name)
{
  SingleViewerWidget* viewer = new SingleViewerWidget(this, m_RenderingManager, name);
  viewer->setObjectName(name);
  viewer->setVisible(false);

  viewer->SetBackgroundColour(m_BackgroundColour);
  viewer->SetRememberSettingsPerWindowLayout(m_RememberSettingsPerWindowLayout);
  viewer->SetDisplayInteractionsEnabled(true);
  viewer->SetLinkedNavigationEnabled(m_LinkedNavigationEnabled);
  viewer->SetCursorPositionBinding(true);
  viewer->SetScaleFactorBinding(true);
  viewer->SetDefaultSingleWindowLayout(m_SingleWindowLayout);
  viewer->SetDefaultMultiWindowLayout(m_MultiWindowLayout);

  this->connect(viewer, SIGNAL(NodesDropped(const std::vector<mitk::DataNode*>&)), SLOT(OnNodesDropped(const std::vector<mitk::DataNode*>&)));
  this->connect(viewer, SIGNAL(WindowSelected()), SLOT(OnWindowSelected()));
  this->connect(viewer, SIGNAL(SelectPreviousViewer()), SLOT(OnSelectPreviousViewer()));
  this->connect(viewer, SIGNAL(SelectNextViewer()), SLOT(OnSelectNextViewer()));
  this->connect(viewer, SIGNAL(SelectViewer(int)), SLOT(OnSelectViewer(int)));
  this->connect(viewer, SIGNAL(TimeGeometryChanged(const mitk::TimeGeometry*)), SLOT(OnTimeGeometryChanged(const mitk::TimeGeometry*)));
  this->connect(viewer, SIGNAL(SelectedPositionChanged(const mitk::Point3D&)), SLOT(OnSelectedPositionChanged(const mitk::Point3D&)));
  this->connect(viewer, SIGNAL(TimeStepChanged(int)), SLOT(OnTimeStepChanged(int)));
  this->connect(viewer, SIGNAL(CursorPositionChanged(WindowOrientation, const mitk::Vector2D&)), SLOT(OnCursorPositionChanged(WindowOrientation, const mitk::Vector2D&)));
  this->connect(viewer, SIGNAL(ScaleFactorChanged(WindowOrientation, double)), SLOT(OnScaleFactorChanged(WindowOrientation, double)));
  this->connect(viewer, SIGNAL(WindowLayoutChanged(WindowLayout)), SLOT(OnWindowLayoutChanged(WindowLayout)));
  this->connect(viewer, SIGNAL(CursorPositionBindingChanged(bool)), SLOT(OnCursorPositionBindingChanged(bool)));
  this->connect(viewer, SIGNAL(ScaleFactorBindingChanged(bool)), SLOT(OnScaleFactorBindingChanged(bool)));
  this->connect(viewer, SIGNAL(CursorVisibilityChanged(bool)), SLOT(OnCursorVisibilityChanged(bool)));
  this->connect(viewer, SIGNAL(DirectionAnnotationsVisibilityChanged(bool)), SLOT(OnDirectionAnnotationsVisibilityChanged(bool)));
  this->connect(viewer, SIGNAL(PositionAnnotationVisibilityChanged(bool)), SLOT(OnPositionAnnotationVisibilityChanged(bool)));
  this->connect(viewer, SIGNAL(IntensityAnnotationVisibilityChanged(bool)), SLOT(OnIntensityAnnotationVisibilityChanged(bool)));
  this->connect(viewer, SIGNAL(PropertyAnnotationVisibilityChanged(bool)), SLOT(OnPropertyAnnotationVisibilityChanged(bool)));

  return viewer;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::RequestUpdateAll()
{
  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    if (viewer->isVisible())
    {
      viewer->RequestUpdate();
    }
  }
}


//-----------------------------------------------------------------------------
int MultiViewerWidget::GetDisplayConvention() const
{
  return m_DisplayConvention;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetDisplayConvention(int displayConvention)
{
  if (displayConvention != m_DisplayConvention)
  {
    m_DisplayConvention = displayConvention;
    for (SingleViewerWidget* viewer: m_Viewers)
    {
      viewer->SetDisplayConvention(displayConvention);
    }
  }
}


//-----------------------------------------------------------------------------
int MultiViewerWidget::GetBindingOptions() const
{
  return m_BindingOptions;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetBindingOptions(int bindingOptions)
{
  if (bindingOptions & CursorBinding)
  {
    bindingOptions |= PositionBinding;
    bindingOptions |= WindowLayoutBinding;
  }

  if (bindingOptions == m_BindingOptions)
  {
    return;
  }

  bool oldPositionBinding = m_BindingOptions & PositionBinding;
  bool newPositionBinding = bindingOptions & PositionBinding;
  if (oldPositionBinding != newPositionBinding)
  {
    if (newPositionBinding)
    {
      SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
      const mitk::Point3D& selectedPosition = selectedViewer->GetSelectedPosition();
      foreach (SingleViewerWidget* otherViewer, m_Viewers)
      {
        if (otherViewer != selectedViewer)
        {
          bool signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetSelectedPosition(selectedPosition);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }

    bool signalsWereBlocked = m_ControlPanel->blockSignals(true);
    m_ControlPanel->SetViewerPositionsBound(newPositionBinding);
    m_ControlPanel->blockSignals(signalsWereBlocked);
  }

  bool oldCursorBinding = m_BindingOptions & CursorBinding;
  bool newCursorBinding = bindingOptions & CursorBinding;
  if (oldCursorBinding != newCursorBinding)
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    bool windowCursorPositionsBound = selectedViewer->GetCursorPositionBinding();

    if (newCursorBinding)
    {
      WindowOrientation orientation = selectedViewer->GetOrientation();
      mitk::Vector2D cursorPosition = selectedViewer->GetCursorPosition(orientation);
      const std::vector<mitk::Vector2D>& cursorPositions = selectedViewer->GetCursorPositions();

      foreach (SingleViewerWidget* otherViewer, m_Viewers)
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

    bool signalsWereBlocked = m_ControlPanel->blockSignals(true);
    m_ControlPanel->SetViewerCursorsBound(newCursorBinding);
    m_ControlPanel->blockSignals(signalsWereBlocked);
  }

  bool oldMagnificationBinding = m_BindingOptions & MagnificationBinding;
  bool newMagnificationBinding = bindingOptions & MagnificationBinding;
  if (oldMagnificationBinding != newMagnificationBinding)
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    bool windowScaleFactorsBound = selectedViewer->GetScaleFactorBinding();

    if (newMagnificationBinding)
    {
      WindowOrientation orientation = selectedViewer->GetOrientation();
      double scaleFactor = selectedViewer->GetScaleFactor(orientation);
      std::vector<double> scaleFactors = selectedViewer->GetScaleFactors();

      foreach (SingleViewerWidget* otherViewer, m_Viewers)
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

    bool signalsWereBlocked = m_ControlPanel->blockSignals(true);
    m_ControlPanel->SetViewerMagnificationsBound(newMagnificationBinding);
    m_ControlPanel->blockSignals(signalsWereBlocked);
  }

  bool oldVisibilityBinding = m_BindingOptions & VisibilityBinding;
  bool newVisibilityBinding = bindingOptions & VisibilityBinding;
  if (oldVisibilityBinding != newVisibilityBinding)
  {
    bool signalsWereBlocked = m_VisibilityManager->blockSignals(true);
    m_VisibilityManager->SetVisibilityBinding(newVisibilityBinding);
    m_VisibilityManager->blockSignals(signalsWereBlocked);

    signalsWereBlocked = m_ControlPanel->blockSignals(true);
    m_ControlPanel->SetViewerVisibilitiesBound(newVisibilityBinding);
    m_ControlPanel->blockSignals(signalsWereBlocked);
  }

  bool oldWindowLayoutBinding = m_BindingOptions & WindowLayoutBinding;
  bool newWindowLayoutBinding = bindingOptions & WindowLayoutBinding;
  if (oldWindowLayoutBinding != newWindowLayoutBinding)
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();

    if (newWindowLayoutBinding)
    {
      WindowLayout windowLayout = selectedViewer->GetWindowLayout();
      foreach (SingleViewerWidget* otherViewer, m_Viewers)
      {
        if (otherViewer != selectedViewer)
        {
          bool signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetWindowLayout(windowLayout);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }

    bool signalsWereBlocked = m_ControlPanel->blockSignals(true);
    m_ControlPanel->SetViewerWindowLayoutsBound(newWindowLayoutBinding);
    m_ControlPanel->blockSignals(signalsWereBlocked);
  }

  bool oldGeometryBinding = m_BindingOptions & GeometryBinding;
  bool newGeometryBinding = bindingOptions & GeometryBinding;
  if (oldGeometryBinding != newGeometryBinding)
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();

    if (newGeometryBinding)
    {
      const mitk::TimeGeometry* timeGeometry = selectedViewer->GetTimeGeometry();
      if (timeGeometry)
      {
        foreach (SingleViewerWidget* viewer, m_Viewers)
        {
          bool signalsWereBlocked = viewer->blockSignals(true);
          viewer->SetBoundTimeGeometry(timeGeometry);
          viewer->SetBoundTimeGeometryActive(true);
          viewer->blockSignals(signalsWereBlocked);
        }
      }
    }
    else
    {
      foreach (SingleViewerWidget* viewer, m_Viewers)
      {
        bool signalsWereBlocked = viewer->blockSignals(true);
        viewer->SetBoundTimeGeometryActive(false);
        viewer->blockSignals(signalsWereBlocked);
      }
    }

    this->OnCursorVisibilityChanged(selectedViewer->IsCursorVisible());


    bool signalsWereBlocked = m_ControlPanel->blockSignals(true);
    m_ControlPanel->SetViewerGeometriesBound(newGeometryBinding);
    m_ControlPanel->blockSignals(signalsWereBlocked);
  }

  m_BindingOptions = bindingOptions;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetInterpolationType(DnDDisplayInterpolationType interpolationType)
{
  m_VisibilityManager->SetInterpolationType(interpolationType);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetDefaultWindowLayout(WindowLayout windowLayout)
{
  m_VisibilityManager->SetDefaultWindowLayout(windowLayout);
  if (niftk::IsSingleWindowLayout(windowLayout))
  {
    this->SetDefaultSingleWindowLayout(windowLayout);
  }
  else
  {
    this->SetDefaultMultiWindowLayout(windowLayout);
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetDefaultSingleWindowLayout(WindowLayout windowLayout)
{
  m_SingleWindowLayout = windowLayout;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetDefaultMultiWindowLayout(WindowLayout windowLayout)
{
  m_MultiWindowLayout = windowLayout;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetShowMagnificationSlider(bool visible)
{
  m_ControlPanel->SetMagnificationControlsVisible(visible);
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::AreShowOptionsVisible() const
{
  return m_ControlPanel->AreShowOptionsVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetShowOptionsVisible(bool visible)
{
  m_ControlPanel->SetShowOptionsVisible(visible);
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::AreWindowLayoutControlsVisible() const
{
  return m_ControlPanel->AreWindowLayoutControlsVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetWindowLayoutControlsVisible(bool visible)
{
  m_ControlPanel->SetWindowLayoutControlsVisible(visible);
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::AreViewerNumberControlsVisible() const
{
  return m_ControlPanel->AreViewerNumberControlsVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetViewerNumberControlsVisible(bool visible)
{
  m_ControlPanel->SetViewerNumberControlsVisible(visible);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetShowDropTypeControls(bool visible)
{
  m_ControlPanel->SetDropTypeControlsVisible(visible);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetDropType(DnDDisplayDropType dropType)
{
  if (dropType != m_ControlPanel->GetDropType())
  {
    m_ControlPanel->SetDropType(dropType);

    m_VisibilityManager->ClearViewers();
    m_VisibilityManager->SetDropType(dropType);
    this->SetThumbnailMode(dropType == DNDDISPLAY_DROP_ALL);
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::IsCursorVisible() const
{
  return m_ControlPanel->IsCursorVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetCursorVisible(bool visible)
{
  m_ControlPanel->SetCursorVisible(visible);

  SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  selectedViewer->SetCursorVisible(visible);

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != selectedViewer && otherViewer->isVisible())
      {
        otherViewer->SetCursorVisible(visible);
      }
    }
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::GetCursorDefaultVisibility() const
{
  return m_CursorDefaultVisibility;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetCursorDefaultVisibility(bool visible)
{
  m_CursorDefaultVisibility = visible;
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::AreDirectionAnnotationsVisible() const
{
  return m_ControlPanel->AreDirectionAnnotationsVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetDirectionAnnotationsVisible(bool visible)
{
  m_ControlPanel->SetDirectionAnnotationsVisible(visible);
  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetDirectionAnnotationsVisible(visible);
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::IsPositionAnnotationVisible() const
{
  return m_ControlPanel->IsPositionAnnotationVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetPositionAnnotationVisible(bool visible)
{
  m_ControlPanel->SetPositionAnnotationVisible(visible);
  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetPositionAnnotationVisible(visible);
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::IsIntensityAnnotationVisible() const
{
  return m_ControlPanel->IsIntensityAnnotationVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetIntensityAnnotationVisible(bool visible)
{
  m_ControlPanel->SetIntensityAnnotationVisible(visible);
  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetIntensityAnnotationVisible(visible);
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::IsPropertyAnnotationVisible() const
{
  return m_ControlPanel->IsPropertyAnnotationVisible();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetPropertyAnnotationVisible(bool visible)
{
  m_ControlPanel->SetPropertyAnnotationVisible(visible);
  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetPropertyAnnotationVisible(visible);
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetPropertiesForAnnotation(const QStringList& propertiesForAnnotation)
{
  m_ControlPanel->SetPropertiesForAnnotation(propertiesForAnnotation);
  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetPropertiesForAnnotation(propertiesForAnnotation);
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetRememberSettingsPerWindowLayout(bool rememberSettingsPerWindowLayout)
{
  m_RememberSettingsPerWindowLayout = rememberSettingsPerWindowLayout;
  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerWindowLayout);
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::GetThumbnailMode() const
{
  return m_ThumbnailMode;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetThumbnailMode(bool thumbnailMode)
{
  if (thumbnailMode != m_ThumbnailMode)
  {
    m_ThumbnailMode = thumbnailMode;

    if (thumbnailMode)
    {
      // We need to remember the "previous" number of rows and columns, so when we switch out
      // of thumbnail mode, we know how many rows and columns to revert to.
      m_ViewerRowsInNonThumbnailMode = m_ViewerRows;
      m_ViewerColumnsInNonThumbnailMode = m_ViewerColumns;
      m_ControlPanel->SetSingleViewerControlsEnabled(false);
      m_ControlPanel->SetViewerNumber(m_MaxViewerRows, m_MaxViewerColumns);
      m_ControlPanel->SetMultiViewerControlsEnabled(false);
      this->SetViewerNumber(m_MaxViewerRows, m_MaxViewerColumns);
    }
    else
    {
      // otherwise we remember the "next" (the number we are being asked for in this method call) number of rows and columns.
      m_ControlPanel->SetSingleViewerControlsEnabled(m_LinkedNavigationEnabled);
      m_ControlPanel->SetMultiViewerControlsEnabled(true);
      m_ControlPanel->SetViewerNumber(m_ViewerRowsInNonThumbnailMode, m_ViewerColumnsInNonThumbnailMode);
      this->SetViewerNumber(m_ViewerRowsInNonThumbnailMode, m_ViewerColumnsInNonThumbnailMode);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetBackgroundColour(QColor backgroundColour)
{
  m_BackgroundColour = backgroundColour;

  foreach (SingleViewerWidget* viewer, m_Viewers)
  {
    viewer->SetBackgroundColour(m_BackgroundColour);
  }
}


//-----------------------------------------------------------------------------
int MultiViewerWidget::GetNumberOfRows() const
{
  return m_ViewerRows;
}


//-----------------------------------------------------------------------------
int MultiViewerWidget::GetNumberOfColumns() const
{
  return m_ViewerColumns;
}


//-----------------------------------------------------------------------------
SingleViewerWidget* MultiViewerWidget::GetViewer(int row, int column) const
{
  SingleViewerWidget* viewer = nullptr;
  if (row >= 0 && row < m_ViewerRows && column >= 0 && column < m_ViewerColumns)
  {
    viewer = m_Viewers[row * m_ViewerColumns + column];
  }
  return viewer;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetViewerNumber(int viewerRows, int viewerColumns)
{
  // If we have the right number of viewers, there is nothing to do, so early exit.
  if (viewerRows == m_ViewerRows && viewerColumns == m_ViewerColumns)
  {
    return;
  }

  // Work out required number of viewers, and hence if we need to create any new ones.
  int requiredNumberOfViewers = viewerRows * viewerColumns;
  int currentNumberOfViewers = m_Viewers.size();

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

  // Make all current viewers invisible, as we are going to destroy layout.
  foreach (SingleViewerWidget* viewer, m_Viewers)
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
      SingleViewerWidget* viewer = this->CreateViewer(QString("DnD-Viewer-%1").arg(currentNumberOfViewers + i));

      m_Viewers.push_back(viewer);
      m_VisibilityManager->RegisterViewer(viewer);
    }
  }
  else if (requiredNumberOfViewers < currentNumberOfViewers)
  {
    // destroy surplus viewers
    m_VisibilityManager->DeregisterViewers(requiredNumberOfViewers, m_Viewers.size());

    for (int i = requiredNumberOfViewers; i < m_Viewers.size(); i++)
    {
      delete m_Viewers[i];
    }

    m_Viewers.erase(m_Viewers.begin() + requiredNumberOfViewers, m_Viewers.end());
  }

  // Put all viewers in the grid.
  // Prior experience suggests we always need a new grid,
  // because otherwise viewers don't appear to remove properly.

  m_TopLevelLayout->removeItem(m_LayoutForRenderWindows);
  delete m_LayoutForRenderWindows;

  m_LayoutForRenderWindows = new QGridLayout();
  m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("MultiViewerWidget::m_LayoutForRenderWindows"));
  m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
  m_LayoutForRenderWindows->setVerticalSpacing(0);
  m_LayoutForRenderWindows->setHorizontalSpacing(0);

  m_TopLevelLayout->addLayout(m_LayoutForRenderWindows, 1, 0);

  int viewerIndex = 0;
  for (int row = 0; row < viewerRows; ++row)
  {
    for (int column = 0; column < viewerColumns; ++column)
    {
      m_LayoutForRenderWindows->addWidget(m_Viewers[viewerIndex], row, column);
      m_Viewers[viewerIndex]->show();
      m_Viewers[viewerIndex]->setEnabled(true);
      ++viewerIndex;
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

  if (m_ViewerColumns != 0)
  {
    int oldSelectedRow = m_SelectedViewerIndex / m_ViewerColumns;
    int oldSelectedColumn = m_SelectedViewerIndex % m_ViewerColumns;

    if (oldSelectedRow < viewerRows
        && oldSelectedColumn < viewerColumns)
    {
      m_SelectedViewerIndex = oldSelectedRow * viewerColumns + oldSelectedColumn;
    }
    else
    {
      m_SelectedViewerIndex = 0;
    }
  }
  else
  {
    m_SelectedViewerIndex = 0;
  }

  m_ViewerRows = viewerRows;
  m_ViewerColumns = viewerColumns;

  SingleViewerWidget* selectedViewer = m_Viewers[m_SelectedViewerIndex];

  // Update row/column of viewers without triggering another layout size change.
  m_ControlPanel->SetViewerNumber(viewerRows, viewerColumns);

  // Now the number of viewers has changed, we need to make sure they are all in synch with all the right properties.
  this->OnCursorVisibilityChanged(selectedViewer->IsCursorVisible());
  this->OnDirectionAnnotationsVisibilityChanged(selectedViewer->AreDirectionAnnotationsVisible());
  this->OnPositionAnnotationVisibilityChanged(selectedViewer->IsPositionAnnotationVisible());
  this->OnIntensityAnnotationVisibilityChanged(selectedViewer->IsIntensityAnnotationVisible());
  this->OnPropertyAnnotationVisibilityChanged(selectedViewer->IsPropertyAnnotationVisible());

  if (m_ControlPanel->AreViewerGeometriesBound())
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    const mitk::TimeGeometry* timeGeometry = selectedViewer->GetTimeGeometry();

    if (timeGeometry)
    {
      foreach (SingleViewerWidget* otherViewer, m_Viewers)
      {
        if (otherViewer != selectedViewer)
        {
          bool signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetBoundTimeGeometry(timeGeometry);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }
  }

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    WindowOrientation orientation = selectedViewer->GetOrientation();
    double magnification = selectedViewer->GetMagnification(orientation);
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnViewerNumberControlChanged(int rows, int columns)
{
  this->SetViewerNumber(rows, columns);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnSelectedPositionChanged(const mitk::Point3D& selectedPosition)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());

  // If the viewer is not found, we do not do anything.
  if (std::find(m_Viewers.begin(), m_Viewers.end(), viewer) == m_Viewers.end())
  {
    return;
  }

  WindowOrientation orientation = viewer->GetOrientation();
  if (orientation != WINDOW_ORIENTATION_UNKNOWN)
  {
    m_ControlPanel->SetSelectedSlice(viewer->GetSelectedSlice(orientation));
  }

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnTimeStepChanged(int timeStep)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());

  // If the viewer is not found, we do not do anything.
  if (std::find(m_Viewers.begin(), m_Viewers.end(), viewer) == m_Viewers.end())
  {
    return;
  }

  m_ControlPanel->SetTimeStep(timeStep);

  if (m_ControlPanel->AreViewerPositionsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnCursorPositionChanged(WindowOrientation orientation, const mitk::Vector2D& cursorPosition)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnScaleFactorChanged(WindowOrientation orientation, double scaleFactor)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());

  double magnification = viewer->GetMagnification(orientation);
  m_ControlPanel->SetMagnification(magnification);

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnCursorPositionBindingChanged(bool bound)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnScaleFactorBindingChanged(bool bound)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnVisibilityBindingChanged(bool bound)
{
  m_ControlPanel->SetViewerVisibilitiesBound(bound);
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::IsFocused()
{
  return this->GetSelectedViewer()->IsFocused();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetFocused()
{
  this->GetSelectedViewer()->SetFocused();
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetSelectedViewer(int viewerIndex)
{
  if (viewerIndex >= 0 && viewerIndex < m_Viewers.size())
  {
    m_Viewers[viewerIndex]->SetFocused();
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnNodesDropped(const std::vector<mitk::DataNode*>& droppedNodes)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(QObject::sender());
  emit NodesDropped(viewer, droppedNodes);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnWindowSelected()
{
  SingleViewerWidget* selectedViewer = qobject_cast<SingleViewerWidget*>(QObject::sender());
  assert(selectedViewer);

  auto it = std::find(m_Viewers.begin(), m_Viewers.end(), selectedViewer);
  assert(it != m_Viewers.end());

  int selectedViewerIndex = it - m_Viewers.begin();

  m_SelectedViewerIndex = selectedViewerIndex;

  m_ControlPanel->SetWindowLayout(selectedViewer->GetWindowLayout());

  int maxTimeStep = selectedViewer->GetMaxTimeStep();
  int timeStep = selectedViewer->GetTimeStep();
  m_ControlPanel->SetMaxTimeStep(maxTimeStep);
  m_ControlPanel->SetTimeStep(timeStep);

  WindowOrientation orientation = selectedViewer->GetOrientation();
  if (orientation != WINDOW_ORIENTATION_UNKNOWN)
  {
    int maxSlice = selectedViewer->GetMaxSlice(orientation);
    int selectedSlice = selectedViewer->GetSelectedSlice(orientation);
    m_ControlPanel->SetMaxSlice(maxSlice);
    m_ControlPanel->SetSelectedSlice(selectedSlice);

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

  m_ControlPanel->SetWindowCursorsBound(selectedViewer->GetCursorPositionBinding());
  m_ControlPanel->SetWindowMagnificationsBound(selectedViewer->GetScaleFactorBinding());

  this->OnCursorVisibilityChanged(selectedViewer->IsCursorVisible());

  emit WindowSelected(selectedViewer);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnSelectPreviousViewer()
{
  this->SetSelectedViewer(m_SelectedViewerIndex - 1);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnSelectNextViewer()
{
  this->SetSelectedViewer(m_SelectedViewerIndex + 1);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnSelectViewer(int viewerIndex)
{
  this->SetSelectedViewer(viewerIndex);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnDropTypeControlChanged(DnDDisplayDropType dropType)
{
  m_VisibilityManager->ClearViewers();
  m_VisibilityManager->SetDropType(dropType);
  this->SetThumbnailMode(dropType == DNDDISPLAY_DROP_ALL);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnDropAccumulateControlChanged(bool checked)
{
  m_VisibilityManager->SetAccumulateWhenDropping(checked);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnSelectedSliceControlChanged(int selectedSlice)
{
  WindowOrientation orientation = this->GetSelectedViewer()->GetOrientation();

  if (orientation != WINDOW_ORIENTATION_UNKNOWN)
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    bool signalsWereBlocked = selectedViewer->blockSignals(true);
    selectedViewer->SetSelectedSlice(orientation, selectedSlice);
    selectedViewer->blockSignals(signalsWereBlocked);

    if (m_ControlPanel->AreViewerPositionsBound())
    {
      foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
    MITK_WARN << "Found an invalid orientation in viewer " << m_SelectedViewerIndex << ", so ignoring request to change to slice " << selectedSlice << std::endl;
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnMagnificationControlChanged(double magnification)
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

  SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  WindowOrientation orientation = selectedViewer->GetOrientation();

  bool signalsWereBlocked = selectedViewer->blockSignals(true);
  selectedViewer->SetMagnification(orientation, magnification);
  selectedViewer->blockSignals(signalsWereBlocked);

  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnTimeStepControlChanged(int timeStep)
{
  this->SetTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetTimeStep(int timeStep)
{
  DnDDisplayDropType dropType = m_ControlPanel->GetDropType();

  SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  bool signalsWereBlocked = selectedViewer->blockSignals(true);
  selectedViewer->SetTimeStep(timeStep);
  selectedViewer->blockSignals(signalsWereBlocked);

  if (dropType == DNDDISPLAY_DROP_ALL)
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnWindowLayoutControlChanged(WindowLayout windowLayout)
{
  if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
    bool signalsWereBlocked = selectedViewer->blockSignals(true);
    selectedViewer->SetWindowLayout(windowLayout);
    selectedViewer->blockSignals(signalsWereBlocked);

    m_ControlPanel->SetWindowLayout(windowLayout);
    m_ControlPanel->SetWindowCursorsBound(selectedViewer->GetCursorPositionBinding());
    m_ControlPanel->SetWindowMagnificationsBound(selectedViewer->GetScaleFactorBinding());

    if (m_ControlPanel->AreViewerWindowLayoutsBound())
    {
      foreach (SingleViewerWidget* otherViewer, m_Viewers)
      {
        if (otherViewer != selectedViewer && otherViewer->isVisible())
        {
          signalsWereBlocked = otherViewer->blockSignals(true);
          otherViewer->SetWindowLayout(windowLayout);
          otherViewer->blockSignals(signalsWereBlocked);
        }
      }
    }

    WindowOrientation orientation = selectedViewer->GetOrientation();

    if (m_ControlPanel->AreViewerCursorsBound())
    {
      mitk::Vector2D cursorPosition = selectedViewer->GetCursorPosition(orientation);
      const std::vector<mitk::Vector2D>& cursorPositions = selectedViewer->GetCursorPositions();

      foreach (SingleViewerWidget* otherViewer, m_Viewers)
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

      foreach (SingleViewerWidget* otherViewer, m_Viewers)
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

    if (niftk::IsSingleWindowLayout(windowLayout))
    {
      m_SingleWindowLayout = windowLayout;
    }
    else
    {
      m_MultiWindowLayout = windowLayout;
    }
  }

}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnWindowLayoutChanged(WindowLayout windowLayout)
{
  SingleViewerWidget* selectedViewer = qobject_cast<SingleViewerWidget*>(this->sender());

  m_ControlPanel->SetWindowLayout(windowLayout);
  m_ControlPanel->SetWindowCursorsBound(selectedViewer->GetCursorPositionBinding());
  m_ControlPanel->SetWindowMagnificationsBound(selectedViewer->GetScaleFactorBinding());

  foreach (SingleViewerWidget* otherViewer, m_Viewers)
  {
    if (otherViewer != selectedViewer && otherViewer->isVisible())
    {
      bool signalsWereBlocked = otherViewer->blockSignals(true);
      bool updateWasBlocked = otherViewer->BlockUpdate(true);
      if (m_ControlPanel->AreViewerWindowLayoutsBound())
      {
        otherViewer->SetWindowLayout(windowLayout);
      }
      if (m_ControlPanel->AreViewerPositionsBound())
      {
        otherViewer->SetSelectedPosition(selectedViewer->GetSelectedPosition());
      }
      if (m_ControlPanel->AreViewerCursorsBound())
      {
        otherViewer->SetCursorPositions(selectedViewer->GetCursorPositions());
      }
      if (m_ControlPanel->AreViewerMagnificationsBound())
      {
        otherViewer->SetScaleFactors(selectedViewer->GetScaleFactors());
      }
      otherViewer->BlockUpdate(updateWasBlocked);
      otherViewer->blockSignals(signalsWereBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnCursorVisibilityChanged(bool visible)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());
  if (!viewer)
  {
    /// Note: this slot is also directly invoked from this class. In this case sender() returns 0.
    viewer = this->GetSelectedViewer();
  }

  m_ControlPanel->SetCursorVisible(visible);

  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnDirectionAnnotationsVisibilityChanged(bool visible)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());
  if (!viewer)
  {
    /// Note: this slot is also directly invoked from this class. In this case sender() returns 0.
    viewer = this->GetSelectedViewer();
  }

  m_ControlPanel->SetDirectionAnnotationsVisible(visible);

  foreach (SingleViewerWidget* otherViewer, m_Viewers)
  {
    if (otherViewer != viewer)
    {
      bool signalsWereBlocked = otherViewer->blockSignals(true);
      otherViewer->SetDirectionAnnotationsVisible(visible);
      otherViewer->blockSignals(signalsWereBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnPositionAnnotationVisibilityChanged(bool visible)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());
  if (!viewer)
  {
    /// Note: this slot is also directly invoked from this class. In this case sender() returns 0.
    viewer = this->GetSelectedViewer();
  }

  m_ControlPanel->SetPositionAnnotationVisible(visible);

  foreach (SingleViewerWidget* otherViewer, m_Viewers)
  {
    if (otherViewer != viewer)
    {
      bool signalsWereBlocked = otherViewer->blockSignals(true);
      otherViewer->SetPositionAnnotationVisible(visible);
      otherViewer->blockSignals(signalsWereBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnIntensityAnnotationVisibilityChanged(bool visible)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());
  if (!viewer)
  {
    /// Note: this slot is also directly invoked from this class. In this case sender() returns 0.
    viewer = this->GetSelectedViewer();
  }

  m_ControlPanel->SetIntensityAnnotationVisible(visible);

  foreach (SingleViewerWidget* otherViewer, m_Viewers)
  {
    if (otherViewer != viewer)
    {
      bool signalsWereBlocked = otherViewer->blockSignals(true);
      otherViewer->SetIntensityAnnotationVisible(visible);
      otherViewer->blockSignals(signalsWereBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnPropertyAnnotationVisibilityChanged(bool visible)
{
  SingleViewerWidget* viewer = qobject_cast<SingleViewerWidget*>(this->sender());
  if (!viewer)
  {
    /// Note: this slot is also directly invoked from this class. In this case sender() returns 0.
    viewer = this->GetSelectedViewer();
  }

  m_ControlPanel->SetPropertyAnnotationVisible(visible);

  foreach (SingleViewerWidget* otherViewer, m_Viewers)
  {
    if (otherViewer != viewer)
    {
      bool signalsWereBlocked = otherViewer->blockSignals(true);
      otherViewer->SetPropertyAnnotationVisible(visible);
      otherViewer->blockSignals(signalsWereBlocked);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnTimeGeometryChanged(const mitk::TimeGeometry* timeGeometry)
{
  SingleViewerWidget* dropOntoViewer = qobject_cast<SingleViewerWidget*>(this->sender());
  SingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  if (dropOntoViewer == selectedViewer)
  {
    int maxTimeStep = selectedViewer->GetMaxTimeStep();
    int timeStep = selectedViewer->GetTimeStep();
    m_ControlPanel->SetMaxTimeStep(maxTimeStep);
    m_ControlPanel->SetTimeStep(timeStep);

    WindowOrientation orientation = selectedViewer->GetOrientation();
    if (orientation != WINDOW_ORIENTATION_UNKNOWN)
    {
      int maxSlice = selectedViewer->GetMaxSlice(orientation);
      int selectedSlice = selectedViewer->GetSelectedSlice(orientation);
      m_ControlPanel->SetMaxSlice(maxSlice);
      m_ControlPanel->SetSelectedSlice(selectedSlice);
    }
    else
    {
      /// TODO disable slice controls as well
      m_ControlPanel->SetMagnificationControlsEnabled(false);
    }
  }

  if (m_ControlPanel->AreViewerGeometriesBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
    {
      if (otherViewer != dropOntoViewer)
      {
        bool signalsWereBlocked = otherViewer->blockSignals(true);
        otherViewer->SetBoundTimeGeometry(timeGeometry);
        otherViewer->blockSignals(signalsWereBlocked);
      }
    }
  }

  if (m_ControlPanel->GetDropType() != DNDDISPLAY_DROP_ALL)
  {
    m_ControlPanel->SetSingleViewerControlsEnabled(true);
  }

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
void MultiViewerWidget::OnWindowCursorBindingControlChanged(bool bound)
{
  /// If the cursor positions are bound across the viewers then the binding property
  /// across the windows of the viewers can be controlled just together. That is, it
  /// is either set for each or none of them.
  if (m_ControlPanel->AreViewerCursorsBound())
  {
    foreach (SingleViewerWidget* viewer, m_Viewers)
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
void MultiViewerWidget::OnWindowMagnificationBindingControlChanged(bool bound)
{
  SingleViewerWidget* selectedViewer = this->GetSelectedViewer();
  bool signalsWereBlocked = selectedViewer->blockSignals(true);
  selectedViewer->SetScaleFactorBinding(bound);
  selectedViewer->blockSignals(signalsWereBlocked);

  /// If the scale factors are bound across the viewers then the binding property
  /// across the windows of the viewers can be controlled just together. That is, it
  /// is either set for each or none of them.
  if (m_ControlPanel->AreViewerMagnificationsBound())
  {
    foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
void MultiViewerWidget::OnCursorVisibilityControlChanged(bool visible)
{
  this->SetCursorVisible(visible);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnShowDirectionAnnotationsControlsChanged(bool visible)
{
  this->SetDirectionAnnotationsVisible(visible);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnShowPositionAnnotationControlsChanged(bool visible)
{
  this->SetPositionAnnotationVisible(visible);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnShowIntensityAnnotationControlsChanged(bool visible)
{
  this->SetIntensityAnnotationVisible(visible);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnShowPropertyAnnotationControlsChanged(bool visible)
{
  m_Viewers[m_SelectedViewerIndex]->SetPropertyAnnotationVisible(visible);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnPropertiesForAnnotationControlsChanged()
{
  m_Viewers[m_SelectedViewerIndex]->SetPropertiesForAnnotation(m_ControlPanel->GetPropertiesForAnnotation());
}


//-----------------------------------------------------------------------------
bool MultiViewerWidget::ToggleCursorVisibility()
{
  this->SetCursorVisible(!this->IsCursorVisible());

  return true;
}


//-----------------------------------------------------------------------------
WindowOrientation MultiViewerWidget::GetOrientation() const
{
  WindowOrientation orientation;

  switch (m_ControlPanel->GetWindowLayout())
  {
  case WINDOW_LAYOUT_AXIAL:
    orientation = WINDOW_ORIENTATION_AXIAL;
    break;
  case WINDOW_LAYOUT_SAGITTAL:
    orientation = WINDOW_ORIENTATION_SAGITTAL;
    break;
  case WINDOW_LAYOUT_CORONAL:
    orientation = WINDOW_ORIENTATION_CORONAL;
    break;
  default:
    orientation = WINDOW_ORIENTATION_UNKNOWN;
    break;
  }

  return orientation;
}


//-----------------------------------------------------------------------------
SingleViewerWidget* MultiViewerWidget::GetSelectedViewer() const
{
  assert(m_SelectedViewerIndex >= 0 && m_SelectedViewerIndex < m_Viewers.size());
  return m_Viewers[m_SelectedViewerIndex];
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* MultiViewerWidget::GetSelectedRenderWindow() const
{
  // NOTE: This MUST always return not-null.
  return this->GetSelectedViewer()->GetSelectedRenderWindow();
}


//-----------------------------------------------------------------------------
QHash<QString,QmitkRenderWindow*> MultiViewerWidget::GetRenderWindows() const
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

  SingleViewerWidget* selectedViewer = this->GetSelectedViewer();

  renderWindows.insert("axial", selectedViewer->GetAxialWindow());
  renderWindows.insert("sagittal", selectedViewer->GetSagittalWindow());
  renderWindows.insert("coronal", selectedViewer->GetCoronalWindow());
  renderWindows.insert("3d", selectedViewer->Get3DWindow());

  int i = 0;
  foreach (SingleViewerWidget* otherViewer, m_Viewers)
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
QmitkRenderWindow* MultiViewerWidget::GetRenderWindow(const QString& id) const
{
  QHash<QString,QmitkRenderWindow*> renderWindows = this->GetRenderWindows();
  QHash<QString,QmitkRenderWindow*>::iterator iter = renderWindows.find(id);
  if (iter != renderWindows.end())
  {
    return iter.value();
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------------
mitk::Point3D MultiViewerWidget::GetSelectedPosition(const QString& id) const
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
      foreach (SingleViewerWidget* viewer, m_Viewers)
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
void MultiViewerWidget::SetSelectedPosition(const mitk::Point3D& selectedPosition, const QString& id)
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
      foreach (SingleViewerWidget* viewer, m_Viewers)
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
bool MultiViewerWidget::IsLinkedNavigationEnabled() const
{
  return m_LinkedNavigationEnabled;
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::EnableLinkedNavigation(bool linkedNavigationEnabled)
{
  if (linkedNavigationEnabled != m_LinkedNavigationEnabled)
  {
    m_LinkedNavigationEnabled = linkedNavigationEnabled;
    foreach (SingleViewerWidget* viewer, m_Viewers)
    {
      viewer->SetLinkedNavigationEnabled(linkedNavigationEnabled);
    }
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnViewerPositionBindingControlChanged(bool bound)
{
  int bindingOptions = m_BindingOptions;

  if (bound)
  {
    bindingOptions |= PositionBinding;
  }
  else
  {
    bindingOptions &= ~PositionBinding;
  }

  bool signalsWereBlocked = this->blockSignals(true);
  this->SetBindingOptions(bindingOptions);
  this->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnViewerCursorBindingControlChanged(bool bound)
{
  int bindingOptions = m_BindingOptions;

  if (bound)
  {
    bindingOptions |= CursorBinding;
  }
  else
  {
    bindingOptions &= ~CursorBinding;
  }

  bool signalsWereBlocked = this->blockSignals(true);
  this->SetBindingOptions(bindingOptions);
  this->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnViewerMagnificationBindingControlChanged(bool bound)
{
  int bindingOptions = m_BindingOptions;

  if (bound)
  {
    bindingOptions |= MagnificationBinding;
  }
  else
  {
    bindingOptions &= ~MagnificationBinding;
  }

  bool signalsWereBlocked = this->blockSignals(true);
  this->SetBindingOptions(bindingOptions);
  this->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnViewerVisibilityBindingControlChanged(bool bound)
{
  int bindingOptions = m_BindingOptions;

  if (bound)
  {
    bindingOptions |= VisibilityBinding;
  }
  else
  {
    bindingOptions &= ~VisibilityBinding;
  }

  bool signalsWereBlocked = this->blockSignals(true);
  this->SetBindingOptions(bindingOptions);
  this->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnViewerWindowLayoutBindingControlChanged(bool bound)
{
  int bindingOptions = m_BindingOptions;

  if (bound)
  {
    bindingOptions |= WindowLayoutBinding;
  }
  else
  {
    bindingOptions &= ~WindowLayoutBinding;
  }

  bool signalsWereBlocked = this->blockSignals(true);
  this->SetBindingOptions(bindingOptions);
  this->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnViewerGeometryBindingControlChanged(bool bound)
{
  int bindingOptions = m_BindingOptions;

  if (bound)
  {
    bindingOptions |= GeometryBinding;
  }
  else
  {
    bindingOptions &= ~GeometryBinding;
  }

  bool signalsWereBlocked = this->blockSignals(true);
  this->SetBindingOptions(bindingOptions);
  this->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnPopupOpened(bool opened)
{
  if (!opened)
  {
    this->GetSelectedViewer()->repaint();
  }
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetSliceTracking(bool tracking)
{
  m_ControlPanel->SetSliceTracking(tracking);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetTimeStepTracking(bool tracking)
{
  m_ControlPanel->SetTimeStepTracking(tracking);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::SetMagnificationTracking(bool tracking)
{
  m_ControlPanel->SetMagnificationTracking(tracking);
}


//-----------------------------------------------------------------------------
void MultiViewerWidget::OnPinButtonToggled(bool checked)
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
bool MultiViewerWidget::eventFilter(QObject* object, QEvent* event)
{
  if (object == m_PinButton && event->type() == QEvent::Enter)
  {
    m_PopupWidget->showPopup();
  }
  return this->QObject::eventFilter(object, event);
}

}
