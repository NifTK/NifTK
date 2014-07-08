/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSingleViewerEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>

#include <QGridLayout>
#include <QToolButton>

#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkIDataStorageService.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>

#include <niftkSingleViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>

#include "QmitkDnDDisplayPreferencePage.h"

#include <ctkPopupWidget.h>
#include <niftkSingleViewerControls.h>

const std::string QmitkSingleViewerEditor::EDITOR_ID = "org.mitk.editors.dnddisplay";

class QmitkSingleViewerEditorPrivate
{
public:
  QmitkSingleViewerEditorPrivate();
  ~QmitkSingleViewerEditorPrivate();

  niftkSingleViewerWidget* m_SingleViewer;
  niftkMultiViewerVisibilityManager* m_VisibilityManager;
  mitk::RenderingManager::Pointer m_RenderingManager;
  berry::IPartListener::Pointer m_PartListener;
  mitk::IRenderingManager* m_RenderingManagerInterface;

  bool m_ShowCursor;
  double m_Magnification;

  unsigned long m_FocusManagerObserverTag;

  // Layouts
  QGridLayout* m_TopLevelLayout;
  QGridLayout* m_LayoutForRenderWindows;

  // Widgets
  QToolButton* m_PinButton;
  ctkPopupWidget* m_PopupWidget;
  niftkSingleViewerControls* m_ControlPanel;
};

//-----------------------------------------------------------------------------
struct QmitkSingleViewerEditorPartListener : public berry::IPartListener
{
  berryObjectMacro(QmitkSingleViewerEditorPartListener)

  //-----------------------------------------------------------------------------
  QmitkSingleViewerEditorPartListener(QmitkSingleViewerEditorPrivate* dd)
    : d(dd)
  {}


  //-----------------------------------------------------------------------------
  Events::Types GetPartEventTypes() const
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }


  //-----------------------------------------------------------------------------
  void PartClosed (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleViewerEditor::EDITOR_ID)
    {
      QmitkSingleViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkSingleViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetSingleViewer() == d->m_SingleViewer)
      {
        d->m_SingleViewer->SetLinkedNavigationEnabled(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleViewerEditor::EDITOR_ID)
    {
      QmitkSingleViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkSingleViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetSingleViewer() == d->m_SingleViewer)
      {
        d->m_SingleViewer->SetLinkedNavigationEnabled(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkSingleViewerEditor::EDITOR_ID)
    {
      QmitkSingleViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkSingleViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetSingleViewer() == d->m_SingleViewer)
      {
        d->m_SingleViewer->SetLinkedNavigationEnabled(true);
      }
    }
  }

private:

  QmitkSingleViewerEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
QmitkSingleViewerEditorPrivate::QmitkSingleViewerEditorPrivate()
: m_SingleViewer(0)
, m_VisibilityManager(0)
, m_RenderingManager(0)
, m_PartListener(new QmitkSingleViewerEditorPartListener(this))
, m_RenderingManagerInterface(0)
, m_ShowCursor(true)
, m_Magnification(0.0)
, m_FocusManagerObserverTag(0)
, m_TopLevelLayout(0)
, m_LayoutForRenderWindows(0)
, m_PinButton(0)
, m_PopupWidget(0)
, m_ControlPanel(0)
{
  m_RenderingManager = mitk::RenderingManager::GetInstance();
  m_RenderingManager->SetConstrainedPaddingZooming(false);
  m_RenderingManagerInterface = mitk::MakeRenderingManagerInterface(m_RenderingManager);
}


//-----------------------------------------------------------------------------
QmitkSingleViewerEditorPrivate::~QmitkSingleViewerEditorPrivate()
{
  if (m_VisibilityManager != NULL)
  {
    delete m_VisibilityManager;
  }

  if (m_RenderingManagerInterface != NULL)
  {
    delete m_RenderingManagerInterface;
  }
}


//-----------------------------------------------------------------------------
QmitkSingleViewerEditor::QmitkSingleViewerEditor()
: d(new QmitkSingleViewerEditorPrivate)
{
}


//-----------------------------------------------------------------------------
QmitkSingleViewerEditor::~QmitkSingleViewerEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener);

  // Deregister focus observer.
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  focusManager->RemoveObserver(d->m_FocusManagerObserverTag);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_SingleViewer == NULL)
  {
    parent->setFocusPolicy(Qt::StrongFocus);

    /************************************
     * Create stuff.
     ************************************/

    d->m_TopLevelLayout = new QGridLayout(parent);
    d->m_TopLevelLayout->setObjectName(QString::fromUtf8("QmitkSingleViewerEditor::m_TopLevelLayout"));
    d->m_TopLevelLayout->setContentsMargins(0, 0, 0, 0);
    d->m_TopLevelLayout->setSpacing(0);

    d->m_LayoutForRenderWindows = new QGridLayout();
    d->m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("QmitkSingleViewerEditor::m_LayoutForRenderWindows"));
    d->m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
    d->m_LayoutForRenderWindows->setSpacing(0);

    QWidget* pinButtonWidget = new QWidget(parent);
    pinButtonWidget->setContentsMargins(0, 0, 0, 0);
    QVBoxLayout* pinButtonWidgetLayout = new QVBoxLayout(pinButtonWidget);
    pinButtonWidgetLayout->setContentsMargins(0, 0, 0, 0);
    pinButtonWidgetLayout->setSpacing(0);
    pinButtonWidget->setLayout(pinButtonWidgetLayout);

    d->m_PopupWidget = new ctkPopupWidget(pinButtonWidget);
    d->m_PopupWidget->setOrientation(Qt::Vertical);
    d->m_PopupWidget->setAnimationEffect(ctkBasePopupWidget::ScrollEffect);
    d->m_PopupWidget->setHorizontalDirection(Qt::LeftToRight);
    d->m_PopupWidget->setVerticalDirection(ctkBasePopupWidget::TopToBottom);
    d->m_PopupWidget->setAutoShow(true);
    d->m_PopupWidget->setAutoHide(true);
    d->m_PopupWidget->setEffectDuration(100);
    d->m_PopupWidget->setContentsMargins(5, 5, 5, 1);
    d->m_PopupWidget->setLineWidth(0);

  #ifdef __APPLE__
    QPalette popupPalette = parent->palette();
    QColor windowColor = popupPalette.color(QPalette::Window);
    windowColor.setAlpha(64);
    popupPalette.setColor(QPalette::Window, windowColor);
    d->m_PopupWidget->setPalette(popupPalette);
  #else
    QPalette popupPalette = parent->palette();
    QColor windowColor = popupPalette.color(QPalette::Window);
    windowColor.setAlpha(128);
    popupPalette.setColor(QPalette::Window, windowColor);
    d->m_PopupWidget->setPalette(popupPalette);
    d->m_PopupWidget->setAttribute(Qt::WA_TranslucentBackground, true);
  #endif

    int buttonRowHeight = 15;
    d->m_PinButton = new QToolButton(parent);
    d->m_PinButton->setContentsMargins(0, 0, 0, 0);
    d->m_PinButton->setCheckable(true);
    d->m_PinButton->setAutoRaise(true);
    d->m_PinButton->setFixedHeight(16);
    QSizePolicy pinButtonSizePolicy;
    pinButtonSizePolicy.setHorizontalPolicy(QSizePolicy::Expanding);
    d->m_PinButton->setSizePolicy(pinButtonSizePolicy);
    // These two lines ensure that the icon appears on the left on each platform.
    d->m_PinButton->setText(" ");
    d->m_PinButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    QIcon pinButtonIcon;
    pinButtonIcon.addFile(":Icons/PushPinIn.png", QSize(), QIcon::Normal, QIcon::On);
    pinButtonIcon.addFile(":Icons/PushPinOut.png", QSize(), QIcon::Normal, QIcon::Off);
    d->m_PinButton->setIcon(pinButtonIcon);

    this->connect(d->m_PinButton, SIGNAL(toggled(bool)), SLOT(OnPinButtonToggled(bool)));
    d->m_PinButton->installEventFilter(this);

    d->m_ControlPanel = this->CreateControlPanel(d->m_PopupWidget);

    pinButtonWidgetLayout->addWidget(d->m_PinButton);

    d->m_TopLevelLayout->addWidget(pinButtonWidget, 0, 0);
    d->m_TopLevelLayout->setRowMinimumHeight(0, buttonRowHeight);
    d->m_TopLevelLayout->addLayout(d->m_LayoutForRenderWindows, 1, 0);


    d->m_LayoutForRenderWindows = new QGridLayout();
    d->m_LayoutForRenderWindows->setObjectName(QString::fromUtf8("QmitkSingleViewerEditor::m_LayoutForRenderWindows"));
    d->m_LayoutForRenderWindows->setContentsMargins(0, 0, 0, 0);
    d->m_LayoutForRenderWindows->setVerticalSpacing(0);
    d->m_LayoutForRenderWindows->setHorizontalSpacing(0);

    d->m_TopLevelLayout->addLayout(d->m_LayoutForRenderWindows, 1, 0);


    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
    berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(EDITOR_ID)).Cast<berry::IBerryPreferences>();
    assert( prefs );

    DnDDisplayInterpolationType defaultInterpolationType =
        (DnDDisplayInterpolationType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2));
    WindowLayout defaultLayout =
        (WindowLayout)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2)); // default = coronal

    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(QmitkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR, "black"));
    QColor backgroundColour(backgroundColourName);
    bool showDirectionAnnotations = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true);
    bool showShowingOptions = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true);
    bool showWindowLayoutControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true);
    bool showMagnificationSlider = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true);
    bool show3DWindowInMultiWindowLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false);
    bool show2DCursors = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true);
    bool rememberSettingsPerLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true);
    bool sliceIndexTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true);
    bool magnificationTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true);
    bool timeStepTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true);

    d->m_VisibilityManager = new niftkMultiViewerVisibilityManager(dataStorage);
    d->m_VisibilityManager->SetInterpolationType(defaultInterpolationType);
    d->m_VisibilityManager->SetDefaultWindowLayout(defaultLayout);
    d->m_VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);

    d->m_RenderingManager->SetDataStorage(dataStorage);

    d->m_ShowCursor = show2DCursors;

    // Create the niftkSingleViewerWidget
    d->m_SingleViewer = new niftkSingleViewerWidget(parent, d->m_RenderingManager);

    // Setup GUI a bit more.
    d->m_SingleViewer->SetBackgroundColour(backgroundColour);
    d->m_SingleViewer->SetDisplayInteractionsEnabled(true);
    d->m_SingleViewer->SetCursorPositionBinding(true);
    d->m_SingleViewer->SetScaleFactorBinding(true);
    d->m_ControlPanel->SetShowOptionsVisible(showShowingOptions);
    d->m_ControlPanel->SetWindowLayoutControlsVisible(showWindowLayoutControls);
//    d->m_SingleViewer->SetCursorDefaultVisibility(show2DCursors);
    d->m_SingleViewer->SetDirectionAnnotationsVisible(showDirectionAnnotations);
    d->m_ControlPanel->SetDirectionAnnotationsVisible(showDirectionAnnotations);
    d->m_SingleViewer->SetShow3DWindowIn2x2WindowLayout(show3DWindowInMultiWindowLayout);
    d->m_ControlPanel->SetMagnificationControlsVisible(showMagnificationSlider);
    d->m_SingleViewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerLayout);
    d->m_ControlPanel->SetSliceTracking(sliceIndexTracking);
    d->m_ControlPanel->SetTimeStepTracking(timeStepTracking);
    d->m_ControlPanel->SetMagnificationTracking(magnificationTracking);
    d->m_VisibilityManager->SetDefaultWindowLayout(defaultLayout);
//    d->m_SingleViewer->SetDefaultSingleWindowLayout(singleWindowLayout);
//    d->m_SingleViewer->SetDefaultMultiWindowLayout(multiWindowLayout);

    d->m_VisibilityManager->RegisterViewer(d->m_SingleViewer);

    this->GetSite()->GetPage()->AddPartListener(berry::IPartListener::Pointer(d->m_PartListener));

    d->m_LayoutForRenderWindows->addWidget(d->m_SingleViewer, 0, 0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<QmitkSingleViewerEditor, const berry::IBerryPreferences*>( this, &QmitkSingleViewerEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());

    // Connect Qt Signals to make it all hang together.

    this->connect(d->m_SingleViewer, SIGNAL(TimeGeometryChanged(const mitk::TimeGeometry*)), SLOT(OnTimeGeometryChanged(const mitk::TimeGeometry*)));
    this->connect(d->m_SingleViewer, SIGNAL(SelectedPositionChanged(const mitk::Point3D&)), SLOT(OnSelectedPositionChanged(const mitk::Point3D&)));
    this->connect(d->m_SingleViewer, SIGNAL(SelectedTimeStepChanged(int)), SLOT(OnSelectedTimeStepChanged(int)));
    this->connect(d->m_SingleViewer, SIGNAL(ScaleFactorChanged(WindowOrientation, double)), SLOT(OnScaleFactorChanged(WindowOrientation, double)));
    this->connect(d->m_SingleViewer, SIGNAL(WindowLayoutChanged(WindowLayout)), SLOT(OnWindowLayoutChanged(WindowLayout)));
    this->connect(d->m_SingleViewer, SIGNAL(CursorVisibilityChanged(bool)), SLOT(OnCursorVisibilityChanged(bool)));

    this->connect(d->m_ControlPanel, SIGNAL(SelectedSliceChanged(int)), SLOT(OnSelectedSliceControlChanged(int)));
    this->connect(d->m_ControlPanel, SIGNAL(TimeStepChanged(int)), SLOT(OnTimeStepControlChanged(int)));
    this->connect(d->m_ControlPanel, SIGNAL(MagnificationChanged(double)), SLOT(OnMagnificationControlChanged(double)));

    this->connect(d->m_ControlPanel, SIGNAL(ShowCursorChanged(bool)), SLOT(OnCursorVisibilityControlChanged(bool)));
    this->connect(d->m_ControlPanel, SIGNAL(ShowDirectionAnnotationsChanged(bool)), SLOT(OnShowDirectionAnnotationsControlChanged(bool)));
    this->connect(d->m_ControlPanel, SIGNAL(Show3DWindowChanged(bool)), SLOT(OnShow3DWindowControlChanged(bool)));

    this->connect(d->m_ControlPanel, SIGNAL(WindowLayoutChanged(WindowLayout)), SLOT(OnWindowLayoutControlChanged(WindowLayout)));
    this->connect(d->m_ControlPanel, SIGNAL(WindowCursorBindingChanged(bool)), SLOT(OnWindowCursorBindingControlChanged(bool)));
    this->connect(d->m_ControlPanel, SIGNAL(WindowMagnificationBindingChanged(bool)), SLOT(OnWindowScaleFactorBindingControlChanged(bool)));

    this->connect(d->m_PopupWidget, SIGNAL(popupOpened(bool)), SLOT(OnPopupOpened(bool)));

    // Register focus observer.
    itk::SimpleMemberCommand<QmitkSingleViewerEditor>::Pointer onFocusChangedCommand = itk::SimpleMemberCommand<QmitkSingleViewerEditor>::New();
    onFocusChangedCommand->SetCallbackFunction(this, &QmitkSingleViewerEditor::OnFocusChanged);
    mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
    d->m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
  }
}


//-----------------------------------------------------------------------------
niftkSingleViewerControls* QmitkSingleViewerEditor::CreateControlPanel(QWidget* parent)
{
  niftkSingleViewerControls* controlPanel = new niftkSingleViewerControls(parent);

  controlPanel->SetWindowCursorsBound(true);
  controlPanel->SetWindowMagnificationsBound(true);

  return controlPanel;
}


//-----------------------------------------------------------------------------
niftkSingleViewerWidget* QmitkSingleViewerEditor::GetSingleViewer()
{
  return d->m_SingleViewer;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::SetFocus()
{
  if (d->m_SingleViewer != 0)
  {
    d->m_SingleViewer->SetFocused();
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnTimeGeometryChanged(const mitk::TimeGeometry* timeGeometry)
{
  Q_UNUSED(timeGeometry);

  d->m_ControlPanel->SetCursorVisible(d->m_ShowCursor);
  d->m_SingleViewer->SetCursorVisible(d->m_ShowCursor);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (d->m_SingleViewer != NULL)
  {
    QString backgroundColourName = QString::fromStdString (prefs->GetByteArray(QmitkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR, "black"));
    QColor backgroundColour(backgroundColourName);
    d->m_SingleViewer->SetBackgroundColour(backgroundColour);
    d->m_VisibilityManager->SetInterpolationType((DnDDisplayInterpolationType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2)));
    d->m_VisibilityManager->SetDefaultWindowLayout((WindowLayout)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2))); // default coronal
    d->m_ControlPanel->SetShowOptionsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
    d->m_ControlPanel->SetWindowLayoutControlsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
    d->m_ControlPanel->SetMagnificationControlsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
    d->m_ShowCursor = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true);

    d->m_SingleViewer->SetDirectionAnnotationsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
    d->m_SingleViewer->SetShow3DWindowIn2x2WindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
    d->m_SingleViewer->SetRememberSettingsPerWindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
    d->m_ControlPanel->SetSliceTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true));
    d->m_ControlPanel->SetTimeStepTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true));
    d->m_ControlPanel->SetMagnificationTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  }
}

//-----------------------------------------------------------------------------
// -------------------  mitk::IRenderWindowPart  ------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkSingleViewerEditor::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow* activeRenderWindow = d->m_SingleViewer->GetSelectedRenderWindow();
  if (!activeRenderWindow)
  {
    activeRenderWindow = d->m_SingleViewer->GetVisibleRenderWindows()[0];
  }
  return activeRenderWindow;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> QmitkSingleViewerEditor::GetQmitkRenderWindows() const
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

  renderWindows.insert("axial", d->m_SingleViewer->GetAxialWindow());
  renderWindows.insert("sagittal", d->m_SingleViewer->GetSagittalWindow());
  renderWindows.insert("coronal", d->m_SingleViewer->GetCoronalWindow());
  renderWindows.insert("3d", d->m_SingleViewer->Get3DWindow());

  return renderWindows;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkSingleViewerEditor::GetQmitkRenderWindow(const QString& id) const
{
  return this->GetQmitkRenderWindows()[id];
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnFocusChanged()
{
  mitk::BaseRenderer* focusedRenderer = mitk::GlobalInteraction::GetInstance()->GetFocus();

  if (d->m_SingleViewer->GetSelectedRenderWindow()->GetRenderer() != focusedRenderer)
  {
    return;
  }

  WindowOrientation orientation = d->m_SingleViewer->GetOrientation();

  if (orientation != WINDOW_ORIENTATION_UNKNOWN)
  {
    bool signalsWereBlocked = d->m_ControlPanel->blockSignals(true);

    int maxSlice = d->m_SingleViewer->GetMaxSlice(orientation);
    int selectedSlice = d->m_SingleViewer->GetSelectedSlice(orientation);
    d->m_ControlPanel->SetMaxSlice(maxSlice);
    d->m_ControlPanel->SetSelectedSlice(selectedSlice);

    d->m_ControlPanel->SetMagnificationControlsEnabled(true);
    double minMagnification = std::ceil(d->m_SingleViewer->GetMinMagnification());
    double maxMagnification = std::floor(d->m_SingleViewer->GetMaxMagnification());
    double magnification = d->m_SingleViewer->GetMagnification(orientation);
    d->m_ControlPanel->SetMinMagnification(minMagnification);
    d->m_ControlPanel->SetMaxMagnification(maxMagnification);
    d->m_ControlPanel->SetMagnification(magnification);

    d->m_ControlPanel->blockSignals(signalsWereBlocked);
  }
  else
  {
    d->m_ControlPanel->SetMagnificationControlsEnabled(false);
  }

  WindowLayout windowLayout = d->m_SingleViewer->GetWindowLayout();

  if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    bool signalsWereBlocked = d->m_ControlPanel->blockSignals(true);
    d->m_ControlPanel->SetWindowLayout(windowLayout);
    d->m_ControlPanel->blockSignals(signalsWereBlocked);
  }
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkSingleViewerEditor::GetSelectedPosition(const QString& id) const
{
  if (id.isNull() || this->GetQmitkRenderWindow(id))
  {
    return d->m_SingleViewer->GetSelectedPosition();
  }

  mitk::Point3D fallBackValue;
  fallBackValue.Fill(0.0);
  return fallBackValue;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::SetSelectedPosition(const mitk::Point3D& position, const QString& id)
{
  if (id.isNull() || this->GetQmitkRenderWindow(id))
  {
    d->m_SingleViewer->SetSelectedPosition(position);
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
}


//-----------------------------------------------------------------------------
bool QmitkSingleViewerEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations. ToDo - maybe get niftkSingleViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
QStringList QmitkSingleViewerEditor::GetDecorations() const
{
  // Deliberately return nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::IRenderingManager* QmitkSingleViewerEditor::GetRenderingManager() const
{
  return d->m_RenderingManagerInterface;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* QmitkSingleViewerEditor::GetSlicesRotator() const
{
  // Deliberately return nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* QmitkSingleViewerEditor::GetSlicesSwiveller() const
{
  // Deliberately return nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::EnableSlicingPlanes(bool enable)
{
  // Deliberately do nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  Q_UNUSED(enable);
}


//-----------------------------------------------------------------------------
bool QmitkSingleViewerEditor::IsSlicingPlanesEnabled() const
{
  // Deliberately do nothing. ToDo - maybe get niftkSingleViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::EnableLinkedNavigation(bool linkedNavigationEnabled)
{
  d->m_SingleViewer->SetLinkedNavigationEnabled(linkedNavigationEnabled);
}


//-----------------------------------------------------------------------------
bool QmitkSingleViewerEditor::IsLinkedNavigationEnabled() const
{
  return d->m_SingleViewer->IsLinkedNavigationEnabled();
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnPopupOpened(bool opened)
{
  if (!opened)
  {
    d->m_SingleViewer->repaint();
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnPinButtonToggled(bool checked)
{
  if (checked)
  {
    d->m_PopupWidget->pinPopup(true);
  }
  else
  {
    d->m_PopupWidget->setAutoHide(true);
  }
}


//---------------------------------------------------------------------------
bool QmitkSingleViewerEditor::eventFilter(QObject* object, QEvent* event)
{
  if (object == d->m_PinButton && event->type() == QEvent::Enter)
  {
    d->m_PopupWidget->showPopup();
  }
  return this->QObject::eventFilter(object, event);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnSelectedSliceControlChanged(int selectedSlice)
{
  WindowOrientation orientation = d->m_SingleViewer->GetOrientation();

  if (orientation != WINDOW_ORIENTATION_UNKNOWN)
  {
    bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
    d->m_SingleViewer->SetSelectedSlice(orientation, selectedSlice);
    d->m_SingleViewer->blockSignals(signalsWereBlocked);

  }
  else
  {
    MITK_WARN << "Found an invalid orientation in viewer, so ignoring request to change to slice " << selectedSlice << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnTimeStepControlChanged(int timeStep)
{
  bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
  d->m_SingleViewer->SetTimeStep(timeStep);
  d->m_SingleViewer->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnMagnificationControlChanged(double magnification)
{
  double roundedMagnification = std::floor(magnification);

  // If we are between two integers, we raise a new event:
  if (magnification != roundedMagnification)
  {
    // If the value has decreased, we have to increase the rounded value.
    if (magnification < d->m_Magnification)
    {
      roundedMagnification += 1.0;
    }

    magnification = roundedMagnification;
    d->m_ControlPanel->SetMagnification(magnification);
  }

  WindowOrientation orientation = d->m_SingleViewer->GetOrientation();

  bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
  d->m_SingleViewer->SetMagnification(orientation, magnification);
  d->m_SingleViewer->blockSignals(signalsWereBlocked);

  d->m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnShowDirectionAnnotationsControlChanged(bool visible)
{
  bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
  d->m_SingleViewer->SetDirectionAnnotationsVisible(visible);
  d->m_SingleViewer->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnCursorVisibilityControlChanged(bool visible)
{
  bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
  d->m_SingleViewer->SetCursorVisible(visible);
  d->m_SingleViewer->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnShow3DWindowControlChanged(bool visible)
{
  bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
  d->m_SingleViewer->SetShow3DWindowIn2x2WindowLayout(visible);
  d->m_SingleViewer->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnWindowCursorBindingControlChanged(bool bound)
{
  bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
  d->m_SingleViewer->SetCursorPositionBinding(bound);
  d->m_SingleViewer->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnWindowScaleFactorBindingControlChanged(bool bound)
{
  bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
  d->m_SingleViewer->SetScaleFactorBinding(bound);
  d->m_SingleViewer->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnWindowLayoutControlChanged(WindowLayout windowLayout)
{
  if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
  {
    bool signalsWereBlocked = d->m_SingleViewer->blockSignals(true);
    d->m_SingleViewer->SetWindowLayout(windowLayout);
    d->m_SingleViewer->blockSignals(signalsWereBlocked);
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnSelectedPositionChanged(const mitk::Point3D& selectedPosition)
{
  niftkSingleViewerWidget* viewer = qobject_cast<niftkSingleViewerWidget*>(this->sender());

  WindowOrientation orientation = d->m_SingleViewer->GetOrientation();
  if (orientation != WINDOW_ORIENTATION_UNKNOWN)
  {
    bool signalsWereBlocked = d->m_ControlPanel->blockSignals(true);
    d->m_ControlPanel->SetSelectedSlice(viewer->GetSelectedSlice(orientation));
    d->m_ControlPanel->blockSignals(signalsWereBlocked);
  }
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnSelectedTimeStepChanged(int timeStep)
{
  bool signalsWereBlocked = d->m_ControlPanel->blockSignals(true);
  d->m_ControlPanel->SetTimeStep(timeStep);
  d->m_ControlPanel->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnScaleFactorChanged(WindowOrientation orientation, double scaleFactor)
{
  double magnification = d->m_SingleViewer->GetMagnification(orientation);

  bool signalsWereBlocked = d->m_ControlPanel->blockSignals(true);
  d->m_ControlPanel->SetMagnification(magnification);
  d->m_ControlPanel->blockSignals(signalsWereBlocked);

  d->m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnCursorVisibilityChanged(bool visible)
{
  bool signalsWereBlocked = d->m_ControlPanel->blockSignals(true);
  d->m_ControlPanel->SetCursorVisible(visible);
  d->m_ControlPanel->blockSignals(signalsWereBlocked);
}


//-----------------------------------------------------------------------------
void QmitkSingleViewerEditor::OnWindowLayoutChanged(WindowLayout windowLayout)
{
  d->m_ControlPanel->SetWindowLayout(windowLayout);
  d->m_ControlPanel->SetWindowCursorsBound(d->m_SingleViewer->GetCursorPositionBinding());
  d->m_ControlPanel->SetWindowMagnificationsBound(d->m_SingleViewer->GetScaleFactorBinding());
}
