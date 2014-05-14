/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSideViewerWidget.h"

#include "QmitkBaseView.h"

#include <berryIWorkbenchPage.h>

#include <itkCommand.h>

#include <mitkBaseRenderer.h>
#include <mitkDataStorage.h>
#include <mitkFocusManager.h>
#include <mitkGeometry3D.h>
#include <mitkGlobalInteraction.h>
#include <mitkIRenderWindowPart.h>
#include <mitkSliceNavigationController.h>

#include <QHBoxLayout>
#include <QSpacerItem>
#include <QVBoxLayout>


//-----------------------------------------------------------------------------
QmitkSideViewerWidget::QmitkSideViewerWidget(QmitkBaseView* view, QWidget* parent)
: m_ContainingView(view)
, m_FocusManagerObserverTag(0)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_MainWindow(0)
, m_MainAxialWindow(0)
, m_MainSagittalWindow(0)
, m_MainCoronalWindow(0)
, m_MainAxialSnc(0)
, m_MainSagittalSnc(0)
, m_MainCoronalSnc(0)
, m_NodeAddedSetter(0)
, m_VisibilityTracker(0)
, m_Magnification(0.0)
, m_MainWindowOrientation(MIDAS_ORIENTATION_UNKNOWN)
, m_SingleWindowLayouts()
, m_MIDASToolNodeNameFilter(0)
, m_TimeGeometry(0)
{
  this->setupUi(parent);

  m_Viewer->SetShow3DWindowIn2x2WindowLayout(false);

  m_CoronalWindowRadioButton->setChecked(true);

  m_SingleWindowLayouts[MIDAS_ORIENTATION_AXIAL] = WINDOW_LAYOUT_CORONAL;
  m_SingleWindowLayouts[MIDAS_ORIENTATION_SAGITTAL] = WINDOW_LAYOUT_CORONAL;
  m_SingleWindowLayouts[MIDAS_ORIENTATION_CORONAL] = WINDOW_LAYOUT_SAGITTAL;

  m_MagnificationSpinBox->setDecimals(2);
  m_MagnificationSpinBox->setSingleStep(1.0);

  double minMagnification = std::ceil(m_Viewer->GetMinMagnification());
  double maxMagnification = std::floor(m_Viewer->GetMaxMagnification());

  m_MagnificationSpinBox->setMinimum(minMagnification);
  m_MagnificationSpinBox->setMaximum(maxMagnification);

  m_ControlsWidget->setEnabled(false);

  std::vector<const mitk::BaseRenderer*> renderers;
  renderers.push_back(m_Viewer->GetAxialWindow()->GetRenderer());
  renderers.push_back(m_Viewer->GetSagittalWindow()->GetRenderer());
  renderers.push_back(m_Viewer->GetCoronalWindow()->GetRenderer());

  m_NodeAddedSetter = mitk::DataNodeAddedVisibilitySetter::New();
  m_MIDASToolNodeNameFilter = mitk::MIDASDataNodeNameStringFilter::New();
  m_NodeAddedSetter->AddFilter(m_MIDASToolNodeNameFilter.GetPointer());
  m_NodeAddedSetter->SetRenderers(renderers);
  m_NodeAddedSetter->SetVisibility(false);

  m_VisibilityTracker = mitk::DataStorageVisibilityTracker::New();
  m_VisibilityTracker->SetNodesToIgnore(m_Viewer->GetWidgetPlanes());
  m_VisibilityTracker->SetManagedRenderers(renderers);

  m_Viewer->SetCursorVisible(true);
  m_Viewer->SetRememberSettingsPerWindowLayout(false);
  m_Viewer->SetDisplayInteractionsEnabled(true);
  m_Viewer->SetCursorPositionBinding(false);
  m_Viewer->SetScaleFactorBinding(true);

  this->connect(m_AxialWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnAxialWindowRadioButtonToggled(bool)));
  this->connect(m_SagittalWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnSagittalWindowRadioButtonToggled(bool)));
  this->connect(m_CoronalWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnCoronalWindowRadioButtonToggled(bool)));
  this->connect(m_MultiWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnMultiWindowRadioButtonToggled(bool)));
  this->connect(m_MultiWindowComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnMultiWindowComboBoxIndexChanged()));

  this->connect(m_Viewer, SIGNAL(WindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout)), SLOT(OnWindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout)));

  this->connect(m_SliceSpinBox, SIGNAL(valueChanged(int)), SLOT(OnSliceSpinBoxValueChanged(int)));
  this->connect(m_Viewer, SIGNAL(SelectedPositionChanged(niftkSingleViewerWidget*, const mitk::Point3D&)), SLOT(OnSelectedPositionChanged(niftkSingleViewerWidget*, const mitk::Point3D&)));
  this->connect(m_MagnificationSpinBox, SIGNAL(valueChanged(double)), SLOT(OnMagnificationSpinBoxValueChanged(double)));
  this->connect(m_Viewer, SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, MIDASOrientation, double)), SLOT(OnScaleFactorChanged(niftkSingleViewerWidget*, MIDASOrientation, double)));

  // Register focus observer.
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    itk::SimpleMemberCommand<QmitkSideViewerWidget>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<QmitkSideViewerWidget>::New();
    onFocusChangedCommand->SetCallbackFunction( this, &QmitkSideViewerWidget::OnFocusChanged );

    m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
  }

  mitk::IRenderWindowPart* selectedEditor = this->GetSelectedEditor();
  if (selectedEditor)
  {
    QmitkRenderWindow* selectedMainWindow = selectedEditor->GetActiveQmitkRenderWindow();
    this->OnMainWindowChanged(selectedMainWindow);
  }
}


//-----------------------------------------------------------------------------
QmitkSideViewerWidget::~QmitkSideViewerWidget()
{
  m_VisibilityTracker->SetTrackedRenderer(0);
  m_Viewer->SetEnabled(false);

  /// Note:
  /// To avoid updating the world geometry three times, we connect the geometry send event
  /// only to the axial window and assume that the three 2D renderer windows have the same
  /// world geometry.
  if (m_MainAxialWindow)
  {
    m_MainAxialSnc->Disconnect(this);
    m_MainAxialSnc->Disconnect(m_Viewer->GetAxialWindow()->GetSliceNavigationController());
    m_Viewer->GetAxialWindow()->GetSliceNavigationController()->Disconnect(m_MainAxialSnc);
  }
  if (m_MainSagittalWindow)
  {
    m_MainSagittalSnc->Disconnect(m_Viewer->GetSagittalWindow()->GetSliceNavigationController());
    m_Viewer->GetSagittalWindow()->GetSliceNavigationController()->Disconnect(m_MainSagittalSnc);
  }
  if (m_MainCoronalWindow)
  {
    m_MainCoronalSnc->Disconnect(m_Viewer->GetCoronalWindow()->GetSliceNavigationController());
    m_Viewer->GetCoronalWindow()->GetSliceNavigationController()->Disconnect(m_MainCoronalSnc);
  }

  // Deregister focus observer.
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }

  // m_NodeAddedSetter deleted by smart pointer.
  // m_VisibilityTracker deleted by smart pointer.
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::SetDataStorage(mitk::DataStorage* dataStorage)
{
  if (dataStorage)
  {
    m_Viewer->SetDataStorage(dataStorage);

    m_NodeAddedSetter->SetDataStorage(dataStorage);
    m_VisibilityTracker->SetDataStorage(dataStorage);
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnAMainWindowDestroyed(QObject* mainWindow)
{
  if (mainWindow == m_MainAxialWindow)
  {
    m_VisibilityTracker->SetTrackedRenderer(0);
    m_Viewer->SetEnabled(false);
    m_Viewer->GetAxialWindow()->GetSliceNavigationController()->Disconnect(m_MainAxialSnc);
    m_MainAxialWindow = 0;
    m_MainAxialSnc = 0;
  }
  else if (mainWindow == m_MainSagittalWindow)
  {
    m_VisibilityTracker->SetTrackedRenderer(0);
    m_Viewer->SetEnabled(false);
    m_Viewer->GetSagittalWindow()->GetSliceNavigationController()->Disconnect(m_MainSagittalSnc);
    m_MainSagittalWindow = 0;
    m_MainSagittalSnc = 0;
  }
  else if (mainWindow == m_MainCoronalWindow)
  {
    m_VisibilityTracker->SetTrackedRenderer(0);
    m_Viewer->SetEnabled(false);
    m_Viewer->GetCoronalWindow()->GetSliceNavigationController()->Disconnect(m_MainCoronalSnc);
    m_MainCoronalWindow = 0;
    m_MainCoronalSnc = 0;
  }
  else
  {
    /// We do not update the viewer. Skip here.
    return;
  }

  m_Viewer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_SingleWindowLayouts[m_MainWindowOrientation] = WINDOW_LAYOUT_AXIAL;
    m_Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
    /// TODO The SetWindowLayout call should emit a signal but it does not at the moment.
    this->OnWindowLayoutChanged(m_Viewer, WINDOW_LAYOUT_AXIAL);
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_SingleWindowLayouts[m_MainWindowOrientation] = WINDOW_LAYOUT_SAGITTAL;
    m_Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
    /// TODO The SetWindowLayout call should emit a signal but it does not at the moment.
    this->OnWindowLayoutChanged(m_Viewer, WINDOW_LAYOUT_SAGITTAL);
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_SingleWindowLayouts[m_MainWindowOrientation] = WINDOW_LAYOUT_CORONAL;
    m_Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
    /// TODO The SetWindowLayout call should emit a signal but it does not at the moment.
    this->OnWindowLayoutChanged(m_Viewer, WINDOW_LAYOUT_CORONAL);
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->OnMultiWindowComboBoxIndexChanged();
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnMultiWindowComboBoxIndexChanged()
{
  if (!m_MultiWindowRadioButton->isChecked())
  {
    m_MultiWindowRadioButton->setChecked(true);
  }

  WindowLayout windowLayout = WINDOW_LAYOUT_UNKNOWN;

  // 2H
  if (m_MultiWindowComboBox->currentIndex() == 0)
  {
    if (m_MainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_SAG_H;
    }
    else if (m_MainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_AX_H;
    }
    else if (m_MainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
    {
      windowLayout = WINDOW_LAYOUT_SAG_AX_H;
    }
  }
  // 2V
  else if (m_MultiWindowComboBox->currentIndex() == 1)
  {
    if (m_MainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_SAG_V;
    }
    else if (m_MainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_AX_V;
    }
    else if (m_MainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
    {
      windowLayout = WINDOW_LAYOUT_SAG_AX_V;
    }
  }

  /// TODO The SetWindowLayout call should emit a signal but it does not at the moment.
  m_Viewer->SetWindowLayout(windowLayout);
  this->OnWindowLayoutChanged(m_Viewer, windowLayout);
}


//-----------------------------------------------------------------------------
WindowLayout QmitkSideViewerWidget::GetMultiWindowLayoutForOrientation(MIDASOrientation mainWindowOrientation)
{
  WindowLayout windowLayout = WINDOW_LAYOUT_UNKNOWN;

  // 2H
  if (m_MultiWindowComboBox->currentIndex() == 0)
  {
    if (mainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_SAG_H;
    }
    else if (mainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_AX_H;
    }
    else if (mainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
    {
      windowLayout = WINDOW_LAYOUT_SAG_AX_H;
    }
  }
  // 2V
  else if (m_MultiWindowComboBox->currentIndex() == 1)
  {
    if (mainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_SAG_V;
    }
    else if (mainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      windowLayout = WINDOW_LAYOUT_COR_AX_V;
    }
    else if (mainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
    {
      windowLayout = WINDOW_LAYOUT_SAG_AX_V;
    }
  }

  return windowLayout;
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnMainWindowGeometryChanged(const mitk::TimeGeometry* timeGeometry)
{
  m_Viewer->SetTimeGeometry(timeGeometry);

  std::vector<mitk::DataNode*> crossHairs = m_Viewer->GetWidgetPlanes();
  /// Note:
  /// This could be any 2D main window. We assume that the same nodes are visible
  /// in each 2D render window of any viewer.
  m_VisibilityTracker->SetTrackedRenderer(m_MainAxialWindow->GetRenderer());
  m_Viewer->SetEnabled(true);
  m_VisibilityTracker->SetNodesToIgnore(crossHairs);
  m_VisibilityTracker->NotifyAll();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnMainWindowOrientationChanged(MIDASOrientation mainWindowOrientation)
{
  WindowLayout windowLayout = WINDOW_LAYOUT_UNKNOWN;

  bool wasBlocked = m_LayoutWidget->blockSignals(true);

  WindowLayout defaultMultiWindowLayout = this->GetMultiWindowLayoutForOrientation(mainWindowOrientation);

  if (!m_MultiWindowRadioButton->isChecked())
  {
    windowLayout = m_SingleWindowLayouts[mainWindowOrientation];

    if (defaultMultiWindowLayout != WINDOW_LAYOUT_UNKNOWN)
    {
      m_Viewer->SetDefaultMultiWindowLayout(defaultMultiWindowLayout);
    }

    m_ControlsWidget->setEnabled(true);
    m_AxialWindowRadioButton->setEnabled(m_MainWindowOrientation != MIDAS_ORIENTATION_AXIAL);
    m_SagittalWindowRadioButton->setEnabled(m_MainWindowOrientation != MIDAS_ORIENTATION_SAGITTAL);
    m_CoronalWindowRadioButton->setEnabled(m_MainWindowOrientation != MIDAS_ORIENTATION_CORONAL);
  }
  else
  {
    windowLayout = defaultMultiWindowLayout;
  }

  m_LayoutWidget->blockSignals(wasBlocked);

  m_Viewer->SetWindowLayout(windowLayout);
  /// TODO The SetWindowLayout call should emit a signal but it does not at the moment.
  this->OnWindowLayoutChanged(m_Viewer, windowLayout);
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  const std::vector<QmitkRenderWindow*>& viewerRenderWindows = m_Viewer->GetRenderWindows();
  for (int i = 0; i < viewerRenderWindows.size(); ++i)
  {
    // If the newly focused window is in this widget, nothing to update. Stop early.
    if (focusedRenderer == viewerRenderWindows[i]->GetRenderer())
    {
      this->OnViewerWindowChanged();
      return;
    }
  }

  mitk::IRenderWindowPart* selectedEditor = this->GetSelectedEditor();
  if (selectedEditor)
  {
    foreach (QmitkRenderWindow* mainWindow, selectedEditor->GetQmitkRenderWindows().values())
    {
      if (focusedRenderer == mainWindow->GetRenderer())
      {
        this->OnMainWindowChanged(mainWindow);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnViewerWindowChanged()
{
  MIDASOrientation orientation = m_Viewer->GetOrientation();

  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    int selectedSlice = m_Viewer->GetSelectedSlice(m_Viewer->GetOrientation());
    int maxSlice = m_Viewer->GetMaxSlice(m_Viewer->GetOrientation());

    bool wasBlocked = m_SliceSpinBox->blockSignals(true);
    m_SliceSpinBox->setMaximum(maxSlice);
    m_SliceSpinBox->setValue(selectedSlice);
    m_SliceSpinBox->setEnabled(true);
    m_SliceSpinBox->blockSignals(wasBlocked);

    double magnification = m_Viewer->GetMagnification(m_Viewer->GetOrientation());
    m_Magnification = magnification;

    wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(magnification);
    m_MagnificationSpinBox->setEnabled(true);
    m_MagnificationSpinBox->blockSignals(wasBlocked);
  }
  else
  {
    bool wasBlocked = m_SliceSpinBox->blockSignals(true);
    m_SliceSpinBox->setValue(0);
    m_SliceSpinBox->setEnabled(false);
    m_SliceSpinBox->blockSignals(wasBlocked);

    m_Magnification = 0;

    wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(0.0);
    m_MagnificationSpinBox->setEnabled(false);
    m_MagnificationSpinBox->blockSignals(wasBlocked);
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnMainWindowChanged(QmitkRenderWindow* mainWindow)
{
  // Get hold of main windows, using QmitkAbstractView lookup mitkIRenderWindowPart.
  QmitkRenderWindow* mainAxialWindow = m_ContainingView->GetRenderWindow("axial");
  QmitkRenderWindow* mainSagittalWindow = m_ContainingView->GetRenderWindow("sagittal");
  QmitkRenderWindow* mainCoronalWindow = m_ContainingView->GetRenderWindow("coronal");

  if (mainWindow != mainAxialWindow
      && mainWindow != mainSagittalWindow
      && mainWindow != mainCoronalWindow)
  {
    return;
  }

  m_MainWindow = mainWindow;

  QmitkRenderWindow* axialWindow = m_Viewer->GetAxialWindow();
  QmitkRenderWindow* sagittalWindow = m_Viewer->GetSagittalWindow();
  QmitkRenderWindow* coronalWindow = m_Viewer->GetCoronalWindow();

  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  /// Note:
  /// To avoid updating the world geometry three times, we connect the geometry send event
  /// only to the axial window and assume that the three 2D renderer windows have the same
  /// world geometry.
  if (m_MainAxialWindow)
  {
    m_MainAxialSnc->Disconnect(this);
    m_MainAxialSnc->Disconnect(axialSnc);
    axialSnc->Disconnect(m_MainAxialSnc);
  }
  if (m_MainSagittalWindow)
  {
    m_MainSagittalSnc->Disconnect(sagittalSnc);
    sagittalSnc->Disconnect(m_MainSagittalSnc);
  }
  if (m_MainCoronalWindow)
  {
    m_MainCoronalSnc->Disconnect(coronalSnc);
    coronalSnc->Disconnect(m_MainCoronalSnc);
  }

  if (!mainWindow)
  {
    m_VisibilityTracker->SetTrackedRenderer(0);
    m_Viewer->SetEnabled(false);

    m_TimeGeometry = 0;

    m_MainAxialWindow = 0;
    m_MainSagittalWindow = 0;
    m_MainCoronalWindow = 0;

    m_MainAxialSnc = 0;
    m_MainSagittalSnc = 0;
    m_MainCoronalSnc = 0;

    m_MainWindowOrientation = MIDAS_ORIENTATION_UNKNOWN;

    m_Viewer->RequestUpdate();

    return;
  }

  m_MainAxialWindow = mainAxialWindow;
  m_MainSagittalWindow = mainSagittalWindow;
  m_MainCoronalWindow = mainCoronalWindow;

  m_MainAxialSnc = mainAxialWindow->GetSliceNavigationController();
  m_MainSagittalSnc = mainSagittalWindow->GetSliceNavigationController();
  m_MainCoronalSnc = mainCoronalWindow->GetSliceNavigationController();

  MIDASOrientation mainWindowOrientation;
  if (mainWindow == mainAxialWindow)
  {
    mainWindowOrientation = MIDAS_ORIENTATION_AXIAL;
  }
  else if (mainWindow == mainSagittalWindow)
  {
    mainWindowOrientation = MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (mainWindow == mainCoronalWindow)
  {
    mainWindowOrientation = MIDAS_ORIENTATION_CORONAL;
  }
  else
  {
    mainWindowOrientation = MIDAS_ORIENTATION_UNKNOWN;
  }

  mitk::TimeGeometry* timeGeometry = const_cast<mitk::TimeGeometry*>(mainWindow->GetRenderer()->GetTimeWorldGeometry());

  /// Note:
  /// The SetWindowLayout function does not change the layout if the viewer does not have
  /// a valid geometry. Therefore, if the viewer is initialised with a geometry for the
  /// first time, we need to set the window layout again, according to the main window
  /// orientation.
  bool geometryFirstInitialised = false;
  if (timeGeometry != m_TimeGeometry)
  {
    if (!m_TimeGeometry)
    {
      geometryFirstInitialised = true;
    }
    m_TimeGeometry = timeGeometry;
    this->OnMainWindowGeometryChanged(timeGeometry);
  }

  if (mainWindowOrientation != m_MainWindowOrientation || geometryFirstInitialised)
  {
    m_MainWindowOrientation = mainWindowOrientation;
    this->OnMainWindowOrientationChanged(mainWindowOrientation);
  }

  /// Note:
  /// To avoid updating the world geometry three times, we connect the geometry send event
  /// only to the axial window and assume that the three 2D renderer windows have the same
  /// world geometry.
  if (m_MainAxialWindow)
  {
    m_MainAxialSnc->ConnectGeometrySendEvent(this);
    m_MainAxialSnc->ConnectGeometryEvents(axialSnc);
    axialSnc->ConnectGeometryEvents(m_MainAxialSnc);
    this->connect(m_MainAxialWindow, SIGNAL(destroyed(QObject*)), SLOT(OnAMainWindowDestroyed(QObject*)));
  }
  if (mainSagittalWindow)
  {
    m_MainSagittalSnc->ConnectGeometryEvents(sagittalSnc);
    sagittalSnc->ConnectGeometryEvents(m_MainSagittalSnc);
    this->connect(m_MainSagittalWindow, SIGNAL(destroyed(QObject*)), SLOT(OnAMainWindowDestroyed(QObject*)));
  }
  if (mainCoronalWindow)
  {
    m_MainCoronalSnc->ConnectGeometryEvents(coronalSnc);
    coronalSnc->ConnectGeometryEvents(m_MainCoronalSnc);
    this->connect(m_MainCoronalWindow, SIGNAL(destroyed(QObject*)), SLOT(OnAMainWindowDestroyed(QObject*)));
  }

  /// Note that changing the window layout resets the geometry, what sets the selected position in the centre.
  /// Therefore, we resend the main window position here.
  m_MainAxialWindow->GetSliceNavigationController()->SendSlice();
  m_MainSagittalWindow->GetSliceNavigationController()->SendSlice();
  m_MainCoronalWindow->GetSliceNavigationController()->SendSlice();

  m_Viewer->RequestUpdate();
}


//-----------------------------------------------------------------------------
mitk::IRenderWindowPart* QmitkSideViewerWidget::GetSelectedEditor()
{
  berry::IWorkbenchPage::Pointer page = m_ContainingView->GetSite()->GetPage();

  // Return the active editor if it implements mitk::IRenderWindowPart
  mitk::IRenderWindowPart* renderPart =
      dynamic_cast<mitk::IRenderWindowPart*>(page->GetActiveEditor().GetPointer());

  if (!renderPart)
  {
    // No suitable active editor found, check visible editors
    std::list<berry::IEditorReference::Pointer> editors = page->GetEditorReferences();
    std::list<berry::IEditorReference::Pointer>::iterator editorsIt = editors.begin();
    std::list<berry::IEditorReference::Pointer>::iterator editorsEnd = editors.end();
    for ( ; editorsIt != editorsEnd; ++editorsIt)
    {
      berry::IWorkbenchPart::Pointer part = (*editorsIt)->GetPart(false);
      if (page->IsPartVisible(part))
      {
        renderPart = dynamic_cast<mitk::IRenderWindowPart*>(part.GetPointer());
        break;
      }
    }
  }

  return renderPart;
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnSelectedPositionChanged(niftkSingleViewerWidget* viewer, const mitk::Point3D& selectedPosition)
{
  MIDASOrientation orientation = m_Viewer->GetOrientation();
  if (orientation != MIDAS_ORIENTATION_UNKNOWN)
  {
    bool wasBlocked = m_SliceSpinBox->blockSignals(true);
    m_SliceSpinBox->setValue(m_Viewer->GetSelectedSlice(orientation));
    m_SliceSpinBox->blockSignals(wasBlocked);
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnScaleFactorChanged(niftkSingleViewerWidget*, MIDASOrientation orientation, double scaleFactor)
{
  double magnification = m_Viewer->GetMagnification(m_Viewer->GetOrientation());

  bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
  m_MagnificationSpinBox->setValue(magnification);
  m_MagnificationSpinBox->blockSignals(wasBlocked);

  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnWindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout windowLayout)
{
  bool axialWindowRadioButtonWasBlocked = m_AxialWindowRadioButton->blockSignals(true);
  bool sagittalWindowRadioButtonWasBlocked = m_SagittalWindowRadioButton->blockSignals(true);
  bool coronalWindowRadioButtonWasBlocked = m_CoronalWindowRadioButton->blockSignals(true);
  bool multiWindowRadioButtonWasBlocked = m_MultiWindowRadioButton->blockSignals(true);
  bool multiWindowComboBoxWasBlocked = m_MultiWindowComboBox->blockSignals(true);

  if (windowLayout == WINDOW_LAYOUT_AXIAL)
  {
    m_AxialWindowRadioButton->setChecked(true);
  }
  else if (windowLayout == WINDOW_LAYOUT_SAGITTAL)
  {
    m_SagittalWindowRadioButton->setChecked(true);
  }
  else if (windowLayout == WINDOW_LAYOUT_CORONAL)
  {
    m_CoronalWindowRadioButton->setChecked(true);
  }
  else if (::IsMultiWindowLayout(windowLayout))
  {
    m_MultiWindowRadioButton->setChecked(true);
    if (windowLayout == WINDOW_LAYOUT_COR_AX_H
        || windowLayout == WINDOW_LAYOUT_COR_SAG_H
        || windowLayout == WINDOW_LAYOUT_SAG_AX_H)
    {
      m_MultiWindowComboBox->setCurrentIndex(0);
    }
    else if (windowLayout == WINDOW_LAYOUT_COR_AX_V
        || windowLayout == WINDOW_LAYOUT_COR_SAG_V
        || windowLayout == WINDOW_LAYOUT_SAG_AX_V)
    {
      m_MultiWindowComboBox->setCurrentIndex(1);
    }
    else if (windowLayout == WINDOW_LAYOUT_ORTHO)
    {
      m_MultiWindowComboBox->setCurrentIndex(2);
    }
  }
  else
  {
    m_AxialWindowRadioButton->setChecked(false);
    m_SagittalWindowRadioButton->setChecked(false);
    m_CoronalWindowRadioButton->setChecked(false);
    m_MultiWindowRadioButton->setChecked(false);
  }

  m_AxialWindowRadioButton->blockSignals(axialWindowRadioButtonWasBlocked);
  m_SagittalWindowRadioButton->blockSignals(sagittalWindowRadioButtonWasBlocked);
  m_CoronalWindowRadioButton->blockSignals(coronalWindowRadioButtonWasBlocked);
  m_MultiWindowRadioButton->blockSignals(multiWindowRadioButtonWasBlocked);
  m_MultiWindowComboBox->blockSignals(multiWindowComboBoxWasBlocked);

  m_WindowLayout = windowLayout;

  /// Note:
  /// The selected window has not necessarily been changed, but it is not costly
  /// to refresh the GUI buttons every time.
  this->OnViewerWindowChanged();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnSliceSpinBoxValueChanged(int slice)
{
  m_Viewer->SetSelectedSlice(m_Viewer->GetOrientation(), slice);
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::OnMagnificationSpinBoxValueChanged(double magnification)
{
  double roundedMagnification = std::floor(magnification);

  // If we are between two integers, we raise a new event:
  if (magnification != roundedMagnification)
  {
    double newMagnification = roundedMagnification;
    // If the value has decreased, we have to increase the rounded value.
    if (magnification < m_Magnification)
    {
      newMagnification += 1.0;
    }

    m_MagnificationSpinBox->setValue(newMagnification);
  }
  else
  {
    m_Viewer->SetMagnification(m_Viewer->GetOrientation(), magnification);
    m_Magnification = magnification;
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerWidget::SetGeometry(const itk::EventObject& geometrySendEvent)
{
  const mitk::SliceNavigationController::GeometrySendEvent* sendEvent =
      dynamic_cast<const mitk::SliceNavigationController::GeometrySendEvent *>(&geometrySendEvent);

  assert(sendEvent);

  this->OnMainWindowGeometryChanged(sendEvent->GetTimeGeometry());
}
