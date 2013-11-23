/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASSegmentationViewWidget.h"
#include "QmitkMIDASBaseSegmentationFunctionality.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <mitkDataStorage.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>
#include <mitkGeometry3D.h>
#include <mitkSliceNavigationController.h>
#include <mitkBaseRenderer.h>
#include <itkCommand.h>


//-----------------------------------------------------------------------------
QmitkMIDASSegmentationViewWidget::QmitkMIDASSegmentationViewWidget(QWidget* parent)
: m_ContainingFunctionality(NULL)
, m_FocusManagerObserverTag(0)
, m_WindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_MainWindowLayout(WINDOW_LAYOUT_UNKNOWN)
, m_MainAxialWindow(NULL)
, m_MainSagittalWindow(NULL)
, m_MainCoronalWindow(NULL)
, m_Main3DWindow(NULL)
, m_MainAxialSnc(0)
, m_MainSagittalSnc(0)
, m_MainCoronalSnc(0)
, m_Renderer(NULL)
, m_NodeAddedSetter(NULL)
, m_VisibilityTracker(NULL)
, m_Magnification(0.0)
, m_SingleWindowLayouts()
{
  this->setupUi(parent);

  m_Viewer->SetSelected(false);

  m_MultiWindowComboBox->addItem("2H");
  m_MultiWindowComboBox->addItem("2V");
  m_MultiWindowComboBox->addItem("2x2");

  m_CoronalWindowRadioButton->setChecked(true);

  m_SingleWindowLayouts[WINDOW_LAYOUT_AXIAL] = WINDOW_LAYOUT_CORONAL;
  m_SingleWindowLayouts[WINDOW_LAYOUT_SAGITTAL] = WINDOW_LAYOUT_CORONAL;
  m_SingleWindowLayouts[WINDOW_LAYOUT_CORONAL] = WINDOW_LAYOUT_SAGITTAL;

  this->ChangeLayout();

  m_MagnificationSpinBox->setDecimals(2);
  m_MagnificationSpinBox->setSingleStep(1.0);

  double minMagnification = std::ceil(m_Viewer->GetMinMagnification());
  double maxMagnification = std::floor(m_Viewer->GetMaxMagnification());

  m_MagnificationSpinBox->setMinimum(minMagnification);
  m_MagnificationSpinBox->setMaximum(maxMagnification);

  m_ControlsWidget->setEnabled(false);

  std::vector<mitk::BaseRenderer*> renderers;
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
  m_VisibilityTracker->SetRenderersToUpdate(renderers);

  m_Viewer->SetCursorGloballyVisible(false);
  m_Viewer->SetCursorVisible(true);
  m_Viewer->SetRememberSettingsPerWindowLayout(true);
  m_Viewer->SetDisplayInteractionsEnabled(true);
  m_Viewer->SetCursorPositionsBound(false);
  m_Viewer->SetScaleFactorsBound(true);

  this->connect(m_AxialWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnAxialWindowRadioButtonToggled(bool)));
  this->connect(m_SagittalWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnSagittalWindowRadioButtonToggled(bool)));
  this->connect(m_CoronalWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnCoronalWindowRadioButtonToggled(bool)));
  this->connect(m_MultiWindowRadioButton, SIGNAL(toggled(bool)), SLOT(OnMultiWindowRadioButtonToggled(bool)));
  this->connect(m_MultiWindowComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnMultiWindowComboBoxIndexChanged()));

  this->connect(m_MagnificationSpinBox, SIGNAL(valueChanged(double)), SLOT(OnMagnificationChanged(double)));
  this->connect(m_Viewer, SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, double)), SLOT(OnScaleFactorChanged(niftkSingleViewerWidget*, double)));
}


//-----------------------------------------------------------------------------
QmitkMIDASSegmentationViewWidget::~QmitkMIDASSegmentationViewWidget()
{
  // m_NodeAddedSetter deleted by smart pointer.
  // m_VisibilityTracker deleted by smart pointer.
  this->Deactivated();
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (storage != NULL)
  {
    m_Viewer->SetDataStorage(storage);

    m_NodeAddedSetter->SetDataStorage(storage);
    m_VisibilityTracker->SetDataStorage(storage);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetContainingFunctionality(QmitkMIDASBaseSegmentationFunctionality* functionality)
{
  m_ContainingFunctionality = functionality;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::Activated()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    itk::SimpleMemberCommand<QmitkMIDASSegmentationViewWidget>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<QmitkMIDASSegmentationViewWidget>::New();
    onFocusChangedCommand->SetCallbackFunction( this, &QmitkMIDASSegmentationViewWidget::OnFocusChanged );

    m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);

    // Force this because:
    // If user launches Drag and Drop Display, loads image, drops into window, and THEN launches this widget,
    // you don't get the first OnFocusChanged as the window has already been clicked on, and then the
    // geometry is initialised incorrectly.
    this->OnFocusChanged();

    m_Viewer->SetSelected(false);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::Deactivated()
{
  if (m_MainAxialSnc)
  {
    mitk::SliceNavigationController* axialSnc = m_Viewer->GetAxialWindow()->GetSliceNavigationController();
    axialSnc->Disconnect(m_MainAxialSnc);
    m_MainAxialSnc->Disconnect(axialSnc);
    m_MainAxialWindow = 0;
    m_MainAxialSnc = 0;
  }
  if (m_MainSagittalSnc)
  {
    mitk::SliceNavigationController* sagittalSnc = m_Viewer->GetSagittalWindow()->GetSliceNavigationController();
    sagittalSnc->Disconnect(m_MainSagittalSnc);
    m_MainSagittalSnc->Disconnect(sagittalSnc);
    m_MainSagittalWindow = 0;
    m_MainSagittalSnc = 0;
  }
  if (m_MainCoronalSnc)
  {
    mitk::SliceNavigationController* coronalSnc = m_Viewer->GetCoronalWindow()->GetSliceNavigationController();
    coronalSnc->Disconnect(m_MainCoronalSnc);
    m_MainCoronalSnc->Disconnect(coronalSnc);
    m_MainCoronalWindow = 0;
    m_MainCoronalSnc = 0;
  }

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }

  m_Viewer->SetEnabled(false);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnAMainWindowDestroyed(QObject* mainWindow)
{
  if (mainWindow == m_MainAxialWindow)
  {
    mitk::SliceNavigationController* axialSnc = m_Viewer->GetAxialWindow()->GetSliceNavigationController();
    axialSnc->Disconnect(m_MainAxialSnc);
    m_MainAxialWindow = 0;
    m_MainAxialSnc = 0;
  }
  else if (mainWindow == m_MainSagittalWindow)
  {
    mitk::SliceNavigationController* sagittalSnc = m_Viewer->GetSagittalWindow()->GetSliceNavigationController();
    sagittalSnc->Disconnect(m_MainSagittalSnc);
    m_MainSagittalWindow = 0;
    m_MainSagittalSnc = 0;
  }
  else if (mainWindow == m_MainCoronalWindow)
  {
    mitk::SliceNavigationController* coronalSnc = m_Viewer->GetCoronalWindow()->GetSliceNavigationController();
    coronalSnc->Disconnect(m_MainCoronalSnc);
    m_MainCoronalWindow = 0;
    m_MainCoronalSnc = 0;
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetEnabled(bool enabled)
{
  m_ControlsWidget->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnAxialWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = WINDOW_LAYOUT_AXIAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = WINDOW_LAYOUT_SAGITTAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = WINDOW_LAYOUT_CORONAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnMultiWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnMultiWindowComboBoxIndexChanged()
{
  m_MultiWindowRadioButton->setChecked(true);
  this->ChangeLayout();
}

//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::ChangeLayout()
{
  WindowLayout nextLayout = WINDOW_LAYOUT_UNKNOWN;

  bool wasBlocked = m_LayoutWidget->blockSignals(true);

  if (m_MultiWindowRadioButton->isChecked())
  {
    // 2H
    if (m_MultiWindowComboBox->currentIndex() == 0)
    {
      if (m_MainWindowLayout == WINDOW_LAYOUT_AXIAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_SAG_H;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_SAGITTAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_AX_H;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_CORONAL)
      {
        nextLayout = WINDOW_LAYOUT_SAG_AX_H;
      }
    }
    // 2V
    else if (m_MultiWindowComboBox->currentIndex() == 1)
    {
      if (m_MainWindowLayout == WINDOW_LAYOUT_AXIAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_SAG_V;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_SAGITTAL)
      {
        nextLayout = WINDOW_LAYOUT_COR_AX_V;
      }
      else if (m_MainWindowLayout == WINDOW_LAYOUT_CORONAL)
      {
        nextLayout = WINDOW_LAYOUT_SAG_AX_V;
      }
    }
    // 2x2
    else if (m_MultiWindowComboBox->currentIndex() == 2)
    {
      nextLayout = WINDOW_LAYOUT_ORTHO;
    }
  }
  else if (::IsSingleWindowLayout(m_MainWindowLayout) && m_MainWindowLayout != WINDOW_LAYOUT_3D)
  {
    nextLayout = m_SingleWindowLayouts[m_MainWindowLayout];

    QRadioButton* nextLayoutRadioButton = 0;
    if (nextLayout == WINDOW_LAYOUT_AXIAL && !m_AxialWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_AxialWindowRadioButton;
    }
    else if (nextLayout == WINDOW_LAYOUT_SAGITTAL && !m_SagittalWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_SagittalWindowRadioButton;
    }
    if (nextLayout == WINDOW_LAYOUT_CORONAL && !m_CoronalWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_CoronalWindowRadioButton;
    }

    if (nextLayoutRadioButton)
    {
      nextLayoutRadioButton->setChecked(true);
    }
  }

  if (!m_MultiWindowRadioButton->isChecked())
  {
    m_ControlsWidget->setEnabled(true);
    m_AxialWindowRadioButton->setEnabled(m_MainWindowLayout != WINDOW_LAYOUT_AXIAL);
    m_SagittalWindowRadioButton->setEnabled(m_MainWindowLayout != WINDOW_LAYOUT_SAGITTAL);
    m_CoronalWindowRadioButton->setEnabled(m_MainWindowLayout != WINDOW_LAYOUT_CORONAL);
  }

  m_LayoutWidget->blockSignals(wasBlocked);

  if (nextLayout != WINDOW_LAYOUT_UNKNOWN && nextLayout != m_WindowLayout)
  {
    m_WindowLayout = nextLayout;
    m_Viewer->SetWindowLayout(m_WindowLayout);

    double magnification = m_Viewer->GetMagnification();

    bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(magnification);
    m_MagnificationSpinBox->blockSignals(wasBlocked);

    emit LayoutChanged(m_WindowLayout);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  QmitkRenderWindow* renderWindow = m_Viewer->GetRenderWindow(focusedRenderer->GetRenderWindow());

  // If the newly focused window is this widget, nothing to update. Stop early.
  if (renderWindow)
  {
    m_Viewer->SetSelectedRenderWindow(renderWindow);

    double magnification = m_Viewer->GetMagnification();
    m_Magnification = magnification;

    bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(magnification);
    m_MagnificationSpinBox->blockSignals(wasBlocked);

    return;
  }

  // Get hold of main windows, using QmitkAbstractView lookup mitkIRenderWindowPart.
  QmitkRenderWindow* mainAxialWindow = m_ContainingFunctionality->GetRenderWindow("axial");
  QmitkRenderWindow* mainSagittalWindow = m_ContainingFunctionality->GetRenderWindow("sagittal");
  QmitkRenderWindow* mainCoronalWindow = m_ContainingFunctionality->GetRenderWindow("coronal");
  QmitkRenderWindow* main3DWindow = m_ContainingFunctionality->GetRenderWindow("3d");

  // Main windows could be NULL if main window not initialised,
  // or no valid QmitkRenderer returned from mitkIRenderWindowPart.
  if (   mainAxialWindow == NULL
      || mainSagittalWindow == NULL
      || mainCoronalWindow == NULL
      || main3DWindow == NULL
      )
  {
    return;
  }

  // Check if the user selected a completely different main window widget, or
  // if the user selected a different layout (axial, coronal, sagittal) within
  // the same DnDMultiWindowWidget.
  bool mainWindowChanged = false;
  if (   mainAxialWindow    != m_MainAxialWindow
      || mainSagittalWindow != m_MainSagittalWindow
      || mainCoronalWindow  != m_MainCoronalWindow
      || main3DWindow       != m_Main3DWindow
      )
  {
    mainWindowChanged = true;
  }

  if (mainWindowChanged)
  {
    mitk::SliceNavigationController* axialSnc = m_Viewer->GetAxialWindow()->GetSliceNavigationController();
    mitk::SliceNavigationController* sagittalSnc = m_Viewer->GetSagittalWindow()->GetSliceNavigationController();
    mitk::SliceNavigationController* coronalSnc = m_Viewer->GetCoronalWindow()->GetSliceNavigationController();

    // If there was a main window then disconnect from it.
    if (m_MainAxialSnc)
    {
      axialSnc->Disconnect(m_MainAxialSnc);
      m_MainAxialSnc->Disconnect(axialSnc);
    }
    if (m_MainSagittalSnc)
    {
      sagittalSnc->Disconnect(m_MainSagittalSnc);
      m_MainSagittalSnc->Disconnect(sagittalSnc);
    }
    if (m_MainCoronalSnc)
    {
      coronalSnc->Disconnect(m_MainCoronalSnc);
      m_MainCoronalSnc->Disconnect(coronalSnc);
    }

    // If there is a new main window then connect to it.
    mitk::SliceNavigationController* mainAxialSnc = mainAxialWindow->GetSliceNavigationController();
    mitk::SliceNavigationController* mainSagittalSnc = mainSagittalWindow->GetSliceNavigationController();
    mitk::SliceNavigationController* mainCoronalSnc = mainCoronalWindow->GetSliceNavigationController();
    if (mainAxialSnc)
    {
      axialSnc->ConnectGeometryEvents(mainAxialSnc);
      mainAxialSnc->ConnectGeometryEvents(axialSnc);
    }
    if (mainSagittalSnc)
    {
      sagittalSnc->ConnectGeometryEvents(mainSagittalSnc);
      mainSagittalSnc->ConnectGeometryEvents(sagittalSnc);
    }
    if (mainCoronalSnc)
    {
      coronalSnc->ConnectGeometryEvents(mainCoronalSnc);
      mainCoronalSnc->ConnectGeometryEvents(coronalSnc);
    }
  }

  // This will only be valid if we are not currently focused on THIS widget.
  // This should always be true at this point due to early exit above.
  WindowLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

  bool mainWindowLayoutChanged = false;
  if (mainWindowLayout != WINDOW_LAYOUT_UNKNOWN && mainWindowLayout != m_MainWindowLayout)
  {
    mainWindowLayoutChanged = true;
    m_MainWindowLayout = mainWindowLayout;
  }

  if (mainWindowChanged || m_Renderer == NULL || (mainWindowLayout != WINDOW_LAYOUT_UNKNOWN && m_WindowLayout == WINDOW_LAYOUT_UNKNOWN))
  {
    const mitk::TimeGeometry* worldTimeGeometry = mainAxialWindow->GetRenderer()->GetTimeWorldGeometry();
    if (worldTimeGeometry)
    {
      mitk::TimeGeometry::Pointer timeGeometry = const_cast<mitk::TimeGeometry*>(worldTimeGeometry);
      assert(timeGeometry);

      m_MainAxialWindow = mainAxialWindow;
      m_MainSagittalWindow = mainSagittalWindow;
      m_MainCoronalWindow = mainCoronalWindow;
      m_Main3DWindow = main3DWindow;

      m_MainAxialSnc = mainAxialWindow->GetSliceNavigationController();
      m_MainSagittalSnc = mainSagittalWindow->GetSliceNavigationController();
      m_MainCoronalSnc = mainCoronalWindow->GetSliceNavigationController();

      // Note:
      // We have to disconnect from the main window SNCs when the main windows are destroyed,
      // and also have to set the data members that store them and their SNCs to 0.
      this->connect(mainAxialWindow, SIGNAL(destroyed(QObject*)), SLOT(OnAMainWindowDestroyed(QObject*)));
      this->connect(mainSagittalWindow, SIGNAL(destroyed(QObject*)), SLOT(OnAMainWindowDestroyed(QObject*)));
      this->connect(mainCoronalWindow, SIGNAL(destroyed(QObject*)), SLOT(OnAMainWindowDestroyed(QObject*)));

      m_Viewer->SetGeometry(timeGeometry);
      m_Viewer->SetBoundGeometryActive(false);
      m_Viewer->SetShow3DWindowIn2x2WindowLayout(true);
      if (!m_Viewer->IsEnabled())
      {
        m_Viewer->SetEnabled(true);
      }

      std::vector<mitk::DataNode*> crossHairs = m_Viewer->GetWidgetPlanes();
      std::vector<mitk::BaseRenderer*> renderersToTrack;
      renderersToTrack.push_back(mainAxialWindow->GetRenderer());
      renderersToTrack.push_back(mainSagittalWindow->GetRenderer());
      renderersToTrack.push_back(mainCoronalWindow->GetRenderer());

      m_VisibilityTracker->SetRenderersToTrack(renderersToTrack);
      m_VisibilityTracker->SetNodesToIgnore(crossHairs);
      m_VisibilityTracker->OnPropertyChanged(); // force update

      m_Viewer->FitToDisplay();
      this->ChangeLayout();
    }
  }
  else if (mainWindowLayoutChanged)
  {
    this->ChangeLayout();
  }

  m_Renderer = focusedRenderer;

  m_Viewer->RequestUpdate();
}


//-----------------------------------------------------------------------------
MIDASOrientation QmitkMIDASSegmentationViewWidget::GetCurrentMainWindowOrientation()
{
  MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  if (focusedRenderer != NULL)
  {
    QmitkRenderWindow* mainWindowAxial = m_ContainingFunctionality->GetRenderWindow("axial");
    QmitkRenderWindow* mainWindowSagittal = m_ContainingFunctionality->GetRenderWindow("sagittal");
    QmitkRenderWindow* mainWindowCoronal = m_ContainingFunctionality->GetRenderWindow("coronal");

    if (mainWindowAxial && mainWindowSagittal && mainWindowCoronal)
    {
      if (focusedRenderer == mainWindowAxial->GetRenderer())
      {
        orientation = MIDAS_ORIENTATION_AXIAL;
      }
      else if (focusedRenderer == mainWindowSagittal->GetRenderer())
      {
        orientation = MIDAS_ORIENTATION_SAGITTAL;
      }
      else if (focusedRenderer == mainWindowCoronal->GetRenderer())
      {
        orientation = MIDAS_ORIENTATION_CORONAL;
      }
    }
  }
  return orientation;
}


//-----------------------------------------------------------------------------
WindowLayout QmitkMIDASSegmentationViewWidget::GetCurrentMainWindowLayout()
{
  WindowLayout mainWindowLayout = WINDOW_LAYOUT_UNKNOWN;
  MIDASOrientation mainWindowOrientation = this->GetCurrentMainWindowOrientation();

  if (mainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
  {
    mainWindowLayout = WINDOW_LAYOUT_AXIAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    mainWindowLayout = WINDOW_LAYOUT_SAGITTAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
  {
    mainWindowLayout = WINDOW_LAYOUT_CORONAL;
  }
  return mainWindowLayout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnScaleFactorChanged(niftkSingleViewerWidget*, double scaleFactor)
{
  double magnification = m_Viewer->GetMagnification();

  bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
  m_MagnificationSpinBox->setValue(magnification);
  m_MagnificationSpinBox->blockSignals(wasBlocked);

  m_Magnification = magnification;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnMagnificationChanged(double magnification)
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
    m_Viewer->SetMagnification(magnification);
    m_Magnification = magnification;
  }
}
