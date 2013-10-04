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
#include <QmitkMIDASSingleViewWidgetListVisibilityManager.h>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpacerItem>
#include <mitkDataStorage.h>
#include <mitkTimeSlicedGeometry.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>
#include <mitkGeometry3D.h>
#include <mitkTimeSlicedGeometry.h>
#include <mitkSliceNavigationController.h>
#include <mitkBaseRenderer.h>
#include <itkCommand.h>
#include <mitkMIDASDataNodeNameStringFilter.h>


//-----------------------------------------------------------------------------
QmitkMIDASSegmentationViewWidget::QmitkMIDASSegmentationViewWidget(QWidget* parent)
: m_ContainingFunctionality(NULL)
, m_FocusManagerObserverTag(0)
, m_Layout(MIDAS_LAYOUT_UNKNOWN)
, m_MainWindowLayout(MIDAS_LAYOUT_UNKNOWN)
, m_MainWindowAxial(NULL)
, m_MainWindowSagittal(NULL)
, m_MainWindowCoronal(NULL)
, m_MainWindow3d(NULL)
, m_CurrentRenderer(NULL)
, m_NodeAddedSetter(NULL)
, m_VisibilityTracker(NULL)
, m_Magnification(0.0)
, m_SingleWindowLayouts()
{
  this->setupUi(parent);

  m_ViewerWidget->SetSelected(false);

  m_MultiWindowComboBox->addItem("2H");
  m_MultiWindowComboBox->addItem("2V");
  m_MultiWindowComboBox->addItem("2x2");

  m_CoronalWindowRadioButton->setChecked(true);

  m_SingleWindowLayouts[MIDAS_LAYOUT_AXIAL] = MIDAS_LAYOUT_CORONAL;
  m_SingleWindowLayouts[MIDAS_LAYOUT_SAGITTAL] = MIDAS_LAYOUT_CORONAL;
  m_SingleWindowLayouts[MIDAS_LAYOUT_CORONAL] = MIDAS_LAYOUT_SAGITTAL;

  this->ChangeLayout();

  m_MagnificationSpinBox->setDecimals(2);
  m_MagnificationSpinBox->setSingleStep(1.0);

  double minMagnification = std::ceil(m_ViewerWidget->GetMinMagnification());
  double maxMagnification = std::floor(m_ViewerWidget->GetMaxMagnification());

  m_MagnificationSpinBox->setMinimum(minMagnification);
  m_MagnificationSpinBox->setMaximum(maxMagnification);

  m_ControlsWidget->setEnabled(false);

  std::vector<mitk::BaseRenderer*> renderers;
  renderers.push_back(m_ViewerWidget->GetAxialWindow()->GetRenderer());
  renderers.push_back(m_ViewerWidget->GetSagittalWindow()->GetRenderer());
  renderers.push_back(m_ViewerWidget->GetCoronalWindow()->GetRenderer());

  m_NodeAddedSetter = mitk::DataNodeAddedVisibilitySetter::New();
  mitk::MIDASDataNodeNameStringFilter::Pointer filter = mitk::MIDASDataNodeNameStringFilter::New();
  m_NodeAddedSetter->AddFilter(filter.GetPointer());
  m_NodeAddedSetter->SetRenderers(renderers);
  m_NodeAddedSetter->SetVisibility(false);

  m_VisibilityTracker = mitk::DataStorageVisibilityTracker::New();
  m_VisibilityTracker->SetRenderersToUpdate(renderers);

  m_ViewerWidget->SetDisplay2DCursorsLocally(true);
  m_ViewerWidget->SetRememberSettingsPerLayout(true);
  m_ViewerWidget->SetDisplayInteractionsEnabled(true);
  m_ViewerWidget->SetCursorPositionsBound(false);
  m_ViewerWidget->SetScaleFactorsBound(true);

  connect(m_AxialWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnAxialWindowRadioButtonToggled(bool)));
  connect(m_SagittalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnSagittalWindowRadioButtonToggled(bool)));
  connect(m_CoronalWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnCoronalWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnMultiWindowRadioButtonToggled(bool)));
  connect(m_MultiWindowComboBox, SIGNAL(currentIndexChanged(int)), SLOT(OnMultiWindowComboBoxIndexChanged()));

  connect(m_MagnificationSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnMagnificationChanged(double)));
  connect(m_ViewerWidget, SIGNAL(ScaleFactorChanged(QmitkMIDASSingleViewWidget*, double)), this, SLOT(OnScaleFactorChanged(QmitkMIDASSingleViewWidget*, double)));
}


//-----------------------------------------------------------------------------
QmitkMIDASSegmentationViewWidget::~QmitkMIDASSegmentationViewWidget()
{
  // m_NodeAddedSetter deleted by smart pointer.
  // m_VisibilityTracker deleted by smart pointer.
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (storage != NULL)
  {
    m_ViewerWidget->SetDataStorage(storage);

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
    m_ViewerWidget->SetSelected(false);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::Deactivated()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }
  m_ViewerWidget->SetEnabled(false);
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
    MIDASLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = MIDAS_LAYOUT_AXIAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnSagittalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    MIDASLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = MIDAS_LAYOUT_SAGITTAL;
    }
    this->ChangeLayout();
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnCoronalWindowRadioButtonToggled(bool checked)
{
  if (checked)
  {
    MIDASLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

    if (::IsSingleWindowLayout(mainWindowLayout))
    {
      m_SingleWindowLayouts[mainWindowLayout] = MIDAS_LAYOUT_CORONAL;
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
  MIDASLayout nextLayout = MIDAS_LAYOUT_UNKNOWN;

  bool wasBlocked = m_LayoutWidget->blockSignals(true);

  if (m_MultiWindowRadioButton->isChecked())
  {
    // 2H
    if (m_MultiWindowComboBox->currentIndex() == 0)
    {
      if (m_MainWindowLayout == MIDAS_LAYOUT_AXIAL)
      {
        nextLayout = MIDAS_LAYOUT_COR_SAG_H;
      }
      else if (m_MainWindowLayout == MIDAS_LAYOUT_SAGITTAL)
      {
        nextLayout = MIDAS_LAYOUT_COR_AX_H;
      }
      else if (m_MainWindowLayout == MIDAS_LAYOUT_CORONAL)
      {
        nextLayout = MIDAS_LAYOUT_SAG_AX_H;
      }
    }
    // 2V
    else if (m_MultiWindowComboBox->currentIndex() == 1)
    {
      if (m_MainWindowLayout == MIDAS_LAYOUT_AXIAL)
      {
        nextLayout = MIDAS_LAYOUT_COR_SAG_V;
      }
      else if (m_MainWindowLayout == MIDAS_LAYOUT_SAGITTAL)
      {
        nextLayout = MIDAS_LAYOUT_COR_AX_V;
      }
      else if (m_MainWindowLayout == MIDAS_LAYOUT_CORONAL)
      {
        nextLayout = MIDAS_LAYOUT_SAG_AX_V;
      }
    }
    // 2x2
    else if (m_MultiWindowComboBox->currentIndex() == 2)
    {
      nextLayout = MIDAS_LAYOUT_ORTHO;
    }
  }
  else if (::IsSingleWindowLayout(m_MainWindowLayout) && m_MainWindowLayout != MIDAS_LAYOUT_3D)
  {
    nextLayout = m_SingleWindowLayouts[m_MainWindowLayout];

    QRadioButton* nextLayoutRadioButton = 0;
    if (nextLayout == MIDAS_LAYOUT_AXIAL && !m_AxialWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_AxialWindowRadioButton;
    }
    else if (nextLayout == MIDAS_LAYOUT_SAGITTAL && !m_SagittalWindowRadioButton->isChecked())
    {
      nextLayoutRadioButton = m_SagittalWindowRadioButton;
    }
    if (nextLayout == MIDAS_LAYOUT_CORONAL && !m_CoronalWindowRadioButton->isChecked())
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
    m_AxialWindowRadioButton->setEnabled(m_MainWindowLayout != MIDAS_LAYOUT_AXIAL);
    m_SagittalWindowRadioButton->setEnabled(m_MainWindowLayout != MIDAS_LAYOUT_SAGITTAL);
    m_CoronalWindowRadioButton->setEnabled(m_MainWindowLayout != MIDAS_LAYOUT_CORONAL);
  }

  m_LayoutWidget->blockSignals(wasBlocked);

  if (nextLayout != MIDAS_LAYOUT_UNKNOWN && nextLayout != m_Layout)
  {
    m_Layout = nextLayout;
    m_ViewerWidget->SetLayout(m_Layout);

    double magnification = m_ViewerWidget->GetMagnification();

    bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(magnification);
    m_MagnificationSpinBox->blockSignals(wasBlocked);

    emit LayoutChanged(m_Layout);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  QmitkRenderWindow* renderWindow = m_ViewerWidget->GetRenderWindow(focusedRenderer->GetRenderWindow());

  // If the newly focused window is this widget, nothing to update. Stop early.
  if (renderWindow)
  {
    m_ViewerWidget->SetSelectedRenderWindow(renderWindow);

    double magnification = m_ViewerWidget->GetMagnification();
    m_Magnification = magnification;

    bool wasBlocked = m_MagnificationSpinBox->blockSignals(true);
    m_MagnificationSpinBox->setValue(magnification);
    m_MagnificationSpinBox->blockSignals(wasBlocked);

    return;
  }

  // Get hold of main windows, using QmitkAbstractView lookup mitkIRenderWindowPart.
  QmitkRenderWindow* mainWindowAxial = m_ContainingFunctionality->GetRenderWindow("axial");
  QmitkRenderWindow* mainWindowSagittal = m_ContainingFunctionality->GetRenderWindow("sagittal");
  QmitkRenderWindow* mainWindowCoronal = m_ContainingFunctionality->GetRenderWindow("coronal");
  QmitkRenderWindow* mainWindow3d = m_ContainingFunctionality->GetRenderWindow("3d");

  // Main windows could be NULL if main window not initialised,
  // or no valid QmitkRenderer returned from mitkIRenderWindowPart.
  if (   mainWindowAxial == NULL
      || mainWindowSagittal == NULL
      || mainWindowCoronal == NULL
      || mainWindow3d == NULL
      )
  {
    return;
  }

  // Check if the user selected a completely different main window widget, or
  // if the user selected a different layout (axial, coronal, sagittal) within
  // the same QmitkMIDASStdMultiWidget.
  bool mainWindowChanged = false;
  if (   mainWindowAxial    != m_MainWindowAxial
      || mainWindowSagittal != m_MainWindowSagittal
      || mainWindowCoronal  != m_MainWindowCoronal
      || mainWindow3d       != m_MainWindow3d
      )
  {
    mainWindowChanged = true;
  }

  // This will only be valid if we are not currently focused on THIS widget.
  // This should always be true at this point due to early exit above.
  MIDASLayout mainWindowLayout = this->GetCurrentMainWindowLayout();

  bool mainWindowLayoutChanged = false;
  if (mainWindowLayout != MIDAS_LAYOUT_UNKNOWN && mainWindowLayout != m_MainWindowLayout)
  {
    mainWindowLayoutChanged = true;
    m_MainWindowLayout = mainWindowLayout;
  }

  if (mainWindowChanged || m_CurrentRenderer == NULL || (mainWindowLayout != MIDAS_LAYOUT_UNKNOWN && m_Layout == MIDAS_LAYOUT_UNKNOWN))
  {
    const mitk::Geometry3D* worldGeometry = mainWindowAxial->GetRenderer()->GetWorldGeometry();
    if (worldGeometry)
    {
      mitk::Geometry3D::Pointer geometry = const_cast<mitk::Geometry3D*>(worldGeometry);
      assert(geometry);

      m_MainWindowAxial = mainWindowAxial;
      m_MainWindowSagittal = mainWindowSagittal;
      m_MainWindowCoronal = mainWindowCoronal;
      m_MainWindow3d = mainWindow3d;

      m_ViewerWidget->SetGeometry(geometry);
      m_ViewerWidget->SetBoundGeometryActive(false);
      m_ViewerWidget->SetNavigationControllerEventListening(true);
      m_ViewerWidget->SetDisplay2DCursorsLocally(true);
      m_ViewerWidget->SetShow3DWindowInOrthoView(true);
      if (!m_ViewerWidget->IsEnabled())
      {
        m_ViewerWidget->SetEnabled(true);
      }

      std::vector<mitk::DataNode*> crossHairs = m_ViewerWidget->GetWidgetPlanes();
      std::vector<mitk::BaseRenderer*> renderersToTrack;
      renderersToTrack.push_back(mainWindowAxial->GetRenderer());

      m_VisibilityTracker->SetRenderersToTrack(renderersToTrack);
      m_VisibilityTracker->SetNodesToIgnore(crossHairs);
      m_VisibilityTracker->OnPropertyChanged(); // force update

      m_ViewerWidget->FitToDisplay();
      this->ChangeLayout();
    }
  }
  else if (mainWindowLayoutChanged)
  {
    this->ChangeLayout();
  }

  m_CurrentRenderer = focusedRenderer;

  m_ViewerWidget->RequestUpdate();
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
MIDASLayout QmitkMIDASSegmentationViewWidget::GetCurrentMainWindowLayout()
{
  MIDASLayout mainWindowLayout = MIDAS_LAYOUT_UNKNOWN;
  MIDASOrientation mainWindowOrientation = this->GetCurrentMainWindowOrientation();

  if (mainWindowOrientation == MIDAS_ORIENTATION_AXIAL)
  {
    mainWindowLayout = MIDAS_LAYOUT_AXIAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    mainWindowLayout = MIDAS_LAYOUT_SAGITTAL;
  }
  else if (mainWindowOrientation == MIDAS_ORIENTATION_CORONAL)
  {
    mainWindowLayout = MIDAS_LAYOUT_CORONAL;
  }
  return mainWindowLayout;
}


//-----------------------------------------------------------------------------
void QmitkMIDASSegmentationViewWidget::OnScaleFactorChanged(QmitkMIDASSingleViewWidget*, double scaleFactor)
{
  double magnification = m_ViewerWidget->GetMagnification();

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
    m_ViewerWidget->SetMagnification(magnification);
    m_Magnification = magnification;
  }
}
