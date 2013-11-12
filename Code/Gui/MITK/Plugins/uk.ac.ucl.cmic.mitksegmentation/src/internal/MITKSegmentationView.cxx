/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "MITKSegmentationView.h"

// MITK
#include <mitkSegmentationObjectFactory.h>
#include <mitkTool.h>
#include <mitkToolManager.h>
#include <mitkAddContourTool.h>
#include <mitkSubtractContourTool.h>
#include <mitkDrawPaintbrushTool.h>
#include <mitkErasePaintbrushTool.h>
#include <mitkRegionGrowingTool.h>
#include <mitkCorrectorTool2D.h>
#include <mitkFillRegionTool.h>
#include <mitkEraseRegionTool.h>
#include <mitkDataStorageUtils.h>

// Qt
#include <QMessageBox>

const std::string MITKSegmentationView::VIEW_ID = "uk.ac.ucl.cmic.mitksegmentation";

MITKSegmentationView::MITKSegmentationView()
: QmitkMIDASBaseSegmentationFunctionality()
, m_Controls(NULL)
, m_Layout(NULL)
, m_ContainerForControlsWidget(NULL)
, m_OutlineBinary(true)
, m_VolumeRendering(false)
{
  m_DefaultSegmentationColor.setRedF(1);
  m_DefaultSegmentationColor.setGreenF(0);
  m_DefaultSegmentationColor.setBlueF(0);
}


MITKSegmentationView::MITKSegmentationView(
    const MITKSegmentationView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


MITKSegmentationView::~MITKSegmentationView()
{
}


std::string MITKSegmentationView::GetViewID() const
{
  return VIEW_ID;
}


void MITKSegmentationView::SetFocus()
{
}


void MITKSegmentationView::CreateQtPartControl( QWidget *parent )
{
  this->SetParent(parent);

  if (!m_Controls)
  {
    m_Layout = new QGridLayout(parent);
    m_Layout->setContentsMargins(0,0,0,0);
    m_Layout->setSpacing(0);
    m_Layout->setRowStretch(0, 0);
    m_Layout->setRowStretch(1, 10);
    m_Layout->setRowStretch(2, 0);
    m_Layout->setRowStretch(3, 0);

    m_ContainerForControlsWidget = new QWidget(parent);

    m_Controls = new Ui::MITKSegmentationViewControls();
    m_Controls->setupUi(m_ContainerForControlsWidget);

    QmitkMIDASBaseSegmentationFunctionality::CreateQtPartControl(parent);

    m_Layout->addWidget(m_ContainerForSelectorWidget,         0, 0);
    m_Layout->addWidget(m_ContainerForToolWidget,             1, 0);
    m_Layout->addWidget(m_ContainerForControlsWidget,         2, 0);
    m_Layout->addWidget(m_ContainerForSegmentationViewWidget, 3, 0);

    // Ideally we would want this, but there is a geometry problem at the moment.
    // When the user views stuff in the MITK viewer, this widget has the wrong geometry.
    // This is most likely to happen for people who are not using MIDAS, and so will
    // be using the MITK display, and this MITK segmentation widget. So for the time
    // being we remove this viewer.
    m_ContainerForSegmentationViewWidget->setVisible(false);
    m_SegmentationView->setVisible(false);

    m_ToolSelector->m_ManualToolSelectionBox->SetLayoutColumns(2);
    m_ToolSelector->m_ManualToolSelectionBox->SetShowNames(true);
    m_ToolSelector->m_ManualToolSelectionBox->SetGenerateAccelerators(true);
    m_ToolSelector->m_ManualToolSelectionBox->SetDisplayedToolGroups("Add Subtract Paint Wipe 'Region Growing' Correction Fill Erase");

    // Force all MITK tools to 2D mode.
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);

    mitk::SegTool2D* mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::AddContourTool>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::SubtractContourTool>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::DrawPaintbrushTool>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::ErasePaintbrushTool>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::RegionGrowingTool>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::CorrectorTool2D>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::FillRegionTool>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    mitkTool = dynamic_cast<mitk::SegTool2D*>(toolManager->GetToolById(toolManager->GetToolIdByToolType<mitk::EraseRegionTool>()));
    assert(mitkTool);
    mitkTool->SetShowMarkerNodes(false);
    mitkTool->SetEnable3DInterpolation(false);

    // Make sure these are up to date when view first shown.
    this->RetrievePreferenceValues();

    // Finally do Qt signals/slots.
    this->CreateConnections();
  }
}


void MITKSegmentationView::CreateConnections()
{
  QmitkMIDASBaseSegmentationFunctionality::CreateConnections();

  if ( m_Controls )
  {
    connect(m_ImageAndSegmentationSelector->m_NewSegmentationButton, SIGNAL(clicked()), this, SLOT(OnCreateNewSegmentationButtonPressed()) );
  }
}


void MITKSegmentationView::EnableSegmentationWidgets(bool b)
{
  // No additional widgets to enable. They are all currently in the base class.
}


bool MITKSegmentationView::IsNodeASegmentationImage(const mitk::DataNode::Pointer node)
{
  assert(node);
  bool result = false;

  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      result = true;
    }
  }
  return result;
}

bool MITKSegmentationView::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node)
{
  return this->IsNodeASegmentationImage(node);
}

void MITKSegmentationView::OnCreateNewSegmentationButtonPressed()
{
  // Create the new segmentation, either using a previously selected one, or create a new volume.
  // Compare with MIDASGeneralSegmentorView and MIDASMorphologicalSegmentorView. In the MIDAS
  // segmentation tools there is a lot of reference data to create. So, if you select a binary image
  // that has no reference data you need to "restart" the segmentation by creating all the reference data.
  // However, this tool is much simpler. The ToolManager should have a reference image, and we segment that.
  //
  mitk::DataNode::Pointer newSegmentation = NULL;

  // Make sure we have a reference images... which should always be true at this point.
  mitk::Image* image = this->GetReferenceImageFromToolManager();
  if (image != NULL)
  {
    mitk::ToolManager::Pointer toolManager = this->GetToolManager();
    assert(toolManager);

    this->WaitCursorOn();

    newSegmentation = CreateNewSegmentation(m_DefaultSegmentationColor);

    // The above method returns NULL if the user exited the colour selection dialog box.
    if (newSegmentation.IsNull())
    {
      return;
    }

    // Apply preference values.
    newSegmentation->SetBoolProperty("outline binary", m_OutlineBinary);
    newSegmentation->SetBoolProperty("volumerendering", m_VolumeRendering);
    if (!m_OutlineBinary)
    {
      newSegmentation->SetOpacity(0.5);
    }

    // Give the ToolManager data to segment.
    mitk::ToolManager::DataVectorType workingData;
    workingData.push_back(newSegmentation);
    toolManager->SetWorkingData(workingData);

    this->EnableSegmentationWidgets(true);

    this->FocusOnCurrentWindow();
    this->RequestRenderWindowUpdate();
    this->WaitCursorOff();

  } // end if we have a reference image.

  // Finally, select the new segmentation node.
  this->SetCurrentSelection(newSegmentation);
}

void MITKSegmentationView::RetrievePreferenceValues()
{
  QmitkMIDASBaseSegmentationFunctionality::RetrievePreferenceValues();

  berry::IPreferencesService::Pointer prefService
      = berry::Platform::GetServiceRegistry()
      .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

    assert( prefService );

    berry::IBerryPreferences::Pointer prefs
        = (prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName()))
          .Cast<berry::IBerryPreferences>();

    assert( prefs );

    m_OutlineBinary = prefs->GetBool("draw outline", true);
    m_VolumeRendering = prefs->GetBool("volume rendering", false);
}
