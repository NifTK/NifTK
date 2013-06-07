/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "TrackedImageView.h"
#include "TrackedImageViewPreferencePage.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <mitkSurface.h>
#include <vtkMatrix4x4.h>
#include "TrackedImageViewActivator.h"
#include <mitkCoordinateAxesData.h>
#include <mitkTrackedImageCommand.h>
#include <mitkFileIOUtils.h>
#include <mitkRenderingManager.h>

const std::string TrackedImageView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedimage";

//-----------------------------------------------------------------------------
TrackedImageView::TrackedImageView()
: m_Controls(NULL)
, m_ImageToProbeTransform(NULL)
, m_ImageToProbeFileName("")
{
}


//-----------------------------------------------------------------------------
TrackedImageView::~TrackedImageView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string TrackedImageView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void TrackedImageView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::TrackedImageView();
    m_Controls->setupUi(parent);

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    m_Controls->m_ImageNode->SetDataStorage(dataStorage);
    m_Controls->m_ImageNode->SetAutoSelectNewItems(false);
    m_Controls->m_ImageNode->SetPredicate(isImage);

    mitk::TNodePredicateDataType<mitk::Surface>::Pointer isSurface = mitk::TNodePredicateDataType<mitk::Surface>::New();
    m_Controls->m_ProbeSurfaceNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeSurfaceNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeSurfaceNode->SetPredicate(isSurface);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
    m_Controls->m_ProbeToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeToWorldNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeToWorldNode->SetPredicate(isTransform);

    // Set up the Render Window.
    // This currently has to be a 2D view, to generate the 2D plane geometry to render
    // which is then used to drive the moving 2D plane we see in 3D. This is how
    // the axial/sagittal/coronal slices work in the QmitkStdMultiWidget.

    m_Controls->m_RenderWindow->GetRenderer()->SetDataStorage(dataStorage);
    mitk::BaseRenderer::GetInstance(m_Controls->m_RenderWindow->GetRenderWindow())->SetMapperID(mitk::BaseRenderer::Standard2D);

    RetrievePreferenceValues();

    ctkServiceReference ref = mitk::TrackedImageViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::TrackedImageViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
    }

    connect(m_Controls->m_ImageNode, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnSelectionChanged(const mitk::DataNode*)));
  }
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TrackedImageView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    m_ImageToProbeFileName = prefs->Get(TrackedImageViewPreferencePage::CALIBRATION_FILE_NAME, "").c_str();
    m_ImageToProbeTransform = mitk::LoadVtkMatrix4x4FromFile(m_ImageToProbeFileName);
  }
}


//-----------------------------------------------------------------------------
void TrackedImageView::SetFocus()
{
  m_Controls->m_ImageNode->setFocus();
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnSelectionChanged(const mitk::DataNode* node)
{
  if (node != NULL)
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (image != NULL && image->GetGeometry() != NULL)
    {
      mitk::RenderingManager::GetInstance()->InitializeView(m_Controls->m_RenderWindow->GetRenderWindow(), image->GetGeometry());
    }
  }
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);

  mitk::DataNode::Pointer imageNode = m_Controls->m_ImageNode->GetSelectedNode();
  mitk::DataNode::Pointer surfaceNode = m_Controls->m_ProbeSurfaceNode->GetSelectedNode();
  mitk::DataNode::Pointer probeToWorldTransform = m_Controls->m_ProbeToWorldNode->GetSelectedNode();

  if (m_ImageToProbeTransform != NULL
      && probeToWorldTransform.IsNotNull()
     )
  {
    mitk::TrackedImageCommand::Pointer command = mitk::TrackedImageCommand::New();
    command->Update(imageNode,
                    surfaceNode,
                    probeToWorldTransform,
                    m_ImageToProbeTransform
                    );

    m_Controls->m_RenderWindow->GetRenderer()->GetDisplayGeometry()->Fit();
  }
}
