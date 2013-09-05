/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
#include <mitkGeometry2DDataMapper2D.h>

const std::string TrackedImageView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedimage";

//-----------------------------------------------------------------------------
TrackedImageView::TrackedImageView()
: m_Controls(NULL)
, m_ImageToTrackingSensorTransform(NULL)
, m_ImageToTrackingSensorFileName("")
, m_PlaneNode(NULL)
{
  m_ImageScaling[0] = 1;
  m_ImageScaling[1] = 1;
}


//-----------------------------------------------------------------------------
TrackedImageView::~TrackedImageView()
{
  mitk::DataStorage* dataStorage = this->GetDataStorage();
  if (dataStorage != NULL && m_PlaneNode.IsNotNull() && dataStorage->Exists(m_PlaneNode))
  {
    dataStorage->Remove(m_PlaneNode);
  }

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

    connect(m_Controls->m_ImageNode, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnSelectionChanged(const mitk::DataNode*)));

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    m_Controls->m_ImageNode->SetDataStorage(dataStorage);
    m_Controls->m_ImageNode->SetAutoSelectNewItems(false);
    m_Controls->m_ImageNode->SetPredicate(isImage);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
    m_Controls->m_ImageToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ImageToWorldNode->SetAutoSelectNewItems(false);
    m_Controls->m_ImageToWorldNode->SetPredicate(isTransform);

    m_Controls->m_DoUpdateCheckBox->setChecked(false);

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
    m_ImageToTrackingSensorFileName = prefs->Get(TrackedImageViewPreferencePage::CALIBRATION_FILE_NAME, "").c_str();
    m_ImageToTrackingSensorTransform = mitk::LoadVtkMatrix4x4FromFile(m_ImageToTrackingSensorFileName);
    m_ImageScaling[0] = prefs->GetDouble(TrackedImageViewPreferencePage::X_SCALING, 1);
    m_ImageScaling[1] = prefs->GetDouble(TrackedImageViewPreferencePage::Y_SCALING, 1);
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

      float white[3] = {1.0f,1.0f,1.0f};
      mitk::Geometry2DDataMapper2D::Pointer mapper(NULL);

      m_PlaneNode = (mitk::BaseRenderer::GetInstance(m_Controls->m_RenderWindow->GetRenderWindow()))->GetCurrentWorldGeometry2DNode();
      m_PlaneNode->SetColor(white, mitk::BaseRenderer::GetInstance(m_Controls->m_RenderWindow->GetRenderWindow()));
      m_PlaneNode->SetProperty("visible", mitk::BoolProperty::New(true));
      m_PlaneNode->SetProperty("name", mitk::StringProperty::New(mitk::TrackedImageCommand::TRACKED_IMAGE_NODE_NAME));
      m_PlaneNode->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(false));
      m_PlaneNode->SetProperty("helper object", mitk::BoolProperty::New(true));
      m_PlaneNode->SetProperty("visible background", mitk::BoolProperty::New(false));

      mapper = mitk::Geometry2DDataMapper2D::New();
      m_PlaneNode->SetMapper(mitk::BaseRenderer::Standard2D, mapper);

      mitk::DataStorage* dataStorage = this->GetDataStorage();
      if (!dataStorage->Exists(m_PlaneNode))
      {
        dataStorage->Add(m_PlaneNode);
      }

      mitk::RenderingManager::GetInstance()->RequestUpdateAll();
    }
  }
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);

  mitk::DataNode::Pointer imageNode = m_Controls->m_ImageNode->GetSelectedNode();
  mitk::DataNode::Pointer trackingSensorToTrackerTransform = m_Controls->m_ImageToWorldNode->GetSelectedNode();

  if (this->m_Controls->m_DoUpdateCheckBox->isChecked()
      && imageNode.IsNotNull()
      && m_ImageToTrackingSensorTransform != NULL
      && trackingSensorToTrackerTransform.IsNotNull()
     )
  {
    mitk::TrackedImageCommand::Pointer command = mitk::TrackedImageCommand::New();
    command->Update(imageNode,
                    trackingSensorToTrackerTransform,
                    m_ImageToTrackingSensorTransform,
                    m_ImageScaling
                    );

    m_PlaneNode->Modified();

    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
    if (image.IsNotNull())
    {
      mitk::RenderingManager::GetInstance()->InitializeView(m_Controls->m_RenderWindow->GetRenderWindow(), image->GetGeometry());
    }
  }
}
