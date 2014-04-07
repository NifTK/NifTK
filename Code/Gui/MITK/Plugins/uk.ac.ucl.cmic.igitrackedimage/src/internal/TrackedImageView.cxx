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
#include <mitkIOUtil.h>
#include <QMessageBox>

const std::string TrackedImageView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedimage";

//-----------------------------------------------------------------------------
TrackedImageView::TrackedImageView()
: m_Controls(NULL)
, m_ImageToTrackingSensorTransform(NULL)
, m_ImageToTrackingSensorFileName("")
, m_PlaneNode(NULL)
, m_ShowCloneImageGroup(false)
{
  m_ImageScaling[0] = 1;
  m_ImageScaling[1] = 1;
  m_NameCounter = 0;
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

    m_Controls->m_CloneImageGroupBox->setVisible(m_ShowCloneImageGroup);   
    connect(m_Controls->m_ClonePushButton, SIGNAL(clicked()), this, SLOT(OnClonePushButtonClicked()));

    m_Controls->m_CloneTrackedImageDirectoryChooser->setFilters(ctkPathLineEdit::Dirs);
    m_Controls->m_CloneTrackedImageDirectoryChooser->setOptions(ctkPathLineEdit::ShowDirsOnly);
    m_Controls->m_CloneTrackedImageDirectoryChooser->setCurrentPath(tr("C:\\Workspace\\Laparoscopic\\tmp"));

    ctkServiceReference ref = mitk::TrackedImageViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::TrackedImageViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
      eventAdmin->publishSignal(this, SIGNAL(Updated(ctkDictionary)),"uk/ac/ucl/cmic/IGITRACKEDIMAGEUPDATE", Qt::DirectConnection);
    }
  }
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();

  m_Controls->m_CloneImageGroupBox->setVisible(m_ShowCloneImageGroup);
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

    m_ShowCloneImageGroup = prefs->GetBool(TrackedImageViewPreferencePage::CLONE_IMAGE, false);

    if (prefs->GetBool(TrackedImageViewPreferencePage::FLIP_X_SCALING, false))
    {
      m_ImageScaling[0] *= -1;
    }
    m_ImageScaling[1] = prefs->GetDouble(TrackedImageViewPreferencePage::Y_SCALING, 1);
    if (prefs->GetBool(TrackedImageViewPreferencePage::FLIP_Y_SCALING, false))
    {
      m_ImageScaling[1] *= -1;
    }
    if(m_PlaneNode.IsNotNull())
    {
      m_PlaneNode->Modified();  
    }
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
      this->m_Controls->m_DoUpdateCheckBox->setChecked(false);
      
      // Set a property saying which node is selected by this plugin.
      for (int i = 0; i < this->m_Controls->m_ImageNode->count(); i++)
      {
        mitk::DataNode::Pointer aNode = this->m_Controls->m_ImageNode->GetNode(i);
        aNode->SetBoolProperty(mitk::TrackedImageCommand::TRACKED_IMAGE_SELECTED_PROPERTY_NAME, false);
      }
      mitk::DataNode::Pointer nodeToUpdate = this->GetDataStorage()->GetNamedNode(node->GetName());
      if (nodeToUpdate.IsNotNull())
      {
        nodeToUpdate->SetBoolProperty(mitk::TrackedImageCommand::TRACKED_IMAGE_SELECTED_PROPERTY_NAME, true);        
      }
      
      // Update the view to match the image geometry.
      mitk::RenderingManager::GetInstance()->InitializeView(m_Controls->m_RenderWindow->GetRenderWindow(), image->GetGeometry());

      // Generate a plane to visualise in the 3D window, and in this plugins Render Window.
      // This view is a shared VTK resource, which may be buggy on some systems.
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
  if (imageNode.IsNotNull())
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
    if (image.IsNotNull())
    {

      mitk::DataNode::Pointer trackingSensorToTrackerTransform = m_Controls->m_ImageToWorldNode->GetSelectedNode();

      if (this->m_Controls->m_DoUpdateCheckBox->isChecked()
          && m_ImageToTrackingSensorTransform != NULL
          && trackingSensorToTrackerTransform.IsNotNull()
         )
      {
        // Check modified times to minimise updates.
        unsigned long trackingSensorToTrackerModifiedTime = trackingSensorToTrackerTransform->GetMTime();
        unsigned long planeModifiedTime = m_PlaneNode->GetMTime(); // proxy for this class.
        
        if (planeModifiedTime < trackingSensorToTrackerModifiedTime)
        {          
          // Start of important bit.
          
          // We publish this update signal immediately after the image plane is updated,
          // as we want the Overlay Display to listen synchronously, and update immediately.
          // We don't want a rendering event to trigger the Overlay Display to re-render at the
          // wrong position, and momentarily display the wrong thing.
          
          mitk::TrackedImageCommand::Pointer command = mitk::TrackedImageCommand::New();
          command->Update(imageNode,
                          trackingSensorToTrackerTransform,
                          m_ImageToTrackingSensorTransform,
                          m_ImageScaling
                          );
          
          m_PlaneNode->Modified();
          
          ctkDictionary properties;
          emit Updated(properties);
          
          // End of important bit.
          
          mitk::RenderingManager::GetInstance()->InitializeView(m_Controls->m_RenderWindow->GetRenderWindow(), image->GetGeometry());
          
        } // if modified times suggest we need an update
      } // end if input is valid
    } // if got an image
  } // if got an image node
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnClonePushButtonClicked()
{
  QString directoryName = m_Controls->m_CloneTrackedImageDirectoryChooser->currentPath();
  if (directoryName.length() == 0)
  {
    QMessageBox msgBox; 
    msgBox.setText("The folder to save is not-selected.");
    msgBox.setInformativeText("Please select a folder to save image node.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  mitk::DataNode::Pointer node = m_Controls->m_ImageNode->GetSelectedNode();
  if ( node.IsNotNull() )
  {    
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    if ( image.IsNotNull() )
    {
      QString imageName = tr("TrackedImageView-%1").arg(m_NameCounter);
      QString fileNameWithGeometry = directoryName + QDir::separator() + imageName + QString(".nii");
      QString fileNameWithoutGeometry = directoryName + QDir::separator() + imageName + QString(".png");

      // clone the origin ultrasound image (without changing orientation) to disk.
      mitk::Image::Pointer savedMitkImage = image->Clone();
      mitk::IOUtil::SaveImage(savedMitkImage, fileNameWithGeometry.toStdString());

      // clone the origin ultrasound image (changing orientation) to disk.
      mitk::Image::Pointer untouchedImage = savedMitkImage->Clone();
      mitk::Geometry3D::Pointer geometry = untouchedImage->GetGeometry();
      if (geometry.IsNotNull())
      {
        vtkSmartPointer<vtkMatrix4x4> identityMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
        identityMatrix->Identity();
        geometry->SetIndexToWorldTransformByVtkMatrix(identityMatrix);
      }
      mitk::IOUtil::SaveImage(untouchedImage, fileNameWithoutGeometry.toStdString());

      // For immediate visualisation, we create a new DataNode with the new image.
      mitk::DataNode::Pointer savedImageNode = mitk::DataNode::New();
      savedImageNode->SetData(savedMitkImage);
      savedImageNode->SetProperty("visible", mitk::BoolProperty::New(true));
      savedImageNode->SetProperty("name", mitk::StringProperty::New(imageName.toStdString()));
      savedImageNode->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(true));
      savedImageNode->SetProperty("helper object", mitk::BoolProperty::New(false));
      savedImageNode->SetVisibility(true);

      // Add to data storage.
      mitk::DataStorage* dataStorage = this->GetDataStorage();
      dataStorage->Add(savedImageNode);

      mitk::RenderingManager::GetInstance()->RequestUpdateAll();
      m_NameCounter++;
    }
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("The image node is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select an image node.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }
}
