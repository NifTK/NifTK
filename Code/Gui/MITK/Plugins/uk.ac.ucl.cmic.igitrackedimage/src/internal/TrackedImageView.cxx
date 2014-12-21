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
#include <mitkTrackedImage.h>
#include <mitkFileIOUtils.h>
#include <mitkRenderingManager.h>
#include <mitkImage2DToTexturePlaneMapper3D.h>
#include <mitkIOUtil.h>
#include <mitkExceptionMacro.h>
#include <QMessageBox>

const std::string TrackedImageView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedimage";

//-----------------------------------------------------------------------------
TrackedImageView::TrackedImageView()
: m_Controls(NULL)
, m_ImageToTrackingSensorTransform(NULL)
, m_ShowCloneImageGroup(false)
, m_NameCounter(0)
{
}


//-----------------------------------------------------------------------------
TrackedImageView::~TrackedImageView()
{
  mitk::DataNode::Pointer imageNode = m_Controls->m_ImageNode->GetSelectedNode();  
  if (imageNode.IsNotNull())
  {
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
    if (image.IsNotNull())
    {
      // Remove any instance of Image2DToTexturePlaneMapper3D
      mitk::Mapper::Pointer mapper = imageNode->GetMapper(mitk::BaseRenderer::Standard3D);
      if (dynamic_cast<mitk::Image2DToTexturePlaneMapper3D*>(mapper.GetPointer()) != NULL)
      {
        imageNode->SetMapper(mitk::BaseRenderer::Standard3D, NULL);
      }
    }
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
    m_Controls->m_ImageNode->setEditable(true);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
    m_Controls->m_ImageToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ImageToWorldNode->SetAutoSelectNewItems(false);
    m_Controls->m_ImageToWorldNode->SetPredicate(isTransform);
    m_Controls->m_ImageToWorldNode->setEditable(true);

    // Set up the Render Window.
    // This currently has to be a 2D view, to generate the 2D plane geometry to render
    // which is then used to drive the moving 2D plane we see in 3D. This is how
    // the axial/sagittal/coronal slices work in the QmitkStdMultiWidget.

    m_Controls->m_RenderWindow->GetRenderer()->SetDataStorage(dataStorage);
    mitk::BaseRenderer::GetInstance(m_Controls->m_RenderWindow->GetRenderWindow())->SetMapperID(mitk::BaseRenderer::Standard2D);

    RetrievePreferenceValues();

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
}


//-----------------------------------------------------------------------------
void TrackedImageView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    std::string calibEmToOpticalFileName = prefs->Get(TrackedImageViewPreferencePage::EMTOWORLDCALIBRATION_FILE_NAME, "").c_str();
    if ( calibEmToOpticalFileName.size() > 0 )
    {
      m_EmToOpticalMatrix = mitk::LoadVtkMatrix4x4FromFile(calibEmToOpticalFileName);
    }
    else
    {
      m_EmToOpticalMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      m_EmToOpticalMatrix->Identity();
    }

    std::string imageToTrackingSensorFileName = prefs->Get(TrackedImageViewPreferencePage::CALIBRATION_FILE_NAME, "").c_str();
    vtkSmartPointer<vtkMatrix4x4> imageToSensorTransform = mitk::LoadVtkMatrix4x4FromFile(imageToTrackingSensorFileName);

    std::string scaleFileName = prefs->Get(TrackedImageViewPreferencePage::SCALE_FILE_NAME, "").c_str();
    vtkSmartPointer<vtkMatrix4x4> image2SensorScale = mitk::LoadVtkMatrix4x4FromFile(scaleFileName);

    if (prefs->GetBool(TrackedImageViewPreferencePage::FLIP_X_SCALING, false))
    {
      double x = image2SensorScale->GetElement(0,0);
      x *= -1;
      image2SensorScale->SetElement(0,0,x);
    }
    if (prefs->GetBool(TrackedImageViewPreferencePage::FLIP_Y_SCALING, false))
    {
      double y = image2SensorScale->GetElement(1,1);
      y *= -1;
      image2SensorScale->SetElement(1,1,y);
    }

    // Calculate image plane to tracker sensor transformation
    m_ImageToTrackingSensorTransform = NULL;
    m_ImageToTrackingSensorTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    m_ImageToTrackingSensorTransform->Identity();
    vtkMatrix4x4::Multiply4x4(imageToSensorTransform, image2SensorScale, m_ImageToTrackingSensorTransform);

    m_ShowCloneImageGroup = prefs->GetBool(TrackedImageViewPreferencePage::CLONE_IMAGE, false);
    m_Controls->m_CloneImageGroupBox->setVisible(m_ShowCloneImageGroup);

    m_Show2DWindow = prefs->GetBool(TrackedImageViewPreferencePage::SHOW_2D_WINDOW, false);
    m_Controls->m_RenderWindow->setVisible(m_Show2DWindow);
    if (m_Show2DWindow)
    {
      m_Controls->m_VerticalLayout->removeItem(m_Controls->m_VerticalSpacer);
      mitk::RenderingManager::GetInstance()->AddRenderWindow(m_Controls->m_RenderWindow->GetRenderWindow());
    }
    else
    {
      m_Controls->m_VerticalLayout->addItem(m_Controls->m_VerticalSpacer);
      mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_Controls->m_RenderWindow->GetRenderWindow());
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
      // Set a property saying which node is selected by this plugin.
      for (int i = 0; i < this->m_Controls->m_ImageNode->count(); i++)
      {
        mitk::DataNode::Pointer aNode = this->m_Controls->m_ImageNode->GetNode(i);
        aNode->SetBoolProperty(mitk::TrackedImage::TRACKED_IMAGE_SELECTED_PROPERTY_NAME, false);

        // Remove any instance of Image2DToTexturePlaneMapper3D
        mitk::Mapper::Pointer mapper = aNode->GetMapper(mitk::BaseRenderer::Standard3D);
        if (dynamic_cast<mitk::Image2DToTexturePlaneMapper3D*>(mapper.GetPointer()) != NULL)
        {
          aNode->SetMapper(mitk::BaseRenderer::Standard3D, NULL);
        }
      }
      mitk::DataNode::Pointer nodeToUpdate = this->GetDataStorage()->GetNamedNode(node->GetName());
      if (nodeToUpdate.IsNotNull())
      {
        nodeToUpdate->SetBoolProperty(mitk::TrackedImage::TRACKED_IMAGE_SELECTED_PROPERTY_NAME, true);

        mitk::Image2DToTexturePlaneMapper3D::Pointer newMapper = mitk::Image2DToTexturePlaneMapper3D::New();
        nodeToUpdate->SetMapper(mitk::BaseRenderer::Standard3D, newMapper);
      }

      // This is expensive, so only update if the window is visible.
      if (m_Show2DWindow)
      {
        mitk::RenderingManager::GetInstance()->InitializeView(m_Controls->m_RenderWindow->GetRenderWindow(), image->GetGeometry());
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

      if (trackingSensorToTrackerTransform.IsNotNull())
      {
        // We publish this update signal immediately after the image plane is updated,
        // as we want the Overlay Display to listen synchronously, and update immediately.
        // We don't want a rendering event to trigger the Overlay Display to re-render at the
        // wrong position, and momentarily display the wrong thing.
          
        mitk::TrackedImage::Pointer command = mitk::TrackedImage::New();
        command->Update(imageNode,
                        trackingSensorToTrackerTransform,
                        *m_ImageToTrackingSensorTransform,
                        *m_EmToOpticalMatrix
                        );
                    
        ctkDictionary properties;
        emit Updated(properties);

        // This is expensive, so only update if the window is visible.
        if (m_Show2DWindow)
        {
          mitk::RenderingManager::GetInstance()->InitializeView(m_Controls->m_RenderWindow->GetRenderWindow(), image->GetGeometry());
        }

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

  bool isSuccessful = false;
  mitk::DataNode::Pointer node = m_Controls->m_ImageNode->GetSelectedNode();
  if ( node.IsNotNull() )
  {    
    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(node->GetData());
    if ( image.IsNotNull() )
    {
      QString imageName = tr("TrackedImageView-%1").arg(m_NameCounter);
      QString fileNameWithoutGeometry = directoryName + QDir::separator() + imageName + QString(".png");
      QString fileNameForGeometry = directoryName + QDir::separator() + imageName + QString("_mtx.txt");

      // clone the image node to keep the geometry information unchanged during the process of saving.
      mitk::Image::Pointer savedMitkImage = image->Clone();

      // Save the 4x4 matrix of the geometry to disk.
      mitk::CoordinateAxesData::Pointer transform = mitk::CoordinateAxesData::New();
      transform->SetGeometry(savedMitkImage->GetGeometry());
      isSuccessful = transform->SaveToFile(fileNameForGeometry.toStdString());
      if (!isSuccessful)
      {
        mitkThrow() << "Failed to save transformation " << fileNameForGeometry.toStdString() << std::endl;
      }

      // clone the origin ultrasound image (changing orientation) to disk.
      mitk::Image::Pointer untouchedImage = savedMitkImage->Clone();
      mitk::BaseGeometry* geometry = untouchedImage->GetGeometry();
      if (geometry)
      {
        vtkSmartPointer<vtkMatrix4x4> identityMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
        identityMatrix->Identity();
        geometry->SetIndexToWorldTransformByVtkMatrix(identityMatrix);
      }
      isSuccessful = mitk::IOUtil::SaveImage(untouchedImage, fileNameWithoutGeometry.toStdString());
      if (!isSuccessful)
      {
        mitkThrow() << "Failed to save un-oriented image " << fileNameWithoutGeometry.toStdString() << std::endl;
      }

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
