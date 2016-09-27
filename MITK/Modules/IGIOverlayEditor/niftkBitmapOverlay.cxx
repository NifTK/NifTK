/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBitmapOverlay.h"
#include <niftkVTKFunctions.h>

#include <itkObject.h>
#include <itkMacro.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkMapper.h>
#include <vtkImageActor.h>
#include <vtkImageMapper.h>
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <mitkImage.h>
#include <mitkVtkLayerController.h>

namespace niftk
{

//-----------------------------------------------------------------------------
BitmapOverlay::BitmapOverlay()
: m_RenderWindow(NULL)
, m_BackRenderer(NULL)
, m_FrontRenderer(NULL)
, m_BackActor(NULL)
, m_FrontActor(NULL)
, m_DataStorage(NULL)
, m_ImageDataNode(NULL)
, m_IsEnabled(false)
, m_Opacity(0.5)
, m_FlipViewUp(true)
{
  m_BackRenderer  = vtkSmartPointer<vtkRenderer>::New();
  m_FrontRenderer = vtkSmartPointer<vtkRenderer>::New();
  m_BackActor     = vtkSmartPointer<vtkImageActor>::New();
  m_FrontActor    = vtkSmartPointer<vtkImageActor>::New();
  m_ClippingRange[0] = 1;
  m_ClippingRange[1] = 10000;
}


//-----------------------------------------------------------------------------
BitmapOverlay::~BitmapOverlay()
{
  if ( m_RenderWindow != NULL )
  {
    if ( this->IsEnabled() )
    {
      this->Disable();
    }
  }
}


//-----------------------------------------------------------------------------
void BitmapOverlay::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  this->Modified();
}


//-----------------------------------------------------------------------------
vtkRenderer* BitmapOverlay::GetVtkRenderer()
{
  return m_BackRenderer;
}


//-----------------------------------------------------------------------------
void BitmapOverlay::SetRenderWindow(vtkRenderWindow* renderWindow)
{
  m_RenderWindow = renderWindow;
  this->Modified();
}


//-----------------------------------------------------------------------------
void BitmapOverlay::SetEnabled(const bool& enable)
{
  if (enable)
  {
    this->Enable();
  }
  else
  {
    this->Disable();
  }
}


//-----------------------------------------------------------------------------
bool BitmapOverlay::IsEnabled()
{
  return m_IsEnabled;
}


//-----------------------------------------------------------------------------
void BitmapOverlay::Enable()
{
  if ( !this->IsEnabled() )
  {
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertBackgroundRenderer(m_BackRenderer, true);
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertForegroundRenderer(m_FrontRenderer, true);
    m_IsEnabled = true;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void BitmapOverlay::Disable()
{
  if ( this->IsEnabled() )
  {
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_BackRenderer);
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_FrontRenderer);
    m_IsEnabled = false;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void BitmapOverlay::SetOpacity(const double& opacity)
{
  m_Opacity = opacity;
  m_BackActor->SetOpacity(1.0);
  m_FrontActor->SetOpacity(m_Opacity);
  this->Modified();
}


//-----------------------------------------------------------------------------
void BitmapOverlay::SetClippingRange(const double& nearZ, const double& farZ)
{
  m_ClippingRange[0] = nearZ;
  m_ClippingRange[1] = farZ;
  this->Modified();
}


//-----------------------------------------------------------------------------
void BitmapOverlay::GetClippingRange(double& nearZ, double& farZ)
{
  nearZ = m_ClippingRange[0];
  farZ = m_ClippingRange[1];
}


//-----------------------------------------------------------------------------
void BitmapOverlay::NodeAdded (const mitk::DataNode *node)
{
  if (m_ImageDataNode.IsNull())
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (image != NULL
        && (image->GetDimension() == 2 || (image->GetDimension() == 3 && image->GetDimensions()[2] == 1))
        )
    {
      this->SetNode(node);
    }
  }
}


//-----------------------------------------------------------------------------
void BitmapOverlay::NodeChanged (const mitk::DataNode *node)
{
  if (m_ImageDataNode.IsNotNull() && node == m_ImageDataNode)
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (image != NULL
        && (image->GetDimension() == 2 || (image->GetDimension() == 3 && image->GetDimensions()[2] == 1))
        )
    {
      // Basically, as we know the node has changed in some way,
      // and its the node we are watching, then we are trying to
      // set modified flags, so that the next rendering pass will
      // update the screen.
      m_FrontActor->SetInputData(image->GetVtkImageData());
      m_BackActor->SetInputData(image->GetVtkImageData());
      image->GetVtkImageData()->Modified();
      this->Modified();
    }
  }
}


//-----------------------------------------------------------------------------
void BitmapOverlay::NodeRemoved(const mitk::DataNode * node)
{
  if (m_ImageDataNode.IsNotNull() && node == m_ImageDataNode)
  {
    this->SetNode(NULL);
  }
}


//-----------------------------------------------------------------------------
bool BitmapOverlay::SetNode(const mitk::DataNode* node)
{
  bool wasSuccessful = false;

  if (m_DataStorage.IsNull())
  {
    MITK_ERROR << "BitmapOverlay::SetNode: Error, DataStorage is NULL" << std::endl;
    return wasSuccessful;
  }

  if(m_RenderWindow != NULL)
  {
    if (node == NULL)
    {
      m_FrontActor->SetInputData(NULL);
      m_BackActor->SetInputData(NULL);
      m_ImageDataNode = NULL;
      wasSuccessful = true;
    }
    else
    {
      mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
      if (image != NULL
          && (image->GetDimension() == 2 || (image->GetDimension() == 3 && image->GetDimensions()[2] == 1))
          )
      {
        m_FrontActor->SetInputData(image->GetVtkImageData());
        m_BackActor->SetInputData(image->GetVtkImageData());

        m_BackRenderer->AddActor(m_BackActor);
        m_FrontRenderer->AddActor(m_FrontActor);

        m_BackRenderer->InteractiveOff();
        m_FrontRenderer->InteractiveOff();

        m_ImageDataNode = const_cast<mitk::DataNode*>(node);

        this->SetOpacity(m_Opacity);
        this->SetupCamera();

        wasSuccessful = true;

      } // end if valid image
    }
  } // end if valid render window.

  if (wasSuccessful)
  {
    this->Modified();
  }
  return wasSuccessful;
}


//-----------------------------------------------------------------------------
void BitmapOverlay::SetupCamera()
{
  vtkImageData *image = m_BackActor->GetInput();
  if (image == NULL)
  {
    // This is ok, as we may get a resize event, and hence
    // this method is called before we have an image set up.
    return;
  }

  if (m_RenderWindow == NULL)
  {
    MITK_ERROR << "BitmapOverlay::SetupCamera: Error, the vtkRenderWindow is NULL" << std::endl;
    return;
  }

  vtkCamera* backCamera = m_BackRenderer->GetActiveCamera();
  if (backCamera == NULL)
  {
    MITK_ERROR << "BitmapOverlay::SetupCamera: Error, the backCamera is NULL" << std::endl;
    return;
  }

  vtkCamera* frontCamera = m_FrontRenderer->GetActiveCamera();
  if (frontCamera == NULL)
  {
    MITK_ERROR << "BitmapOverlay::SetupCamera: Error, the frontCamera is NULL" << std::endl;
    return;
  }

  int    windowSize[2];
  int    imageSize[3];
  double spacing[3];
  double origin[3];
  double xAxis[3] = {1, 0, 0};
  double yAxis[3] = {0, 1, 0};

  image->GetDimensions(imageSize);
  image->GetOrigin(origin);
  image->GetSpacing(spacing);

  windowSize[0] = m_RenderWindow->GetSize()[0];
  windowSize[1] = m_RenderWindow->GetSize()[1];

  niftk::SetCameraParallelTo2DImage(imageSize, windowSize, origin, spacing,
                                    xAxis, yAxis, m_ClippingRange, m_FlipViewUp, *backCamera);

  niftk::SetCameraParallelTo2DImage(imageSize, windowSize, origin, spacing,
                                    xAxis, yAxis, m_ClippingRange, m_FlipViewUp, *frontCamera);

  this->Modified();
}

} // end namespace
