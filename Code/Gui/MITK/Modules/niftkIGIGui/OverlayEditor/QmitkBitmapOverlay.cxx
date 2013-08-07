/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#include "QmitkBitmapOverlay.h"

#include <mitkVtkLayerController.h>
#include <itkObject.h>
#include <itkMacro.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkMapper.h>
#include <vtkImageActor.h>
#include <vtkImageMapper.h>
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <niftkVTKFunctions.h>

//-----------------------------------------------------------------------------
QmitkBitmapOverlay::QmitkBitmapOverlay()
: m_RenderWindow(NULL)
, m_BackRenderer(NULL)
, m_FrontRenderer(NULL)
, m_BackActor(NULL)
, m_FrontActor(NULL)
, m_Mapper(NULL)
, m_DataStorage(NULL)
, m_ImageDataNode(NULL)
, m_IsEnabled(false)
, m_Opacity(0.5)
, m_AutoSelectNodes(true)
, m_FlipViewUp(true)
{
  m_BackRenderer        = vtkRenderer::New();
  m_FrontRenderer       = vtkRenderer::New();
  m_BackActor           = vtkImageActor::New();
  m_FrontActor          = vtkImageActor::New();
  m_Mapper              = vtkImageMapper::New();
}


//-----------------------------------------------------------------------------
QmitkBitmapOverlay::~QmitkBitmapOverlay()
{

  if ( m_RenderWindow != NULL )
  {
    if ( this->IsEnabled() )
    {
      this->Disable();
    }
  }

  if ( m_Mapper != NULL )
  {
    m_Mapper->Delete();
  }

  if ( m_BackActor != NULL )
  {
    m_BackActor->Delete();
  }

  if ( m_FrontActor != NULL )
  {
    m_FrontActor->Delete();
  }

  if ( m_BackRenderer != NULL )
  {
    m_BackRenderer->Delete();
  }

  if ( m_FrontRenderer != NULL )
  {
    m_FrontRenderer->Delete();
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::SetDataStorage (mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  this->Modified();
}


//-----------------------------------------------------------------------------
vtkRenderer* QmitkBitmapOverlay::GetVtkRenderer()
{
  return m_BackRenderer;
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::SetRenderWindow( vtkRenderWindow* renderWindow )
{
  m_RenderWindow = renderWindow;
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::SetEnabled(const bool& enable)
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
bool QmitkBitmapOverlay::IsEnabled()
{
  return  m_IsEnabled;
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::Enable()
{
  if ( !this->IsEnabled() )
  {
    // mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertBackgroundRenderer(m_BackRenderer,false);
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertForegroundRenderer(m_FrontRenderer,false);
    m_IsEnabled = true;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::Disable()
{
  if ( this->IsEnabled() )
  {
    // mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_BackRenderer);
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_FrontRenderer);
    m_IsEnabled = false;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::SetOpacity(const double& opacity)
{
  m_Opacity = opacity;
  m_BackActor->SetOpacity(1.0);
  m_FrontActor->SetOpacity(m_Opacity);

  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::AutoSelectDataNode(const mitk::DataNode* node)
{
  if (node != NULL && this->GetAutoSelectNodes())
  {
    // ToDo: These strings hard coded, as this widget in niftkCoreGui rather than niftkIGIGui.
    if (node->GetName() == "NVIDIA SDI stream 0")
    {
      this->SetFlipViewUp(true);
      this->SetNode(node);
    }
    else if (node->GetName() == "OpenCV image")
    {
      this->SetFlipViewUp(true);
      this->SetNode(node);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::NodeAdded (const mitk::DataNode * node)
{
  if (m_ImageDataNode.IsNull())
  {
    this->AutoSelectDataNode(node);
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::NodeChanged (const mitk::DataNode * node)
{
  if (m_ImageDataNode.IsNull())
  {
    this->AutoSelectDataNode(node);
  }
  else if (node == m_ImageDataNode )
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (image != NULL)
    {
      m_FrontActor->SetInput(image->GetVtkImageData());
      m_BackActor->SetInput(image->GetVtkImageData());
      image->GetVtkImageData()->Modified();

      this->Modified();
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::NodeRemoved (const mitk::DataNode * node )
{
  if ( node == m_ImageDataNode )
  {
    this->SetNode(NULL);
  }
}


//-----------------------------------------------------------------------------
bool QmitkBitmapOverlay::SetNode(const mitk::DataNode* node)
{
  bool wasSuccessful = false;

  if (m_DataStorage.IsNull())
  {
    MITK_ERROR << "QmitkBitmapOverlay::SetNode: Error, DataStorage is NULL" << std::endl;
    return wasSuccessful;
  }

  if(m_RenderWindow != NULL)
  {

    if (node == NULL)
    {
      m_FrontActor->SetInput(NULL);
      m_BackActor->SetInput(NULL);
      m_ImageDataNode = NULL;
      wasSuccessful = true;
    }
    else
    {
      mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
      if (image != NULL)
      {
        m_FrontActor->SetInput(image->GetVtkImageData());
        m_BackActor->SetInput(image->GetVtkImageData());

        m_BackRenderer->AddActor( m_BackActor );
        m_FrontRenderer->AddActor( m_FrontActor );

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
void QmitkBitmapOverlay::SetupCamera()
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
    MITK_ERROR << "QmitkBitmapOverlay::SetupCamera: Error, the vtkRenderWindow is NULL" << std::endl;
    return;
  }

  vtkCamera* backCamera = m_BackRenderer->GetActiveCamera();
  if (backCamera == NULL)
  {
    MITK_ERROR << "QmitkBitmapOverlay::SetupCamera: Error, the backCamera is NULL" << std::endl;
    return;
  }

  vtkCamera* frontCamera = m_FrontRenderer->GetActiveCamera();
  if (frontCamera == NULL)
  {
    MITK_ERROR << "QmitkBitmapOverlay::SetupCamera: Error, the frontCamera is NULL" << std::endl;
    return;
  }

  int    windowSize[2];
  int    imageSize[3];
  double spacing[3];
  double origin[3];
  double clippingRange[2];
  double xAxis[3] = {1, 0, 0};
  double yAxis[3] = {0, 1, 0};

  clippingRange[0] = 1;
  clippingRange[1] = 100000;

  image->GetDimensions(imageSize);
  image->GetOrigin(origin);
  image->GetSpacing(spacing);

  windowSize[0] = m_RenderWindow->GetSize()[0];
  windowSize[1] = m_RenderWindow->GetSize()[1];

  niftk::SetCameraParallelTo2DImage(imageSize, windowSize, origin, spacing, xAxis, yAxis, clippingRange, m_FlipViewUp, *backCamera);
  niftk::SetCameraParallelTo2DImage(imageSize, windowSize, origin, spacing, xAxis, yAxis, clippingRange, m_FlipViewUp, *frontCamera);

  this->Modified();
}







