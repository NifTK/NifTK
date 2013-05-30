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

  this->DeRegisterDataStorageListeners();

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
void QmitkBitmapOverlay::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
      (this, &QmitkBitmapOverlay::NodeAdded ) );

    m_DataStorage->RemoveNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
       (this, &QmitkBitmapOverlay::NodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
      (this, &QmitkBitmapOverlay::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::SetDataStorage (mitk::DataStorage::Pointer dataStorage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != dataStorage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = dataStorage;

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.AddListener
      (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
       (this, &QmitkBitmapOverlay::NodeAdded ) );

    m_DataStorage->RemoveNodeEvent.AddListener
      (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
       (this, &QmitkBitmapOverlay::NodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
      (this, &QmitkBitmapOverlay::NodeChanged ) );
  }

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
      this->SetFlipViewUp(false);
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
  this->AutoSelectDataNode(node);
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

  if(this->IsEnabled())
  {
    this->Disable();
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
        this->Enable();

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
  vtkCamera* backCamera = m_BackRenderer->GetActiveCamera();
  vtkCamera* frontCamera = m_FrontRenderer->GetActiveCamera();

  backCamera->SetClippingRange(1,100000);
  frontCamera->SetClippingRange(1,100000);

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

  double spacing[3];
  double origin[3];
  int    dimensions[3];

  image->GetSpacing(spacing);
  image->GetOrigin(origin);
  image->GetDimensions(dimensions);

  double focalPoint[3];
  double position[3];

  for ( unsigned int cc = 0; cc < 3; ++cc)
  {
    focalPoint[cc] = origin[cc] + ( spacing[cc] * (dimensions[cc] - 1) ) / 2.0;
    position[cc]   = focalPoint[cc];
  }

  int idx = 2;
  const double distanceToFocalPoint = 1000;

  if ( m_FlipViewUp )
  {
    backCamera->SetViewUp (0,-1,0);
    frontCamera->SetViewUp (0,-1,0);
    position[idx] = -distanceToFocalPoint;
  }
  else
  {
    backCamera->SetViewUp (0,1,0);
    frontCamera->SetViewUp (0,1,0);
    position[idx] = distanceToFocalPoint;
  }

  backCamera->ParallelProjectionOn();
  backCamera->SetPosition (position);
  backCamera->SetFocalPoint (focalPoint);

  frontCamera->ParallelProjectionOn();
  frontCamera->SetPosition (position);
  frontCamera->SetFocalPoint (focalPoint);

  int windowWidth = m_RenderWindow->GetSize()[0];
  int windowHeight = m_RenderWindow->GetSize()[1];

  double imageWidth = dimensions[0]*spacing[0];
  double imageHeight = dimensions[1]*spacing[1];

  double widthRatio = imageWidth / windowWidth;
  double heightRatio = imageHeight / windowHeight;

  double scale = 1;
  if (widthRatio > heightRatio)
  {
    scale = 0.5*imageWidth*((double)windowHeight/(double)windowWidth);
  }
  else
  {
    scale = 0.5*imageHeight;
  }

  backCamera->SetParallelScale( scale );
  frontCamera->SetParallelScale( scale );
  this->Modified();
}







