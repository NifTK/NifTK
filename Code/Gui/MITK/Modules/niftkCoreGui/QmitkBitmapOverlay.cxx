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
, m_BackCamera(NULL)
, m_FrontCamera(NULL)
, m_DataStorage(NULL)
, m_IsEnabled(false)
, m_Opacity(0.5)
, m_ImageDataNode(NULL)
, m_ImageData(NULL)
, m_UsingNVIDIA(false)
, m_ImageInNode(NULL)
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

  if ( m_ImageData != NULL)
  {
    delete[] m_ImageData;
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
bool QmitkBitmapOverlay::IsEnabled()
{
  return  m_IsEnabled;
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::Enable()
{
  if(m_IsEnabled)
    return;

 if(m_RenderWindow != NULL)
  {
    //check the data storage for a suitable target
    //if (!  m_ImageDataNode.IsNull() )
    if ( ! m_DataStorage.IsNull() )
    {
      m_ImageDataNode = m_DataStorage->GetNamedNode("NVIDIA SDI stream 0");
      if ( m_ImageDataNode.IsNotNull() )
      {
        m_UsingNVIDIA=true;
      }
      else
      {
        m_ImageDataNode = m_DataStorage->GetNamedNode("OpenCV image");
        m_UsingNVIDIA=false;
      }
      if ( ! m_ImageDataNode.IsNull() )
      {
        m_ImageInNode = dynamic_cast<mitk::Image*>(m_ImageDataNode->GetData());  
    
        if ( ! m_ImageInNode.IsNull() )
        {
          m_FrontActor->SetInput(m_ImageInNode->GetVtkImageData());
          m_BackActor->SetInput(m_ImageInNode->GetVtkImageData());


          m_BackActor->SetOpacity(1.0);
          m_FrontActor->SetOpacity(m_Opacity);

          m_BackRenderer->AddActor( m_BackActor );
          m_FrontRenderer->AddActor( m_FrontActor );
          m_BackRenderer->InteractiveOff();
          m_FrontRenderer->InteractiveOff();

          SetupCamera();

         // mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertBackgroundRenderer(m_BackRenderer,false);
          mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertForegroundRenderer(m_FrontRenderer,false);

          m_IsEnabled = true;
          m_DataStorage->ChangedNodeEvent.AddListener 
            (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
            (this, &QmitkBitmapOverlay::NodeChanged ) );
        }
      }

      m_DataStorage->AddNodeEvent.AddListener 
        (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
         (this, &QmitkBitmapOverlay::NodeAdded ) );

      m_DataStorage->RemoveNodeEvent.AddListener 
        (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
         (this, &QmitkBitmapOverlay::NodeRemoved ) );
    }

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
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::SetupCamera()
{
  // set the vtk camera in way that stretches the logo all over the renderwindow
 // m_RenderWindow->Render();
  vtkImageData * image = m_BackActor->GetInput();
  m_BackCamera = m_BackRenderer->GetActiveCamera();
  m_FrontCamera = m_FrontRenderer->GetActiveCamera();
  m_BackCamera->SetClippingRange(1,100000);
  m_FrontCamera->SetClippingRange(1,100000);

  if ( !image )
    return;

  double spacing[3];
  double origin[3];
  int   dimensions[3];

  image->GetSpacing(spacing);
  image->GetOrigin(origin);
  image->GetDimensions(dimensions);

  double focalPoint[3];
  double position[3];

  for ( unsigned int cc = 0; cc < 3; cc++)
  {
    focalPoint[cc] = origin[cc] + ( spacing[cc] * dimensions[cc] ) / 2.0;
    position[cc]   = focalPoint[cc];
  }

  int idx = 2;
  const double distanceToFocalPoint = 1000;
  if ( m_UsingNVIDIA )
  {
    m_BackCamera->SetViewUp (0,1,0);
    m_FrontCamera->SetViewUp (0,1,0);
    position[idx] = distanceToFocalPoint;
  }
  else
  {
    m_BackCamera->SetViewUp (0,-1,0);
    m_FrontCamera->SetViewUp (0,-1,0);
    position[idx] = -distanceToFocalPoint;
  }
  m_BackCamera->ParallelProjectionOn();
  m_BackCamera->SetPosition (position);
  m_BackCamera->SetFocalPoint (focalPoint);
  m_FrontCamera->ParallelProjectionOn();
  m_FrontCamera->SetPosition (position);
  m_FrontCamera->SetFocalPoint (focalPoint);

  int d1 = (idx + 1) % 3;
  int d2 = (idx + 2) % 3;

  double max = std::max(dimensions[d1],dimensions[d2]);
  std::cerr << "Max = " << max << std::endl;
  m_BackCamera->SetParallelScale( max / 2 );
  m_FrontCamera->SetParallelScale( max / 2 );
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::NodeChanged (const mitk::DataNode * node)
{
  if ( node == m_ImageDataNode ) 
  {
    if ( ! m_ImageInNode.IsNull() )
    {
      m_ImageInNode = dynamic_cast<mitk::Image*>(m_ImageDataNode->GetData());
      if ( ! m_ImageInNode.IsNull() )
      {
        m_BackActor->SetInput(m_ImageInNode->GetVtkImageData());
        m_FrontActor->SetInput(m_ImageInNode->GetVtkImageData());
      }
      m_ImageInNode->GetVtkImageData()->Modified();
    }
    else 
    {
      m_ImageInNode = dynamic_cast<mitk::Image*>(m_ImageDataNode->GetData());
      if ( ! m_ImageInNode.IsNull() )
      {
         m_FrontActor->SetInput(m_ImageInNode->GetVtkImageData());
         m_BackActor->SetInput(m_ImageInNode->GetVtkImageData());

         m_BackActor->SetOpacity(1.0);
         m_FrontActor->SetOpacity(m_Opacity);

         m_BackRenderer->AddActor( m_BackActor );
         m_FrontRenderer->AddActor( m_FrontActor );
         m_BackRenderer->InteractiveOff();
         m_FrontRenderer->InteractiveOff();

         SetupCamera();

        // mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertBackgroundRenderer(m_BackRenderer,false);
         mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertForegroundRenderer(m_FrontRenderer,false);

         m_IsEnabled = true;
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::NodeAdded (const mitk::DataNode * node)
{
  if ( m_ImageDataNode.IsNull() )
  {
    m_ImageDataNode = m_DataStorage->GetNamedNode("NVIDIA SDI stream 0");
    if ( m_ImageDataNode.IsNotNull() )
    {
      m_UsingNVIDIA=true;
    }
    else
    {
      m_UsingNVIDIA=false;
      m_ImageDataNode = m_DataStorage->GetNamedNode("OpenCV image");
    }
    if ( ! m_ImageDataNode.IsNull() )
    {
      mitk::Image::Pointer imageInNode; 
      m_ImageInNode = dynamic_cast<mitk::Image*>(m_ImageDataNode->GetData());  
    
      if ( ! m_ImageInNode.IsNull() )
      {
         m_FrontActor->SetInput(m_ImageInNode->GetVtkImageData());
         m_BackActor->SetInput(m_ImageInNode->GetVtkImageData());

         m_BackActor->SetOpacity(1.0);
         m_FrontActor->SetOpacity(m_Opacity);

         m_BackRenderer->AddActor( m_BackActor );
         m_FrontRenderer->AddActor( m_FrontActor );
         m_BackRenderer->InteractiveOff();
         m_FrontRenderer->InteractiveOff();

         SetupCamera();

       //  mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertBackgroundRenderer(m_BackRenderer,false);
         mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertForegroundRenderer(m_FrontRenderer,false);

         m_IsEnabled = true;
       }

      m_DataStorage->ChangedNodeEvent.AddListener 
        (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
        (this, &QmitkBitmapOverlay::NodeChanged ) );
     }
  }
}


//-----------------------------------------------------------------------------
void QmitkBitmapOverlay::NodeRemoved (const mitk::DataNode * node )
{
  if ( node == m_ImageDataNode )
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener 
      (mitk::MessageDelegate1<QmitkBitmapOverlay, const mitk::DataNode*>
      (this, &QmitkBitmapOverlay::NodeChanged ) );
   // mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_BackRenderer);
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_FrontRenderer);
    m_ImageDataNode = NULL; 
  }
}



