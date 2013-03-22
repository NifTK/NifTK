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

#include "mitkVtkLayerController.h"

#include <mitkStandardFileLocations.h>
#include <mitkConfig.h>
#include <itkObject.h>
#include <itkMacro.h>
#include <itksys/SystemTools.hxx>

#include <vtkImageImport.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkMapper.h>
#include <vtkImageActor.h>
#include <vtkImageMapper.h>
#include <vtkPolyData.h>
#include <vtkCamera.h>
#include <vtkObjectFactory.h>
#include <vtkRendererCollection.h>
#include <vtkPNGReader.h>
#include <vtkImageData.h>
#include <vtkConfigure.h>
#include <vtkImageFlip.h>

#include <mbilogo.h>

#include <algorithm>


BitmapOverlay::BitmapOverlay()
:m_ImageData(NULL)
,m_ImageDataNode(NULL)
{
  m_RenderWindow      = NULL;
  m_BackRenderer          = vtkRenderer::New();
  m_FrontRenderer          = vtkRenderer::New();
  m_BackActor             = vtkImageActor::New();
  m_FrontActor             = vtkImageActor::New();
  m_Mapper            = vtkImageMapper::New();
  m_PngReader         = vtkPNGReader::New();
  m_VtkImageImport    = vtkImageImport::New();

  m_LogoPosition  = BitmapOverlay::LowerRight;

  m_IsEnabled                  = false;
  m_ForceShowMBIDepartmentLogo = false;

  m_ZoomFactor = 1.15;
  m_Opacity    = 0.5;

  m_FileName  = "";
  m_PngReader->SetFileName(m_FileName.c_str());
}

BitmapOverlay::~BitmapOverlay()
{
  if ( m_RenderWindow != NULL )
    if ( this->IsEnabled() )
      this->Disable();

  if ( m_Mapper != NULL )
    m_Mapper->Delete();

  if ( m_BackActor!=NULL )
    m_BackActor->Delete();

  if ( m_FrontActor!=NULL )
    m_FrontActor->Delete();
  if ( m_BackRenderer != NULL )
    m_BackRenderer->Delete();
  if ( m_FrontRenderer != NULL )
    m_FrontRenderer->Delete();

  if ( m_PngReader != NULL )
    m_PngReader->Delete();

  if ( m_VtkImageImport != NULL )
    m_VtkImageImport->Delete();

  if ( m_ImageData != NULL)
    delete[] m_ImageData;
}

/**
 * Sets the renderwindow, in which the logo
 * will be shown. Make sure, you have called this function
 * before calling Enable()
 */
void BitmapOverlay::SetRenderWindow( vtkRenderWindow* renderWindow )
{
  m_RenderWindow = renderWindow;
}

/**
 * Returns the vtkRenderWindow, which is used
 * for displaying the logo
 */
vtkRenderWindow* BitmapOverlay::GetRenderWindow()
{
  return m_RenderWindow;
}

/**
 * Returns the renderer responsible for
 * rendering the  logo into the
 * vtkRenderWindow
 */
vtkRenderer* BitmapOverlay::GetVtkRenderer()
{
  return m_BackRenderer;
}

/**
 * Returns the actor associated with the  logo
 */
vtkImageActor* BitmapOverlay::GetActor()
{
  return m_BackActor;
}

/**
 * Returns the mapper associated with the
 * logo.
 */
vtkImageMapper* BitmapOverlay::GetMapper()
{
  return m_Mapper;
}

void BitmapOverlay::SetLogoSource(const char* filename)
{
  std::string file = filename;
  if(file.length() != 0)
  {
    m_FileName  = filename;
    m_PngReader->SetFileName(m_FileName.c_str());
  }
}

/**
 * Enables drawing of the overlay
 * If you want to disable it, call the Disable() function.
 */
void BitmapOverlay::Enable()
{
  if(m_IsEnabled)
    return;

  if(m_RenderWindow != NULL)
  {
    //check the data storage for a suitable target
    if (!  m_ImageDataNode.IsNull() )
    {
     // mitk::BaseData * data = m_ImageDataNode->GetData();
      mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(m_ImageDataNode->GetData());  
    
      if ( ! imageInNode.IsNull() )
      {
        m_BackActor->SetInput(imageInNode->GetVtkImageData());
        m_FrontActor->SetInput(imageInNode->GetVtkImageData());
        if ( ! m_DataStorage.IsNull() )
        {
          m_DataStorage->ChangedNodeEvent.AddListener (mitk::MessageDelegate1<BitmapOverlay, const mitk::DataNode*> 
            (this, &BitmapOverlay::NodeChangedProxy ) );
         }
      }

    }

    m_BackActor->SetOpacity(1.0);
    m_FrontActor->SetOpacity(m_Opacity);

    m_BackRenderer->AddActor( m_BackActor );
    m_FrontRenderer->AddActor( m_FrontActor );
    m_BackRenderer->InteractiveOff();
    m_FrontRenderer->InteractiveOff();

    SetupCamera();
    SetupPosition();

    mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertBackgroundRenderer(m_BackRenderer,false);
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertForegroundRenderer(m_FrontRenderer,false);

    m_IsEnabled = true;
  }
}

void BitmapOverlay::NodeChangedProxy (const mitk::DataNode * node)
{
  if ( node == m_ImageDataNode ) 
  {
    mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(m_ImageDataNode->GetData());
    if ( ! imageInNode.IsNull() )
    {
       m_BackActor->SetInput(imageInNode->GetVtkImageData());
       m_FrontActor->SetInput(imageInNode->GetVtkImageData());
    }
    imageInNode->GetVtkImageData()->Modified();
  }
}

void BitmapOverlay::SetupCamera()
{
  // set the vtk camera in way that stretches the logo all over the renderwindow

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


  m_BackCamera->SetViewUp (0,1,0);
  m_FrontCamera->SetViewUp (0,1,0);
  int idx = 2;
  const double distanceToFocalPoint = 1000;
  position[idx] = distanceToFocalPoint;

  m_BackCamera->ParallelProjectionOn();
  m_BackCamera->SetPosition (position);
  m_BackCamera->SetFocalPoint (focalPoint);
  m_FrontCamera->ParallelProjectionOn();
  m_FrontCamera->SetPosition (position);
  m_FrontCamera->SetFocalPoint (focalPoint);

  int d1 = (idx + 1) % 3;
  int d2 = (idx + 2) % 3;

  double max = std::max(dimensions[d1],dimensions[d2]);

  m_BackCamera->SetParallelScale( max / 2 );
  m_FrontCamera->SetParallelScale( max / 2 );
}

void BitmapOverlay::SetupPosition()
{ // Position and Scale of the logo

  double newPos[4];
  int dimensions[3];
  vtkImageData * image = m_BackActor->GetInput();
  image->GetDimensions(dimensions);
  // normalize image dimensions
  double max = std::max(dimensions[0],dimensions[1]);
  double normX = dimensions[0] / max;
  double normY = dimensions[1] / max;

  double buffer = 0; // buffer to the boarder of the renderwindow

  newPos[0] = (0 + buffer);
  newPos[1] = (0 + buffer);
  newPos[2] = 1.0 * normX * m_ZoomFactor;
  newPos[3] = 1.0 * normY * m_ZoomFactor;

  m_BackRenderer->SetViewport(newPos);
  m_FrontRenderer->SetViewport(newPos);
}

void BitmapOverlay::ForceMBILogoVisible(bool visible)
{
  m_ForceShowMBIDepartmentLogo = visible;
}

void BitmapOverlay::SetZoomFactor( double factor )
{
  m_ZoomFactor = factor;
}
void BitmapOverlay::SetOpacity(double opacity)
{
  m_Opacity = opacity;
}

/**
 * Disables drawing of the logo.
 * If you want to enable it, call the Enable() function.
 */
void BitmapOverlay::Disable()
{
  if ( this->IsEnabled() && !m_ForceShowMBIDepartmentLogo )
  {
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_BackRenderer);
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_FrontRenderer);
    m_IsEnabled = false;
  }
}

/**
 * Checks, if the logo is currently
 * enabled (visible)
 */
bool BitmapOverlay::IsEnabled()
{
  return  m_IsEnabled;
}


void BitmapOverlay::SetRequestedRegionToLargestPossibleRegion()
{
    //nothing to do
}

bool BitmapOverlay::RequestedRegionIsOutsideOfTheBufferedRegion()
{
    return false;
}

bool BitmapOverlay::VerifyRequestedRegion()
{
    return true;
}

void BitmapOverlay::SetRequestedRegion(itk::DataObject*)
{
    //nothing to do
}

void BitmapOverlay::SetDataNode (mitk::DataNode::Pointer dataNode)
{
  this->m_ImageDataNode = dataNode;
}
void BitmapOverlay::SetDataStorage (mitk::DataStorage::Pointer dataStorage)
{
  this->m_DataStorage = dataStorage;
}
