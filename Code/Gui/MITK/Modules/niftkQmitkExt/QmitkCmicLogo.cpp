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

#include "QmitkCmicLogo.h"

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
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkCubeSource.h>
#include <vtkObjectFactory.h>
#include <vtkRendererCollection.h>
#include <vtkPNGReader.h>
#include <vtkImageData.h>
#include <vtkConfigure.h>
#include <vtkImageFlip.h>

#include <mbilogo.h>

#include <algorithm>


CMICLogo::CMICLogo()
:m_ImageData(NULL)
{
  m_RenderWindow      = NULL;
  m_Renderer          = vtkRenderer::New();
  m_OuterCubeActor    = vtkActor::New();
  m_InnerCubeActor    = vtkActor::New();
  m_MiddleCubeActor    = vtkActor::New();
  m_Mapper            = vtkPolyDataMapper::New();
  m_PngReader         = vtkPNGReader::New();
  m_VtkImageImport    = vtkImageImport::New();
  
  m_LogoPosition  = CMICLogo::LowerRight;
 
  m_IsEnabled                  = false;
  m_ForceShowMBIDepartmentLogo = false;

  m_ZoomFactor = 1.15;
  m_Opacity    = 0.5;

  m_FileName  = "";
  m_PngReader->SetFileName(m_FileName.c_str());
}

CMICLogo::~CMICLogo()
{
  if ( m_RenderWindow != NULL )
    if ( this->IsEnabled() )
      this->Disable();
  
  if ( m_Mapper != NULL )
    m_Mapper->Delete();
  
  if ( m_OuterCubeActor!=NULL )
    m_OuterCubeActor->Delete();
  if ( m_InnerCubeActor!=NULL )
    m_InnerCubeActor->Delete();
  if ( m_MiddleCubeActor!=NULL )
    m_MiddleCubeActor->Delete();
  
  if ( m_Renderer != NULL )
    m_Renderer->Delete();

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
void CMICLogo::SetRenderWindow( vtkRenderWindow* renderWindow )
{
  m_RenderWindow = renderWindow;
}

/**
 * Returns the vtkRenderWindow, which is used
 * for displaying the logo
 */
vtkRenderWindow* CMICLogo::GetRenderWindow()
{
  return m_RenderWindow;
}

/**
 * Returns the renderer responsible for
 * rendering the  logo into the
 * vtkRenderWindow
 */
vtkRenderer* CMICLogo::GetVtkRenderer()
{
  return m_Renderer;
}

/**
 * Returns the actor associated with the  logo
 */
vtkActor* CMICLogo::GetActor()
{
  return m_OuterCubeActor;
}

/**
 * Returns the mapper associated with the 
 * logo.
 */
vtkPolyDataMapper* CMICLogo::GetMapper()
{
  return m_Mapper;
}

void CMICLogo::SetLogoSource(const char* filename)
{
  std::string file = filename;
  if(file.length() != 0)
  {
    m_FileName  = filename;
    m_PngReader->SetFileName(m_FileName.c_str());
  }
}

/**
 * Enables drawing of the logo.
 * If you want to disable it, call the Disable() function.
 */
void CMICLogo::Enable()
{
  if(m_IsEnabled)
    return;

  if(m_RenderWindow != NULL)
  {
    vtkCubeSource* OuterBox = vtkCubeSource::New();
  //  OuterBox->SetHeight(10);
    OuterBox->SetXLength(100);
    OuterBox->SetYLength(100);
    OuterBox->SetZLength(100);
    OuterBox->SetCenter(1500,-800,100);
    vtkPolyDataMapper* OuterBoxMapper = vtkPolyDataMapper::New();
    OuterBoxMapper->SetInput(OuterBox->GetOutput());
    m_OuterCubeActor->SetMapper(OuterBoxMapper);
    m_OuterCubeActor->GetProperty()->SetOpacity(0.10);


 //   m_OuterCubeActor->SetOpacity(m_Opacity);
    
    m_Renderer->AddActor( m_OuterCubeActor );
    m_Renderer->InteractiveOff();
    
    SetupCamera();
  //  SetupPosition();
    
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertForegroundRenderer(m_Renderer,false);
    
    m_IsEnabled = true;
  }
}


void CMICLogo::SetupCamera()
{
  // set the vtk camera in way that stretches the logo all over the renderwindow

  m_Camera = m_Renderer->GetActiveCamera();
  m_Camera->SetClippingRange(1,100000);

  double focalPoint[3];
  double position[3];
  
  position[0] = 0; 
  position[1] = 0;
  position[2] = 0;

  focalPoint[0] = 0;
  focalPoint[0] = 0;
  focalPoint[0] = 1;

  m_Camera->SetViewUp (0,1,0);

  m_Camera->ParallelProjectionOn();
  m_Camera->SetPosition (position);
  m_Camera->SetFocalPoint (focalPoint);

  m_Camera->SetParallelScale( 1000 );
}

void CMICLogo::SetupPosition()
{ // Position and Scale of the logo
 // m_Renderer->SetViewport(newPos);
}

void CMICLogo::ForceMBILogoVisible(bool visible)
{
  m_ForceShowMBIDepartmentLogo = visible;
}

void CMICLogo::SetZoomFactor( double factor )
{
  m_ZoomFactor = factor;
}
void CMICLogo::SetOpacity(double opacity)
{
  m_Opacity = opacity;
}

/**
 * Disables drawing of the logo.
 * If you want to enable it, call the Enable() function.
 */
void CMICLogo::Disable()
{
  if ( this->IsEnabled() && !m_ForceShowMBIDepartmentLogo )
  {
    mitk::VtkLayerController::GetInstance(m_RenderWindow)->RemoveRenderer(m_Renderer);
    m_IsEnabled = false;
  }
}

/**
 * Checks, if the logo is currently
 * enabled (visible)
 */
bool CMICLogo::IsEnabled()
{
  return  m_IsEnabled;
}


void CMICLogo::SetRequestedRegionToLargestPossibleRegion()
{
    //nothing to do
}

bool CMICLogo::RequestedRegionIsOutsideOfTheBufferedRegion()
{
    return false;    
}

bool CMICLogo::VerifyRequestedRegion()
{
    return true;
}

void CMICLogo::SetRequestedRegion(itk::DataObject*)
{
    //nothing to do
}

