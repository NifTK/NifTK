/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkMakeGeometry.h>
#include <mitkIOUtil.h>
#include <niftkIGIMakeGeometryCLP.h>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;
  bool geomOK = false;
  //mitk::Surface::Pointer surface = mitk::Surface::New();
  mitk::Surface::Pointer surface = NULL;
  if ( geometry == "backwall" )
  {
    surface = MakeAWall(0);
  }
  if ( geometry == "frontwall" )
  {
    surface = MakeAWall(2);
  }
  if ( geometry == "leftwall" )
  {
    surface = MakeAWall(1);
  }
  if ( geometry == "rightwall" )
  {
    surface = MakeAWall(3);
  }
  if ( geometry == "ceiling" )
  {
    surface = MakeAWall(4);
  }
  if ( geometry == "floor" )
  {
    surface = MakeAWall(5);
  }
  if ( geometry == "laparoscope" )
  {
    surface = MakeLaparoscope (rigidBodyFile, handeye);
  }
  if ( geometry == "pointer" )
  {
    surface = MakePointer (rigidBodyFile, handeye);
  }
  if ( geometry == "reference" )
  {
    surface = MakeReference (rigidBodyFile, handeye);
  }
  if ( geometry == "XAxis" )
  {
    surface = MakeXAxes ();
  }
  if ( geometry == "YAxis" )
  {
    surface = MakeYAxes ();
  }
  if ( geometry == "ZAxis" )
  {
    surface = MakeZAxes ();
  }
  if ( geometry == "laplensAxes" )
  {
    surface = MakeLapLensAxes ();
  }
  if ( geometry == "optotrak" )
  {
    surface = MakeOptotrak ();
  }
  if ( geometry == "transRectalUSProbe" )
  {
    surface = MakeTransrectalUSProbe (handeye);
  }
  if ( surface.IsNull() ) 
  {
    MITK_ERROR << "Failed to make specified geometry, available options are: ";
    MITK_INFO << " backwall frontwall leftwall rightwall ceiling floor";
    MITK_INFO << " laparoscope pointer reference";
    MITK_INFO << " XAxis YAxis ZAxis laplensAxes";
    MITK_INFO << " optotrak transRectalUSProbe";
    exit (EXIT_FAILURE);
  }

  mitk::IOUtil::SaveSurface (surface,output);
  if ( Visualise )
  {
    vtkSmartPointer<vtkPolyData> vtkSurface = surface->GetVtkPolyData();
    vtkSmartPointer<vtkPolyDataMapper> sourceMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
   // sourceMapper->SetInputData(vtkSurface);
    sourceMapper->SetInputConnection(vtkSurface->GetProducerPort());
    vtkSmartPointer<vtkActor> sourceActor = vtkSmartPointer<vtkActor>::New();
    sourceActor->SetMapper(sourceMapper);
    sourceActor->GetProperty()->SetColor(1,1,1);
    sourceActor->GetProperty()->SetPointSize(4);
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);
    renderer->AddActor(sourceActor);
    renderer->SetBackground(.96, .04, .84); // Background a pleasant shade CMIC pink
    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();
     
  }



}
