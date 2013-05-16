/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
/*!
* \file niftkVTKDistanceToSurface.cxx
* \page niftkVTKDistanceToSurface
* \section niftkVTKDistanceToSurface Summary Measure the distance between a set of points and a vtk poly data surface
*
* The distances are sorted and out put to std out or a specified file.
*
* \section niftkVTKIterativeClosestPointRegisterCaveat Caveats
* \li vtkIterativeClosestPointTransform is a point to surface iterative closest point algorithm.
* Therefore at least one of the input polydata must contain surfaces.
*
*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <niftkVTKDistanceToSurfaceCLP.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <niftkvtk4PointsReader.h>
#include <vtkFunctions.h>
#include <vtkSmartPointer.h>

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkDelaunay2D.h>
#include <vtkPolyDataMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkDoubleArray.h>
#include <vtkLookupTable.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>

#include <boost/algorithm/string/predicate.hpp>

/**
 * \brief Measure the distances of a set of points from a poly data surface
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  // Validate command line args
  if (targetPolyData.length() == 0 || sourcePoints.length() == 0)
    {
      commandLine.getOutput()->usage(commandLine);
      return EXIT_FAILURE;
    }

  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<niftkvtk4PointsReader> sourceReader = vtkSmartPointer<niftkvtk4PointsReader>::New();
  sourceReader->SetFileName(sourcePoints.c_str());
  sourceReader->Setm_ReadWeights(false);
  sourceReader->Update();
  source->ShallowCopy (sourceReader->GetOutput());
  std::cout << "Loaded text file:" << sourcePoints << std::endl;
  
  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  targetReader->SetFileName(targetPolyData.c_str());
  targetReader->Update();
  target->ShallowCopy (targetReader->GetOutput());
  std::cout << "Loaded PolyData:" << targetPolyData << std::endl;
  
  DistanceToSurface(source, target);



  if ( ! noVisualisation )
  {
    vtkSmartPointer<vtkPolyDataMapper> sourceMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
      
    std::cerr << "There are no cells in the source data, running delaunay filter\n";
    vtkSmartPointer<vtkDelaunay2D> delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
#if VTK_MAJOR_VERSION <= 5
    delaunay->SetInput(source);
    sourceMapper->SetInputConnection(delaunay->GetOutputPort());
    delaunay->Update();
#else
    delaunay->SetInputData(source);
    sourceMapper->SetInputData(delaunay);
    delaunay->Update();
#endif
    vtkSmartPointer<vtkActor> sourceActor = vtkSmartPointer<vtkActor>::New();
    sourceActor->SetMapper(sourceMapper);
    sourceActor->GetProperty()->SetColor(1,0,0);
    sourceActor->GetProperty()->SetPointSize(4);
    sourceActor->GetProperty()->SetRepresentationToPoints();

    vtkSmartPointer<vtkPolyDataMapper> targetMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    targetMapper->SetInputConnection(target->GetProducerPort());
#else
    targetMapper->SetInputData(target);
#endif

    vtkSmartPointer<vtkActor> targetActor = vtkSmartPointer<vtkActor>::New();
    targetActor->SetMapper(targetMapper);
    targetActor->GetProperty()->SetColor(0,1,0);
    targetActor->GetProperty()->SetOpacity(1.0);

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add the actor to the scene
    renderer->AddActor(sourceActor);
    renderer->AddActor(targetActor);
    renderer->SetBackground(.3, .6, .3); // Background color green

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();
  }
}
