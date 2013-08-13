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
#include <niftkConversionUtils.h>
#include <niftkVTKDistanceToSurfaceCLP.h>
#include <niftkVTK4PointsReader.h>
#include <niftkVTKFunctions.h>

#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedIntArray.h>
#include <vtkSortDataArray.h>
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
#include <vtkLookupTable.h>

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

  std::ostream * fp = &std::cout;
  std::ofstream fout;
  if ( output.length() != 0 ) 
  {
    fout.open(output.c_str() );
    fp=&fout;
  }
  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<niftk::VTK4PointsReader> sourceReader = vtkSmartPointer<niftk::VTK4PointsReader>::New();
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
  
  niftk::DistanceToSurface(source, target);

  vtkSmartPointer<vtkDoubleArray> distancesArray = vtkSmartPointer<vtkDoubleArray>::New();
  vtkSmartPointer<vtkUnsignedIntArray> idarray = vtkSmartPointer<vtkUnsignedIntArray>::New();

  distancesArray->SetNumberOfComponents(1);
  idarray->SetNumberOfComponents(1);
  double max_dist = 0;
  double min_dist = 0;
  for ( int i = 0 ; i < source->GetNumberOfPoints() ; i ++ )
  {
    double distance = source->GetPointData()->GetScalars()->GetComponent(i,0);
    if ( i == 0 )
    {
      max_dist = distance;
      min_dist = distance;
    }
    else
    {
      min_dist = distance < min_dist ? distance : min_dist;
      max_dist = distance > max_dist ? distance : max_dist;
    }
    distancesArray->InsertNextValue(distance);
    idarray->InsertNextValue(i);
  }

  std::cout << "Max = " << max_dist << "  Min = " << min_dist << std::endl;

  vtkSmartPointer<vtkSortDataArray> sorter = vtkSmartPointer<vtkSortDataArray>::New();
  sorter->Sort(distancesArray,  idarray);

  double p[3];
  for ( int i = 0 ; i < source->GetNumberOfPoints() ; i++ )
  {
    source->GetPoint( static_cast<vtkIdType>(idarray->GetComponent(i,0)), p);
    *fp << idarray->GetComponent(i,0) << "\t" << p[0] << "\t" << p[1] << "\t" << p[2];
    *fp << "\t" << distancesArray->GetComponent(i,0) << std::endl;
  }
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
    //build a lookup table
    vtkSmartPointer<vtkLookupTable> colorLookupTable = vtkSmartPointer<vtkLookupTable>::New();
    colorLookupTable->SetTableRange ( min_dist, max_dist);
    colorLookupTable->Build();
    sourceMapper->SetLookupTable (colorLookupTable);
    vtkSmartPointer<vtkActor> sourceActor = vtkSmartPointer<vtkActor>::New();
    sourceActor->SetMapper(sourceMapper);
    sourceActor->GetProperty()->SetColor(1,0,0);
    sourceActor->GetProperty()->SetPointSize(1);
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
    targetActor->GetProperty()->SetOpacity(0.5);

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
