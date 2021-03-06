/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
/*!
* \file niftkVTKIterativeClosestPointRegister.cxx
* \page niftkVTKIterativeClosestPointRegister
* \section niftkVTKIterativeClosestPointRegsisterSummary Uses vtkIterativeClosestPointTransform to register two vtk polydata sets
*
* This program uses vtkIterativeClosestPointTransform via niftkVTKIterativeClosestPoint.
* Optionally the transformed source may be written to a vtkpolydata file.
*
* \section niftkVTKIterativeClosestPointRegisterCaveat Caveats
* \li vtkIterativeClosestPointTransform is a point to surface iterative closest point algorithm.
* Therefore at least one of the input polydata must contain surfaces.
*
*/

#include <niftkVTKIterativeClosestPointRegisterCLP.h>
#include <niftkVTKIterativeClosestPoint.h>
#include <niftkVTK4PointsReader.h>
#include <niftkVTKFunctions.h>

#include <niftkConversionUtils.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
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
#include <vtkMinimalStandardRandomSequence.h>

#include <boost/algorithm/string/predicate.hpp>

struct arguments
{
  std::string targetPolyDataFile;
  std::string sourcePolyDataFile;
  int maxIterations;
  int maxPoints;
  bool visualise;
  bool randomTransform;
  bool perturbTarget;
  bool writeout;
  std::string outPolyDataFile;
};


// Define interaction style
class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
public:
  static KeyPressInteractorStyle* New();
  vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);

  virtual void OnKeyPress() override
  {
    // Get the keypress
    vtkRenderWindowInteractor *rwi = this->Interactor;
    std::string key = rwi->GetKeySym();

    if(key == "Right")
    {
      if ( target->GetProperty()->GetOpacity() == 0.0 )
      {
        target->GetProperty()->SetOpacity(1.0);
        source->GetProperty()->SetOpacity(0.2);
      }
      else
      {
        if ( solution->GetProperty()->GetOpacity() < 1.0 )
        {
          solution->GetProperty()->SetOpacity(solution->GetProperty()->GetOpacity() + 0.5);
        }
      }
    }
    if(key == "Left")
    {
      if ( solution->GetProperty()->GetOpacity() > 0.0 )
      {
        solution->GetProperty()->SetOpacity(solution->GetProperty()->GetOpacity() - 0.5);
      }
      else
      {
        if ( target->GetProperty()->GetOpacity() == 1.0 )
        {
          target->GetProperty()->SetOpacity(0.0);
          source->GetProperty()->SetOpacity(1.0);
        }
      }
    }
    // Forward events
    vtkInteractorStyleTrackballCamera::OnKeyPress();
    this->GetCurrentRenderer()->GetRenderWindow()->Render();
  }

  vtkActor *source;
  vtkActor *solution;
  vtkActor *target;
};
vtkStandardNewMacro(KeyPressInteractorStyle);

/**
 * \brief Run a VTK ICP on two poly data
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;
  args.maxIterations = maxNumberOfIterations;
  args.maxPoints = maxNumberOfPoints;
  args.visualise = !noVisualisation;
  args.randomTransform = rndTrans;
  args.perturbTarget = rndPerturb;
  args.writeout = outputTransformedSource.length() > 0;
  args.sourcePolyDataFile = sourcePolyData;
  args.targetPolyDataFile = targetPolyData;
  args.outPolyDataFile = outputTransformedSource;

  // Validate command line args
  if (args.sourcePolyDataFile.length() == 0 || args.targetPolyDataFile.length() == 0)
    {
      commandLine.getOutput()->usage(commandLine);
      return EXIT_FAILURE;
    }

  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> c_source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> solution = vtkSmartPointer<vtkPolyData>::New();

  if ( boost::algorithm::ends_with(args.sourcePolyDataFile , ".txt") )
  {
    vtkSmartPointer<niftk::VTK4PointsReader> sourceReader = vtkSmartPointer<niftk::VTK4PointsReader>::New();
    sourceReader->SetFileName(args.sourcePolyDataFile.c_str());
    sourceReader->SetClippingOn(2,50, 200);
    sourceReader->Update();
    source->ShallowCopy (sourceReader->GetOutput());
    std::cout << "Loaded text file:" << args.sourcePolyDataFile << std::endl;
  }
  else
  {
    vtkSmartPointer<vtkPolyDataReader> sourceReader = vtkSmartPointer<vtkPolyDataReader>::New();
    sourceReader->SetFileName(args.sourcePolyDataFile.c_str());
    sourceReader->Update();
    source->ShallowCopy (sourceReader->GetOutput());
    std::cout << "Loaded PolyData:" << args.sourcePolyDataFile << std::endl;
  }
  if ( args.randomTransform )
  {
    c_source->DeepCopy (source);
  }

  if ( boost::algorithm::ends_with(args.targetPolyDataFile , ".txt") )
  {
    vtkSmartPointer<niftk::VTK4PointsReader> targetReader = vtkSmartPointer<niftk::VTK4PointsReader>::New();
    targetReader->SetFileName(args.targetPolyDataFile.c_str());
    targetReader->SetClippingOn(2,50, 200);
    targetReader->Update();
    target->ShallowCopy (targetReader->GetOutput());
    std::cout << "Loaded text file:" << args.targetPolyDataFile << std::endl;
  }
  else
  {
    vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
    targetReader->SetFileName(args.targetPolyDataFile.c_str());
    targetReader->Update();
    target->ShallowCopy (targetReader->GetOutput());
    std::cout << "Loaded PolyData:" << args.targetPolyDataFile << std::endl;
  }

  niftk::VTKIterativeClosestPoint * icp = new niftk::VTKIterativeClosestPoint();
  icp->SetICPMaxLandmarks(args.maxPoints);
  icp->SetICPMaxIterations(args.maxIterations);
  icp->SetSource(source);
  icp->SetTarget(target);


  vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New();
  if ( args.randomTransform )
  {
    vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
    Uni_Rand->SetSeed(time(NULL));
    double scaleSD = -1.0;
    StartTrans = niftk::RandomTransform ( 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0, *Uni_Rand, scaleSD);
    niftk::TranslatePolyData ( source , StartTrans);
  }
  if ( args.perturbTarget )
  {
    niftk::PerturbPolyData(target, 1.0, 1.0 , 1.0);
  }
  icp->Run();

  vtkSmartPointer<vtkMatrix4x4> m = icp->GetTransform();
  std::cout << "The Resulting transform is " << *m << std::endl;
  if ( args.randomTransform || args.visualise )
  {
    icp->ApplyTransform(solution);
  }
  //If testing with random transform put out an error metric
  if ( args.randomTransform )
  {
    vtkSmartPointer<vtkMatrix4x4> Residual  = vtkSmartPointer<vtkMatrix4x4>::New();
    StartTrans->Concatenate(m);
    StartTrans->GetInverse(Residual);
    double * StartPoint = new double [4];
    double * EndPoint = new double [4];
    StartPoint [0 ] = 160;
    StartPoint [1] = 80;
    StartPoint [2] = 160;
    StartPoint [3] = 1;
    EndPoint= Residual->MultiplyDoublePoint(StartPoint);
    double MagError = 0;
    for ( int i = 0; i < 4; i ++ )
    {
      MagError += (EndPoint[i] - StartPoint[i]) * ( EndPoint[i] - StartPoint[i]);
    }
    MagError = sqrt(MagError);
    std::cout << "Residual Error = "  << MagError << std::endl;
    std::cout << "Residual Transform = " << *Residual;
    //do a difference image
    niftk::DistancesToColorMap(c_source, solution);

  }

  if ( args.writeout == true )
  {
     vtkSmartPointer<vtkPolyData> solution = vtkSmartPointer<vtkPolyData>::New();
     icp->ApplyTransform(solution);
     vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
     writer->SetInputDataObject(solution);
     writer->SetFileName(args.outPolyDataFile.c_str());
     writer->Update();
  }
  if ( args.visualise )
  {

    vtkSmartPointer<vtkPolyDataMapper> sourceMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    if ( source->GetNumberOfCells() == 0 )
    {
      std::cerr << "There are no cells in the source data, running delaunay filter\n";
      vtkSmartPointer<vtkDelaunay2D> delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
      delaunay->SetInputDataObject(source);
      sourceMapper->SetInputDataObject(delaunay->GetOutput());
      delaunay->Update();
    }
    else
    {
      sourceMapper->SetInputDataObject(source);
    }
    vtkSmartPointer<vtkActor> sourceActor = vtkSmartPointer<vtkActor>::New();
    sourceActor->SetMapper(sourceMapper);
    sourceActor->GetProperty()->SetColor(1,0,0);
    sourceActor->GetProperty()->SetPointSize(4);

    vtkSmartPointer<vtkPolyDataMapper> targetMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    if ( target->GetNumberOfCells() == 0 )
    {
      std::cerr << "There are no cells in the target data, running delaunay filter\n";
      vtkSmartPointer<vtkDelaunay2D> delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
      delaunay->SetInputDataObject(target);
      targetMapper->SetInputDataObject(delaunay->GetOutput());
      delaunay->Update();
    }
    else
    {
      targetMapper->SetInputDataObject(target);
    }

    vtkSmartPointer<vtkActor> targetActor = vtkSmartPointer<vtkActor>::New();
    targetActor->SetMapper(targetMapper);
    targetActor->GetProperty()->SetColor(0,1,0);
    targetActor->GetProperty()->SetPointSize(4);
    targetActor->GetProperty()->SetOpacity(0.0);
    targetActor->GetProperty()->SetRepresentationToPoints();

    vtkSmartPointer<vtkPolyDataMapper> solutionMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    if ( solution->GetNumberOfCells() == 0 )
    {
      vtkSmartPointer<vtkDelaunay2D> delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
      delaunay->SetInputDataObject(solution);
      solutionMapper->SetInputDataObject(delaunay->GetOutput());
      delaunay->Update();
    }
    else
    {
      solutionMapper->SetInputDataObject(solution);
    }
    vtkSmartPointer<vtkActor> solutionActor =
    vtkSmartPointer<vtkActor>::New();
    solutionActor->SetMapper(solutionMapper);
    solutionActor->GetProperty()->SetColor(0,0,1);
    solutionActor->GetProperty()->SetOpacity(0.0);
    solutionActor->GetProperty()->SetPointSize(3);

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkSmartPointer<KeyPressInteractorStyle> style = vtkSmartPointer<KeyPressInteractorStyle>::New();
    style->target = targetActor;
    style->source = sourceActor;
    style->solution = solutionActor;
    renderWindowInteractor->SetInteractorStyle(style);
    style->SetCurrentRenderer(renderer);

    // Add the actor to the scene
    renderer->AddActor(sourceActor);
    renderer->AddActor(targetActor);
    renderer->AddActor(solutionActor);
    renderer->SetBackground(.3, .6, .3); // Background color green

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();

  }
}
