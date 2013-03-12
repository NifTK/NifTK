/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkTransformPolyDataFilter.h"
#include "niftkVTKIterativeClosestPoint.h"
#include "vtkFunctions.h"

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkDelaunay2D.h>
#include <vtkPolyDataMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>

/*!
 * \file niftkVTKIterativeClosestPointRegister.cxx
 * \page niftkVTKIterativeClosestPointRegister
 * \section niftkVTKIterativeClosestPointRegsisterSummary Uses vtkIterativeClosestPointTransform to register to VTK poly data sets
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Register two VTK polydata objects using iterative closest points." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -t targetPolyData.vtk -s sourcePolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    --target    <filename>        Target VTK Poly Data." << std::endl;
    std::cout << "    --source    <filename>        Source VTK Poly Data." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    --novisualise                 Turn off visualisation" << std::endl;
    std::cout << "    --maxpoints <pointstouse>     Set the maximum number of points to use in the registration, default is " << __NIFTTKVTKICPNPOINTS <<  std::endl;
    std::cout << "    --maxit     <maxiterations>   Set the maximum number of iterations to use, default is " << __NIFTTKVTKICPMAXITERATIONS << std::endl << std::endl;
    std::cout << "*** [for testing]   ***" << std::endl << std::endl;
    std::cout << "    --rndtrans                    Transform the source with random transform prior to running" << std::endl;
    std::cout << "    --perturb                     randomly perturb the target points prior to registration" << std::endl << std::endl;


  }

struct arguments
{
  std::string targetPolyDataFile;
  std::string sourcePolyDataFile;
  int maxIterations;
  int maxPoints;
  bool visualise;
  bool randomTransform;
  bool perturbTarget;
};


// Define interaction style
class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
  public:
    static KeyPressInteractorStyle* New();
    vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera);

    virtual void OnKeyPress() 
    {
      // Get the keypress
      vtkRenderWindowInteractor *rwi = this->Interactor;
      std::string key = rwi->GetKeySym();
      
      // Output the key that was pressed
      std::cout << "Pressed " << key << std::endl;
      
      if(key == "Right")
      {
        if ( target->GetProperty()->GetOpacity() == 0.0 )
        {
          target->GetProperty()->SetOpacity(1.0);
        }
        else
        {
          if ( solution->GetProperty()->GetOpacity() == 0.0 )
          {
            solution->GetProperty()->SetOpacity(1.0);
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
  // To pass around command line args
  struct arguments args;
  args.maxIterations = __NIFTTKVTKICPMAXITERATIONS;
  args.maxPoints = __NIFTTKVTKICPNPOINTS;
  args.visualise = true;
  args.randomTransform = false;
  args.perturbTarget = false;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "--target") == 0){
      args.targetPolyDataFile=argv[++i];
      std::cout << "Set --target=" << args.targetPolyDataFile << std::endl;
    }
    else if(strcmp(argv[i], "--source") == 0){
      args.sourcePolyDataFile=argv[++i];
      std::cout << "Set --source=" << args.sourcePolyDataFile << std::endl;
    }
    else if(strcmp(argv[i], "--novisualise") == 0){
      args.visualise = false;
      std::cout << "Set Visualise off" << std::endl;
    }    
    else if(strcmp(argv[i], "--maxpoints") == 0){
      args.maxPoints = atoi(argv[++i]);
      std::cout << "Set max points to " << args.maxPoints << std::endl;
    }    
    else if(strcmp(argv[i], "--maxit") == 0){
      args.maxIterations = atoi(argv[++i]);
      std::cout << "Set max iterations to " << args.maxIterations << std::endl;
    }    
    else if(strcmp(argv[i], "--rndtrans") == 0){
      args.randomTransform = true;
      std::cout << "Set random transform on" << std::endl;
    }    
    else if(strcmp(argv[i], "--perturb") == 0){
      args.perturbTarget = true;
      std::cout << "Set perturb target on" << std::endl;
    }    
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.sourcePolyDataFile.length() == 0 || args.targetPolyDataFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();
  
  std::cout << "Loading PolyData:" << args.sourcePolyDataFile << std::endl;
  vtkSmartPointer<vtkPolyDataReader> sourceReader = vtkSmartPointer<vtkPolyDataReader>::New();
  sourceReader->SetFileName(args.sourcePolyDataFile.c_str());
  sourceReader->Update();
  source->ShallowCopy (sourceReader->GetOutput()); 
  std::cout << "Loaded PolyData:" << args.sourcePolyDataFile << std::endl;
  
  std::cout << "Loading PolyData:" << args.targetPolyDataFile << std::endl;
  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  targetReader->SetFileName(args.targetPolyDataFile.c_str());
  targetReader->Update();
  target->ShallowCopy (targetReader->GetOutput()); 
  std::cout << "Loaded PolyData:" << args.targetPolyDataFile << std::endl;
  
  niftk::IterativeClosestPoint * icp = new niftk::IterativeClosestPoint(); 
  icp->SetMaxLandmarks(args.maxPoints);
  icp->SetMaxIterations(args.maxIterations);
  icp->SetSource(source);
  icp->SetTarget(target);
  
  if ( args.randomTransform )
  {
    vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New(); 
    RandomTransform ( StartTrans, 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0 );
    TranslatePolyData ( source , StartTrans);

  }
  if ( args.perturbTarget ) 
  {
    PerturbPolyData(target, 1.0, 1.0 , 1.0);
  }
  icp->Run();

  if ( args.visualise ) 
  {
    
    vtkSmartPointer<vtkPolyDataMapper> sourceMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    if ( source->GetNumberOfCells() == 0 )
    {
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
    }
    else
    {
#if VTK_MAJOR_VERSION <= 5
      sourceMapper->SetInputConnection(source->GetProducerPort());
#else
      sourceMapper->SetInputData(source);
#endif
    }
    vtkSmartPointer<vtkActor> sourceActor = vtkSmartPointer<vtkActor>::New();
    sourceActor->SetMapper(sourceMapper);
    sourceActor->GetProperty()->SetColor(1,0,0);
    sourceActor->GetProperty()->SetPointSize(4);

    sourceActor->GetProperty()->SetRepresentationToWireframe();

    vtkSmartPointer<vtkPolyDataMapper> targetMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    if ( target->GetNumberOfCells() == 0 )
    {
      std::cerr << "There are no cells in the target data, running delaunay filter\n";
      vtkSmartPointer<vtkDelaunay2D> delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
#if VTK_MAJOR_VERSION <= 5
      delaunay->SetInput(target);
      targetMapper->SetInputConnection(delaunay->GetOutputPort());
      delaunay->Update();
#else
      delaunay->SetInputData(target);
      targetMapper->SetInputData(delaunay);
      delaunay->Update();
#endif
    }
    else
    {
#if VTK_MAJOR_VERSION <= 5
      targetMapper->SetInputConnection(target->GetProducerPort());
#else
      targetMapper->SetInputData(target);
#endif
    }
  
    vtkSmartPointer<vtkActor> targetActor = vtkSmartPointer<vtkActor>::New();
    targetActor->SetMapper(targetMapper);
    targetActor->GetProperty()->SetColor(0,1,0);
    targetActor->GetProperty()->SetPointSize(4);
    targetActor->GetProperty()->SetOpacity(0.0);
   // targetActor->GetProperty()->SetRepresentationToWireframe();
  
    vtkSmartPointer<vtkPolyDataMapper> solutionMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    vtkSmartPointer<vtkPolyData> solution = vtkSmartPointer<vtkPolyData>::New();

    icp->ApplyTransform(solution);
    if ( solution->GetNumberOfCells() == 0 ) 
    {
      vtkSmartPointer<vtkDelaunay2D> delaunay = vtkSmartPointer<vtkDelaunay2D>::New();
#if VTK_MAJOR_VERSION <= 5
      delaunay->SetInput(solution);
      solutionMapper->SetInputConnection(delaunay->GetOutputPort());
      delaunay->Update();
#else
      delaunay->SetInputData(solution);
      solutionMapper->SetInputData(delaunay);
      delaunay->Update();
#endif
    }
    else
    {
#if VTK_MAJOR_VERSION <= 5
      solutionMapper->SetInputConnection(solution->GetProducerPort());
#else
      solutionMapper->SetInputData(solution);
#endif
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

//  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
 // writer->SetInput(filter->GetOutput());
 // writer->SetFileName(args.outputPolyDataFile.c_str());
 // writer->Update();
}

