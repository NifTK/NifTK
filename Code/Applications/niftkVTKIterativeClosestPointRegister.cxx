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
    std::cout << "    --visualise                   Visualise registration." << std::endl;
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

/**
 * \brief Transform's VTK poly data file by any number of affine transformations.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  

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
    else if(strcmp(argv[i], "--visualise") == 0){
      args.visualise = true;
      std::cout << "Set Visualise on" << std::endl;
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

  std::cout << "Loading PolyData:" << args.sourcePolyDataFile << std::endl;
  
  vtkPolyDataReader *reader = vtkPolyDataReader::New();
  reader->SetFileName(args.sourcePolyDataFile.c_str());
  reader->Update();
  
  std::cout << "Loaded PolyData:" << args.sourcePolyDataFile << std::endl;
  
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
 // writer->SetInput(filter->GetOutput());
 // writer->SetFileName(args.outputPolyDataFile.c_str());
 // writer->Update();
}

