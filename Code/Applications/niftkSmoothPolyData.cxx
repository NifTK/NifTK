/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmoothPolyDataFilter.h>

/*!
 * \file niftkSmoothPolyData.cxx
 * \page niftkSmoothPolyData
 * \section niftkSmoothPolyDataSummary Runs the VTK vtkSmoothPolyDataFilter on a vtkPolyData.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the VTK vtkSmoothPolyDataFilter on a vtkPolyData." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i            <filename>     Input VTK Poly Data." << std::endl;
    std::cout << "    -o            <filename>     Output VTK Poly Data." << std::endl << std::endl;     
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -iters        <int>    [20]  Number of iterations" << std::endl;
    std::cout << "    -threshold    <float>   [0]  Convergence threshold" << std::endl;
    std::cout << "    -relax        <float> [0.15] Relaxation factor" << std::endl;
    std::cout << "    -featureAngle <float>  [45]  Angle that determines whether something is truely a feature and shouldn't be smoothed" << std::endl;
    std::cout << "    -edgeAngle    <float>  [15]  Angle to determine if something is an edge" << std::endl;
    std::cout << "    -smoothFeatures              Turn on feature smoothing" << std::endl;
    std::cout << "    -noEdgeSmoothing             Turn off edge smoothing" << std::endl;
    std::cout << "    -generateErrorScalars        Generates error scalars" << std::endl;
    std::cout << "    -generateErrorVectors        Generates error vectors" << std::endl;
  }

struct arguments
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;
  int numberIterations;
  float convergenceThreshold;
  float edgeAngle;
  float featureAngle;
  float relaxationFactor;
  bool smoothFeatures;
  bool smoothEdges;
  bool generateErrorScalars;
  bool generateErrorVectors;
};

/**
 * \brief Runs vtkDecimatePro filter.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.numberIterations = 20;
  args.convergenceThreshold = 0;
  args.edgeAngle = 15;
  args.featureAngle = 45;
  args.smoothFeatures = false;
  args.smoothEdges = true;
  args.generateErrorScalars = false;
  args.generateErrorVectors = false;
  args.relaxationFactor = 0.15;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputPolyDataFile=argv[++i];
      std::cout << "Set -i=" << args.inputPolyDataFile << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputPolyDataFile=argv[++i];
      std::cout << "Set -o=" << args.outputPolyDataFile << std::endl;
    }
    else if(strcmp(argv[i], "-iters") == 0){
      args.numberIterations=atoi(argv[++i]);
      std::cout << "Set -iters=" << niftk::ConvertToString(args.numberIterations) << std::endl;
    }
    else if(strcmp(argv[i], "-threshold") == 0){
      args.convergenceThreshold=atof(argv[++i]);
      std::cout << "Set -threshold=" << niftk::ConvertToString(args.convergenceThreshold) << std::endl;
    }
    else if(strcmp(argv[i], "-featureAngle") == 0){
      args.featureAngle=atof(argv[++i]);
      std::cout << "Set -featureAngle=" << niftk::ConvertToString(args.featureAngle) << std::endl;
    }
    else if(strcmp(argv[i], "-edgeAngle") == 0){
      args.edgeAngle=atof(argv[++i]);
      std::cout << "Set -edgeAngle=" << niftk::ConvertToString(args.edgeAngle) << std::endl;
    }
    else if(strcmp(argv[i], "-relax") == 0){
      args.relaxationFactor=atof(argv[++i]);
      std::cout << "Set -relax=" << niftk::ConvertToString(args.relaxationFactor) << std::endl;
    }
    else if(strcmp(argv[i], "-smoothFeatures") == 0){
      args.smoothFeatures = true;
      std::cout << "Set -smoothFeatures=" << niftk::ConvertToString(args.smoothFeatures) << std::endl;
    }
    else if(strcmp(argv[i], "-noEdgeSmoothing") == 0){
      args.smoothEdges = false;
      std::cout << "Set -noEdgeSmoothing=" << niftk::ConvertToString(args.smoothEdges) << std::endl;
    }
    else if(strcmp(argv[i], "-generateErrorScalars") == 0){
      args.generateErrorScalars = false;
      std::cout << "Set -generateErrorScalars=" << niftk::ConvertToString(args.generateErrorScalars) << std::endl;
    }
    else if(strcmp(argv[i], "-generateErrorVectors") == 0){
      args.generateErrorVectors = false;
      std::cout << "Set -generateErrorVectors=" << niftk::ConvertToString(args.generateErrorVectors) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }           
  }
 
  // Validate command line args
  if (args.outputPolyDataFile.length() == 0 || args.inputPolyDataFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *reader = vtkPolyDataReader::New();
  reader->SetFileName(args.inputPolyDataFile.c_str());
 
  vtkSmoothPolyDataFilter *filter = vtkSmoothPolyDataFilter::New();
  filter->SetInput(reader->GetOutput());
  filter->SetConvergence(args.convergenceThreshold);
  filter->SetNumberOfIterations(args.numberIterations);
  filter->SetRelaxationFactor(args.relaxationFactor);
  filter->SetFeatureAngle(args.featureAngle);
  filter->SetEdgeAngle(args.edgeAngle);
  filter->SetFeatureEdgeSmoothing(args.smoothFeatures);
  filter->SetBoundarySmoothing(args.smoothEdges);
  filter->SetGenerateErrorScalars(args.generateErrorScalars);
  filter->SetGenerateErrorVectors(args.generateErrorVectors);
 
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetInput(filter->GetOutput());
  writer->SetFileName(args.outputPolyDataFile.c_str());
  writer->Update();
}
