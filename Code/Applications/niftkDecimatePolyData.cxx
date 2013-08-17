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
#include <vtkDecimatePro.h>

/*!
 * \file niftkDecimatePolyData.cxx
 * \page niftkDecimatePolyData
 * \section niftkDecimatePolyDataSummary Runs the VTK vtkDecimatePro filter on a vtkPolyData.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the VTK vtkDecimatePro filter on a vtkPolyData." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i      <filename>        Input VTK Poly Data. eg. 0.9 means 90% reduction, 0.1 means 10% reduction" << std::endl;
    std::cout << "    -o      <filename>        Output VTK Poly Data." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -target <float>           Target Reduction Factor" << std::endl;
    std::cout << "    -maxErr <float>           Maximum Error" << std::endl;
    std::cout << "    -feat   <float>           Feature Angle" << std::endl;
    std::cout << "    -preserve                 Preserve topology. Default off, as it may prevent you reaching the target reduction factor." << std::endl << std::endl;
  }

struct arguments
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;
  double targetReductionFactor;
  double maximumError;
  double featureAngle;
  bool preserveTopology;
  bool maximumErrorSpecified;
  bool featureAngleSpecified;
  bool targetReductionFactorSpecified;
};

/**
 * \brief Runs vtkDecimatePro filter.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.preserveTopology = false;
  args.maximumErrorSpecified = false;
  args.featureAngleSpecified = false;
  args.targetReductionFactorSpecified = false;
  

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
    else if(strcmp(argv[i], "-target") == 0){
      args.targetReductionFactor=atof(argv[++i]);
      args.targetReductionFactorSpecified = true;
      std::cout << "Added -target=" << niftk::ConvertToString(args.targetReductionFactor) << std::endl;
    }
    else if(strcmp(argv[i], "-maxErr") == 0){
      args.maximumError=atof(argv[++i]);
      args.maximumErrorSpecified = true;
      std::cout << "Added -maxErr=" << niftk::ConvertToString(args.maximumError) << std::endl;
    }
    else if(strcmp(argv[i], "-feat") == 0){
      args.featureAngle=atof(argv[++i]);
      args.featureAngleSpecified = true;
      std::cout << "Added -feat=" << niftk::ConvertToString(args.featureAngle) << std::endl;
    }
    else if(strcmp(argv[i], "-preserve") == 0){
      args.preserveTopology = true;
      std::cout << "Set -preserve=" << niftk::ConvertToString(args.preserveTopology) << std::endl;
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
  
  vtkDecimatePro *filter = vtkDecimatePro::New();
  filter->SetInput(reader->GetOutput());
  filter->SetPreserveTopology(args.preserveTopology);
  
  if (args.featureAngleSpecified) filter->SetFeatureAngle(args.featureAngle);
  if (args.maximumErrorSpecified) filter->SetMaximumError(args.maximumError);
  if (args.targetReductionFactorSpecified) filter->SetTargetReduction(args.targetReductionFactor);
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetInput(filter->GetOutput());
  writer->SetFileName(args.outputPolyDataFile.c_str());
  writer->Update();
}

