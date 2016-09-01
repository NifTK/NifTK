/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkLogHelper.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkLinearSubdivisionFilter.h>

/*!
 * \file niftkLinearSubdivisionPolyDataFilter.cxx
 * \page niftkLinearSubdivisionPolyDataFilter
 * \section niftkLinearSubdivisionPolyDataFilterSummary Runs the VTK vtkLinearSubdivisionFilter on a vtkPolyData.
 */
void Usage(char *exec)
  {
    niftk::LogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Runs the VTK vtkLinearSubdivision filter on a vtkPolyData." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputPolyData.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i      <filename>        Input VTK Poly Data." << std::endl;
    std::cout << "    -o      <filename>        Output VTK Poly Data." << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -number <int>             Number of subdivisions." << std::endl;

  }

struct arguments
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;
  int numberOfSubdivisions;
};

/**
 * \brief Runs vtkLinearSubdivisionFilter.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.numberOfSubdivisions = 1;

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
    else if(strcmp(argv[i], "-number") == 0){
      args.numberOfSubdivisions=atoi(argv[++i]);
      std::cout << "Added -number=" << niftk::ConvertToString(args.numberOfSubdivisions) << std::endl;
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

  vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
  reader->SetFileName(args.inputPolyDataFile.c_str());
  
  vtkSmartPointer<vtkLinearSubdivisionFilter> filter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
  filter->SetInputConnection(reader->GetOutputPort());
  filter->SetNumberOfSubdivisions(args.numberOfSubdivisions);

  vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
  writer->SetInputConnection(filter->GetOutputPort());
  writer->SetFileName(args.outputPolyDataFile.c_str());
  writer->SetFileTypeToASCII();
  writer->Update();
}

