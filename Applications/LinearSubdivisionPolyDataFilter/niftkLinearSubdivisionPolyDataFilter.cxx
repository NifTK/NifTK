/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
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
struct niftk::CommandLineArgumentDescription clArgList[] = 
{
  {OPT_STRING|OPT_REQ, "i" , "fileName", "Input VTK Poly Data."},
  {OPT_STRING|OPT_REQ, "o", "fileName", "Output VTK Poly Data."},
  {OPT_INT, "number", "int", "[1]  Number of subdivisions"},
  {OPT_DONE, NULL, NULL, 
    "Runs the VTK vtkLinearSubdivision filter on a vtkPolyData.\n"
  }
};

enum
{
  O_INPUT_POLYDATA,

  O_OUTPUT_POLYDATA, 

  O_ITERATIONS
};

/**
 * \brief Runs vtkLinearSubdivisionFilter.
 */
int main(int argc, char** argv)
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;
  int numberOfSubdivisions = 1;
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);
  
  CommandLineOptions.GetArgument(O_INPUT_POLYDATA, inputPolyDataFile);

  CommandLineOptions.GetArgument(O_OUTPUT_POLYDATA, outputPolyDataFile);

  CommandLineOptions.GetArgument(O_ITERATIONS, numberOfSubdivisions);

  vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
  reader->SetFileName(inputPolyDataFile.c_str());
  
  vtkSmartPointer<vtkLinearSubdivisionFilter> filter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
  filter->SetInputConnection(reader->GetOutputPort());
  filter->SetNumberOfSubdivisions(numberOfSubdivisions);

  vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
  writer->SetInputConnection(filter->GetOutputPort());
  writer->SetFileName(outputPolyDataFile.c_str());
  writer->SetFileTypeToASCII();
  writer->Update();
}
