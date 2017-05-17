/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkLogHelper.h>
#include <niftkConversionUtils.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmoothPolyDataFilter.h>
#include <niftkCommandLineParser.h>

/*!
 * \file niftkSmoothPolyData.cxx
 * \page niftkSmoothPolyData
 * \section niftkSmoothPolyDataSummary Runs the VTK vtkSmoothPolyDataFilter on a vtkPolyData.
 */


struct niftk::CommandLineArgumentDescription clArgList[] = 
{
  {OPT_STRING|OPT_REQ, "i" , "fileName", "Input VTK Poly Data."},
  {OPT_STRING|OPT_REQ, "o", "fileName", "Output VTK Poly Data."},
  {OPT_INT, "iters", "int", "[20]  Number of iterations"},
  {OPT_FLOAT, "threshold", "float" , "[0]  Convergence threshold"},
  {OPT_FLOAT, "edgeAngle", "float", "[15] Angle to determine if something is an edge"},
  {OPT_FLOAT, "featureAngle", "float", "[45]  Angle that determines whether something is truely a feature and shouldn't be smoothed"},
  {OPT_FLOAT, "relax", "float", "[0.15] Relaxation factor"},
  {OPT_SWITCH, "smoothFeatures", NULL, "Turn on feature smoothing"},
  {OPT_SWITCH, "noEdgeSmoothing", NULL, "Turn off edge smoothing"},
  {OPT_SWITCH, "generateErrorScalars", NULL, "Generates error scalars"},
  {OPT_SWITCH, "generateErrorVectors", NULL, "Generates error vectors"},
  {OPT_DONE, NULL, NULL, 
    "Runs the VTK vtkSmoothPolyDataFilter on a vtkPolyData.\n"
  }
};


enum {
  O_INPUT_POLYDATA,

  O_OUTPUT_POLYDATA, 

  O_ITERATIONS, 

  O_THRESHOLD, 

  O_EDGEANGLE, 

  O_FEATUREANGLE,

  O_RELAX, 
  
  O_SMOOTHFEATURES, 

  O_NOEDGESMOOTHING, 

  O_GENERATE_ERROR_SCALARS,

  O_GENERATE_ERROR_VECTORS
};


/**
 * \brief Runs vtkDecimatePro filter.
 */
int main(int argc, char** argv)
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;
  int numberIterations = 20;
  float convergenceThreshold = 0;
  float edgeAngle = 15;
  float featureAngle = 45;
  float relaxationFactor = 0.15;
  bool smoothFeatures = false;
  bool smoothEdges = true;
  bool generateErrorScalars = false;
  bool generateErrorVectors = false;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);
  
  CommandLineOptions.GetArgument(O_INPUT_POLYDATA, inputPolyDataFile);

  CommandLineOptions.GetArgument(O_OUTPUT_POLYDATA, outputPolyDataFile);

  CommandLineOptions.GetArgument(O_ITERATIONS, numberIterations);

  CommandLineOptions.GetArgument(O_THRESHOLD, convergenceThreshold);

  CommandLineOptions.GetArgument(O_EDGEANGLE, edgeAngle);

  CommandLineOptions.GetArgument(O_FEATUREANGLE, featureAngle);

  CommandLineOptions.GetArgument(O_RELAX, relaxationFactor);
  
  CommandLineOptions.GetArgument(O_SMOOTHFEATURES, smoothFeatures); 

  CommandLineOptions.GetArgument(O_NOEDGESMOOTHING, smoothEdges);

  CommandLineOptions.GetArgument(O_GENERATE_ERROR_SCALARS, generateErrorScalars);

  CommandLineOptions.GetArgument(O_GENERATE_ERROR_VECTORS, generateErrorVectors);

  vtkPolyDataReader *reader = vtkPolyDataReader::New();
  reader->SetFileName(inputPolyDataFile.c_str());

  vtkSmoothPolyDataFilter *filter = vtkSmoothPolyDataFilter::New();
  filter->SetInputConnection(reader->GetOutputPort());
  filter->SetConvergence(convergenceThreshold);
  filter->SetNumberOfIterations(numberIterations);
  filter->SetRelaxationFactor(relaxationFactor);
  filter->SetFeatureAngle(featureAngle);
  filter->SetEdgeAngle(edgeAngle);
  filter->SetFeatureEdgeSmoothing(smoothFeatures);
  filter->SetBoundarySmoothing(smoothEdges);
  filter->SetGenerateErrorScalars(generateErrorScalars);
  filter->SetGenerateErrorVectors(generateErrorVectors);
 
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetInputConnection(filter->GetOutputPort());
  writer->SetFileName(outputPolyDataFile.c_str());
  writer->Update();
}
