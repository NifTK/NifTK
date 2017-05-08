/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkCommandLineParser.h>
#include <niftkConversionUtils.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkDecimatePro.h>

/*!
 * \file niftkDecimatePolyData.cxx
 * \page niftkDecimatePolyData
 * \section niftkDecimatePolyDataSummary Runs the VTK vtkDecimatePro filter on a vtkPolyData.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = 
{
  {OPT_STRING|OPT_REQ, "i" , "fileName", "Input VTK Poly Data."},
  {OPT_STRING|OPT_REQ, "o", "fileName", "Output VTK Poly Data."},

  {OPT_FLOAT, "target", "float" , "Target Reduction Factor."},
  {OPT_FLOAT, "maxErr", "float", "Maximum Error."},
  {OPT_FLOAT, "feat", "float", " Feature Angle."},
  {OPT_SWITCH, "preserve", NULL, "Preserve topology. Default off, as it may prevent you reaching the target reduction factor."},

  {OPT_DONE, NULL, NULL, 
    "Runs the VTK vtkDecimatePro filter on a vtkPolyData.\n"
  }
};

enum 
{
  O_INPUT_POLYDATA,

  O_OUTPUT_POLYDATA, 

  O_TARGET_RED, 

  O_MAX_ERROR, 

  O_FEATUREANGLE,

  O_PRESERVE,
};


/**
 * \brief Runs vtkDecimatePro filter.
 */
int main(int argc, char** argv)
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile; 
  
  float featureAngle = 0;
  float targetReductionFactor = 0;
  float maxError = 0;

  bool preserveTopology = false;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);
  
  CommandLineOptions.GetArgument(O_INPUT_POLYDATA, inputPolyDataFile);

  CommandLineOptions.GetArgument(O_OUTPUT_POLYDATA, outputPolyDataFile);

  CommandLineOptions.GetArgument(O_TARGET_RED, targetReductionFactor);

  CommandLineOptions.GetArgument(O_MAX_ERROR, maxError);

  CommandLineOptions.GetArgument(O_FEATUREANGLE, featureAngle);
  
  CommandLineOptions.GetArgument(O_PRESERVE, preserveTopology); 

  vtkPolyDataReader *reader = vtkPolyDataReader::New();
  reader->SetFileName(inputPolyDataFile.c_str());
  
  vtkDecimatePro *filter = vtkDecimatePro::New();
  filter->SetInputConnection(reader->GetOutputPort());
  filter->SetPreserveTopology(preserveTopology);
  
  if (featureAngle > 0)
  {
    filter->SetFeatureAngle(featureAngle);
  }

  if (maxError > 0)
  {
    filter->SetMaximumError(maxError);
  }

  if (targetReductionFactor)
  {
    filter->SetTargetReduction(targetReductionFactor);
  }

  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetInputConnection(filter->GetOutputPort());
  writer->SetFileName(outputPolyDataFile.c_str());
  writer->Update();
}
