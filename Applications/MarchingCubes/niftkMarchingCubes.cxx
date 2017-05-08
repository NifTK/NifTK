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
#include <vtkDataObject.h>
#include <vtkStructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkStructuredGridReader.h>
#include <vtkStructuredPointsReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkContourFilter.h>
#include <vtkMarchingCubes.h>
#include <NifTKConfigure.h>
#include <vtkPolyDataNormals.h>
#include <vtkSmartPointer.h>


/*!
 * \file niftkMarchingCubes.cxx
 * \page niftkMarchingCubes
 * \section niftkMarchingCubesSummary Takes an image as a VTK structured grid (or structured points), and performs a marching cubes iso-surface contour extraction.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = 
{
  {OPT_STRING|OPT_REQ, "i" , "fileName", "Input image as VTK structured grid."},
  {OPT_STRING|OPT_REQ, "o", "fileName", "Output VTK Poly Data."},
  {OPT_FLOAT, "iso", "float" , "[128]  Threshold value to extract."},
  {OPT_SWITCH, "points", NULL, "Input is structured points."},
  {OPT_SWITCH, "withScalars", NULL, "Computes scalar values for each vertex."},
  {OPT_SWITCH, "withNormals", NULL, "Computes normals at each vertex."},
  {OPT_SWITCH, "withGradient", NULL, "Computes gradient at each vertex."},
  {OPT_DONE, NULL, NULL, 
    "Takes an image as a VTK structured grid (NOT structured points),"
    "and performs a marching cubes iso-surface extraction.\n"
  }
};

enum 
{
  O_INPUT_IMAGE,

  O_OUTPUT_POLYDATA, 

  O_THRESHOLD, 

  O_INPUT_POINTS, 

  O_COMPUTE_SCALARS,

  O_COMPUTE_NORMALS,

  O_COMPUTE_GRADIENT
};
/**
 * \brief Runs marching cubes.
 */
int main(int argc, char** argv)
{
  std::string inputImage;
  std::string outputPolyData;

  float isoSurfaceValue = 128;

  bool isStructuredPoints;
  bool withScalars;
  bool withGradients;
  bool withNormals;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);
  
  CommandLineOptions.GetArgument(O_INPUT_IMAGE, inputImage);

  CommandLineOptions.GetArgument(O_OUTPUT_POLYDATA, outputPolyData);

  CommandLineOptions.GetArgument(O_THRESHOLD, isoSurfaceValue);

  CommandLineOptions.GetArgument(O_INPUT_POINTS, isStructuredPoints);
  
  CommandLineOptions.GetArgument(O_COMPUTE_SCALARS, withScalars);
  
  CommandLineOptions.GetArgument(O_COMPUTE_NORMALS, withGradients);

  CommandLineOptions.GetArgument(O_COMPUTE_GRADIENT, withNormals);
  
  // Read structured points
  // ~~~~~~~~~~~~~~~~~~~~~~
  if (isStructuredPoints)
  {

    vtkSmartPointer<vtkStructuredPointsReader> 
      reader = vtkSmartPointer<vtkStructuredPointsReader>::New();

    reader->SetFileName(inputImage.c_str());

    reader->Update();
     
  
    // Create the marching cubes filter

    vtkSmartPointer<vtkMarchingCubes> 
      filter = vtkSmartPointer<vtkMarchingCubes>::New();

    filter->SetInputConnection( reader->GetOutputPort() );

    filter->SetValue( 0, isoSurfaceValue );

    filter->SetComputeScalars( withScalars );
    filter->SetComputeGradients( withGradients );
    filter->SetComputeNormals( withNormals );
  
    filter->Update();


    // Write the ouput

    vtkSmartPointer<vtkPolyDataWriter> 
      writer = vtkSmartPointer<vtkPolyDataWriter>::New();

    writer->SetInputDataObject( filter->GetOutput() );
    writer->SetFileName( outputPolyData.c_str() );

    writer->Update();
  }


  // Read structured grid
  // ~~~~~~~~~~~~~~~~~~~~
  
  else
  {

    vtkSmartPointer<vtkStructuredGridReader>
      reader = vtkSmartPointer<vtkStructuredGridReader>::New();
    reader->SetFileName(inputImage.c_str());

    reader->Update();
     

    // Create the marching cubes filter

    vtkSmartPointer<vtkContourFilter>
      filter = vtkSmartPointer<vtkContourFilter>::New();

    filter->SetInputDataObject( reader->GetOutput() );

    filter->SetValue( 0, isoSurfaceValue );

    filter->SetComputeScalars( withScalars );
    filter->SetComputeGradients( withGradients );
    filter->SetComputeNormals( withNormals );
  
    filter->Update();


    // Compute normals

    vtkSmartPointer<vtkPolyDataNormals>
      normals = vtkSmartPointer<vtkPolyDataNormals>::New();

    normals->SetInputDataObject( filter->GetOutput() );
    normals->SetFeatureAngle(60.0);

    normals->Update();


    // Write the ouput

    vtkSmartPointer<vtkPolyDataWriter>
      writer = vtkSmartPointer<vtkPolyDataWriter>::New();

    writer->SetInputDataObject( normals->GetOutput() );
    writer->SetFileName( outputPolyData.c_str() );

    writer->Update();
  }
}
