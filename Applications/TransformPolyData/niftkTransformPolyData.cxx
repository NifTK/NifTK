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
#include <niftkCommandLineParser.h>

#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <itkEulerAffineTransform.h>
#include <itkMatrix.h>
#include <vtkIndent.h>
#include <vtkTransformPolyDataFilter.h>

/*!
 * \file niftkTransformPolyData.cxx
 * \page niftkTransformPolyData
 * \section niftkTransformPolyDataSummary Transform's a VTK Poly Data file by any number of affine transformations.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_STRING|OPT_REQ, "i", "filename", " Input VTK Poly Data."},
  {OPT_STRING|OPT_REQ, "o", "filename", " Output VTK Poly Data."},
  {OPT_STRING, "t", "filename", "Affine transform file. Text file, 4 rows of 4 numbers."
    "This option can be repeated."},
  {OPT_MORE, NULL, "...", NULL},
   
  {OPT_DONE, NULL, NULL, 
   "Transforms a VTK Poly Data file by any number of affine transformations.\n"
  }
};

enum { 
  O_FILE_INPUT, 
  
  O_FILE_OUTPUT,

  O_FILE_TRANSFORM,
  
  O_MORE
};

/**
 * \brief Transform's VTK poly data file by any number of affine transformations.
 */
int main(int argc, char** argv)
{
  std::string inputPolyDataFile;
  std::string outputPolyDataFile;

  std::string fileTransform;
  std::vector<std::string> transformNames;
  niftk::ml fileTransforms;

  // To pass around command line args
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);

  CommandLineOptions.GetArgument(O_FILE_INPUT, inputPolyDataFile);

  CommandLineOptions.GetArgument(O_FILE_OUTPUT, outputPolyDataFile);

  CommandLineOptions.GetArgument(O_FILE_TRANSFORM, fileTransform);
  transformNames.push_back(fileTransform);

  int arg;
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) 
  {
    int nTransforms = argc - arg;
    char** files = &argv[arg];

    for (unsigned int i = 0; i < nTransforms; i++)
    {
      if (files[i] != "-t")
      {
        transformNames.push_back(files[i]);
      }
    }
  }

  typedef itk::EulerAffineTransform<double, 3, 3> AffineTransformType;
  typedef itk::Matrix<double, 4, 4> AffineMatrixType; 
  
  AffineMatrixType itkMatrix;
  AffineTransformType::Pointer itkTransform = AffineTransformType::New();
  
  vtkMatrix4x4 *vtkMatrix = vtkMatrix4x4::New();
  vtkTransform *vtkTransform = vtkTransform::New();
  vtkTransform->Identity();
  vtkTransform->PostMultiply();
  
  // Read transformations.
  unsigned int i, j, k;
  
  for (i = 0; i < transformNames.size(); i++)
    {
      itkTransform->LoadFullAffineMatrix(transformNames[i]);
      itkMatrix = itkTransform->GetFullAffineMatrix();
      
      for (j = 0; j < 4; j++)
        {
          for (k = 0; k < 4; k++)
            {
              vtkMatrix->SetElement(j,k,itkMatrix(j,k));
            }
        }
      vtkTransform->Concatenate(vtkMatrix);

      std::cout << "Loading Transform:" << transformNames[i] << std::endl;
  }
  
  vtkPolyDataReader *reader = vtkPolyDataReader::New();
  reader->SetFileName(inputPolyDataFile.c_str());
    
  vtkTransformPolyDataFilter *filter = vtkTransformPolyDataFilter::New();
  filter->SetInputConnection(reader->GetOutputPort());
  filter->SetTransform(vtkTransform);

  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetInputConnection(filter->GetOutputPort());
  writer->SetFileName(outputPolyDataFile.c_str());
  writer->Update();
}

