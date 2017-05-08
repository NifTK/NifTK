/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkLogHelper.h>
#include <niftkCommandLineParser.h>
#include <niftkConversionUtils.h>
#include <itkEulerAffineTransform.h>

/*!
 * \file niftkInvertAffineTransform.cxx
 * \page niftkInvertAffineTransform
 * \section niftkInvertAffineTransformSummary Inverts an affine transform.
 */

struct niftk::CommandLineArgumentDescription clArgList[] = 
{
  {OPT_STRING|OPT_REQ, "i", "string", " Input matrix, a plain text file, 4 rows, 4 columns."},
  {OPT_STRING|OPT_REQ, "o", "string", "Output matrix, in same format as input."},

  {OPT_DONE, NULL, NULL, 
   "Program to invert an affine transformation.\n"
  }
};

enum
{
  O_INPUT_TRANSFORM, 

  O_OUTPUT_TRANSFORM
};


/**
 * \brief Calculates inverse of an affine transform.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  std::string inputTransform;
  std::string outputTransform;    

  // Parse command line acd rgs
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);

  CommandLineOptions.GetArgument( O_INPUT_TRANSFORM, inputTransform );

  CommandLineOptions.GetArgument( O_OUTPUT_TRANSFORM, outputTransform );

  // Short and sweet.... well... not as short and sweet as Matlab!
  
  typedef itk::EulerAffineTransform<double, 3, 3> MatrixType;
  
  MatrixType::Pointer inputMatrix = MatrixType::New();
  inputMatrix->LoadFullAffineMatrix(inputTransform);
  
  MatrixType::Pointer outputMatrix = MatrixType::New();
  inputMatrix->GetInverse(outputMatrix);
  outputMatrix->SaveFullAffineMatrix(outputTransform);
}
