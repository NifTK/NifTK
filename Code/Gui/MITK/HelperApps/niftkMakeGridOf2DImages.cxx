/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkCommandLineHelper.h>
#include <mitkMakeGridOf2DImages.h>
#include <niftkMakeGridOf2DImagesCLP.h>

/*!
 * \file niftkMakeGridOf2DImages.cxx
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  // Validate command line args
  if (outputImage.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::MakeGridOf2DImages::Pointer gridMaker = mitk::MakeGridOf2DImages::New();
    gridMaker->MakeGrid(directoryName, imageSize, gridDimensions, outputImage);
    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 2;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 3;
  }

  return returnStatus;
}
