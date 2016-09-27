/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkCaffeSegCLP.h>
#include <mitkException.h>
#include <mitkVector.h>
#include <mitkDataNode.h>

#include <cstdlib>
#include <limits>
#include <stdexcept>

int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  try
  {

    if (model.empty() || weights.empty())
    {
      commandLine.getOutput()->usage(commandLine);
      return returnStatus;
    }

    if (!inputImage.empty() && !inputDir.empty())
    {
      MITK_ERROR << "You should not specify both --inputImage and --inputDir. Its one or the other." << std::endl;
      return returnStatus;
    }

    // Load input image, and create node.
    mitk::DataNode::ConstPointer inputImage;
    mitk::DataNode::Pointer outputImage = mitk::DataNode::New();

//    niftk::CaffeFCNSegmentor::Pointer manager = niftk::CaffeManager::New(model, weights);

    // ToDo: Create an mitk::DataStorage like we do in unit tests 
    // manager->Segment(dataStorage, inputImage);

    // ToDo: Write output image
    //
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
