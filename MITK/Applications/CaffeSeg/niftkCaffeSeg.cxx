/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkCaffeSegCLP.h>
#include <niftkCaffeFCNSegmentor.h>
#include <mitkException.h>
#include <mitkVector.h>
#include <mitkDataNode.h>
#include <mitkBaseData.h>
#include <mitkImage.h>
#include <mitkIOUtil.h>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <caffe/caffe.hpp>

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

    if (!outputImage.empty() && !inputDir.empty())
    {
      MITK_ERROR << "You should not specify both --outputImage and --inputDir. Its one or the other." << std::endl;
      return returnStatus;
    }

    std::vector<mitk::BaseData::Pointer> images = mitk::IOUtil::Load(inputImage);
    mitk::Image::Pointer ipImage = dynamic_cast<mitk::Image*>(images[0].GetPointer());

    mitk::Image::Pointer opImage = mitk::Image::New();
    mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();
    unsigned int dim[] = { ipImage->GetDimension(0), ipImage->GetDimension(1)};
    opImage->Initialize( pt, 2, dim);

    int dummyArgc = 1;
    caffe::GlobalInit(&dummyArgc, &argv);

    niftk::CaffeFCNSegmentor::Pointer manager
      = niftk::CaffeFCNSegmentor::New(model, weights, inputLayer, outputBlob);
    manager->SetOffset(offset);
    manager->Segment(ipImage, opImage);

    mitk::IOUtil::Save(opImage, outputImage);

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
