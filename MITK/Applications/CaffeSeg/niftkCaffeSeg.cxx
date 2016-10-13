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
#include <niftkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <mitkException.h>
#include <mitkVector.h>
#include <mitkDataNode.h>
#include <mitkBaseData.h>
#include <mitkImage.h>
#include <mitkIOUtil.h>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#define GLOG_NO_ABBREVIATED_SEVERITIES
#pragma push_macro("STRICT")
#undef STRICT
#include <caffe/caffe.hpp>
#pragma pop_macro("STRICT")
#else
#include <caffe/caffe.hpp>
#endif

int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  try
  {
    if (model.empty() || weights.empty())
    {
      commandLine.getOutput()->usage(commandLine);
      return returnStatus + 1;
    }

    if (inputImage.empty() && inputDir.empty())
    {
      MITK_ERROR << "You should specify either --inputImage or --inputDir.";
      return returnStatus + 2;
    }

    if (!inputImage.empty() && !inputDir.empty())
    {
      MITK_ERROR << "You should not specify both --inputImage and --inputDir. Its one or the other.";
      return returnStatus + 3;
    }

    if (!outputImage.empty() && !inputDir.empty())
    {
      MITK_ERROR << "You should not specify both --outputImage and --inputDir. Its one or the other.";
      return returnStatus + 4;
    }

    if (outputImage.empty() && !inputDir.empty())
    {
      MITK_ERROR << "If you specify --inputImage you must also specify --outputImage.";
      return returnStatus + 5;
    }

    int dummyArgc = 1; // Caffe doesn't like arguments with - in.
    caffe::GlobalInit(&dummyArgc, &argv);

    niftk::CaffeFCNSegmentor::Pointer manager
      = niftk::CaffeFCNSegmentor::New(model, weights, inputLayer, outputBlob, gpu /* only works if compiled in */);
    manager->SetTransposingMode(transpose);

    MITK_INFO << "Transposing mode:" << manager->GetTransposingMode();

    std::vector<std::string> filesToProcess;

    if (!inputDir.empty())
    {
      filesToProcess = niftk::GetFilesInDirectory(inputDir);
      if (filesToProcess.size() == 0)
      {
        std::ostringstream errorMessage;
        errorMessage << "No files in directory:" << inputDir << std::endl;
        mitkThrow() << errorMessage.str();
      }
    }
    else if (!inputImage.empty())
    {
      filesToProcess.push_back(inputImage);
    }

    mitk::Image::Pointer opImage = mitk::Image::New();
    mitk::PixelType pt = mitk::MakeScalarPixelType<unsigned char>();

    for (int i = 0; i < filesToProcess.size(); i++)
    {
      std::vector<mitk::BaseData::Pointer> images = mitk::IOUtil::Load(filesToProcess[i]);
      if (images.size() > 1)
      {
        mitkThrow() << "Loading " << filesToProcess[i] << ", resulted in > 1 image???";
      }
      if (images.size() == 0)
      {
        mitkThrow() << "Failed to load:" << filesToProcess[i];
      }

      mitk::Image::Pointer ipImage = dynamic_cast<mitk::Image*>(images[0].GetPointer());

      if (opImage->GetDimension(0) != ipImage->GetDimension(0)
          || opImage->GetDimension(1) != ipImage->GetDimension(1)
          )
      {
        unsigned int dim[] = { ipImage->GetDimension(0), ipImage->GetDimension(1)};
        opImage->Initialize( pt, 2, dim);
      }

      manager->Segment(ipImage, opImage);

      if (outputImage.empty())
      {
        mitk::IOUtil::Save(opImage, filesToProcess[i] + "_Mask.png");
      }
      else
      {
        mitk::IOUtil::Save(opImage, outputImage);
      }
    }

    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 100;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 101;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 102;
  }
  return returnStatus;
}
