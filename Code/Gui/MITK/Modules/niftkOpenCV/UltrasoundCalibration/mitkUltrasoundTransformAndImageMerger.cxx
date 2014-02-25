/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundTransformAndImageMerger.h"
#include <mitkExceptionMacro.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkFileIOUtils.h>
#include <mitkCameraCalibrationFacade.h>
#include <niftkFileHelper.h>
#include <niftkVTKFunctions.h>
#include <mitkIOUtil.h>
#include <itkImage.h>
#include <itkImageFileWriter.h>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundTransformAndImageMerger::~UltrasoundTransformAndImageMerger()
{
}


//-----------------------------------------------------------------------------
UltrasoundTransformAndImageMerger::UltrasoundTransformAndImageMerger()
{
}


//-----------------------------------------------------------------------------
void UltrasoundTransformAndImageMerger::Merge(
    const std::string& inputMatrixDirectory,
    const std::string& inputImageDirectory,
    const std::string& outputFileName
    )
{
  std::vector< cv::Mat > matrices;

  std::vector<std::string> matrixFiles = niftk::GetFilesInDirectory(inputMatrixDirectory);
  std::sort(matrixFiles.begin(), matrixFiles.end());

  std::vector<std::string> imageFiles = niftk::GetFilesInDirectory(inputImageDirectory);
  std::sort(imageFiles.begin(), imageFiles.end());

  if (matrixFiles.size() != imageFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "The matrix directory:" << std::endl << "  " << inputMatrixDirectory << std::endl << "and the image directory:" << std::endl << "  " << inputImageDirectory << "contain a different number of files!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  matrices = LoadMatricesFromDirectory (inputMatrixDirectory);

  // Load all images.

  std::vector<mitk::Image::Pointer> images;
  for (int i = 0; i < imageFiles.size(); i++)
  {
    images.push_back(mitk::IOUtil::LoadImage(imageFiles[i]));
  }

  if (matrices.size() != images.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "Loaded " << matrices.size() << " matrices, and loaded a difference number of images " << images.size() << std::endl;
    mitkThrow() << errorMessage.str();
  }

  typedef itk::Image<unsigned char, 3> ImageType;
  ImageType::Pointer outputImage = ImageType::New();

  int sizeX = images[0]->GetDimension(0);
  int sizeY = images[0]->GetDimension(1);

  ImageType::SizeType size;
  size[0] = sizeX;
  size[1] = sizeY;
  size[2] = images.size();

  ImageType::IndexType offset;
  offset.Fill(0);

  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(offset);

  ImageType::SpacingType spacing;
  spacing.Fill(1);

  ImageType::PointType origin;
  origin.Fill(0);

  ImageType::DirectionType direction;
  direction.SetIdentity();

  outputImage->SetSpacing(spacing);
  outputImage->SetOrigin(origin);
  outputImage->SetRegions(region);
  outputImage->SetDirection(direction);
  outputImage->Allocate();
  outputImage->FillBuffer(0);

  // Fill 3D image, slice by slice. This is slow, but we wont do it often.
  mitk::Index3D inputImageIndex;
  ImageType::IndexType outputImageIndex;

  for (unsigned int i = 0; i < images.size(); i++)
  {
    for (unsigned int y = 0; y < sizeY; y++)
    {
      for (unsigned int x = 0; x < sizeX; x++)
      {
        inputImageIndex[0] = x;
        inputImageIndex[1] = y;
        inputImageIndex[2] = 0;

        outputImageIndex[0] = x;
        outputImageIndex[1] = y;
        outputImageIndex[2] = i;

        mitk::Image::Pointer tmpImage = images[i];
        outputImage->SetPixel(outputImageIndex, tmpImage->GetPixelValueByIndex(inputImageIndex));
      }
    }
  }

  // Now we write a volume. We just output the extra header info required.
  // The user can manually append it to the file if necessary.
  itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
  writer->SetFileName(outputFileName);
  writer->SetInput(outputImage);
  writer->Update();

  std::cout << "Written to " << outputFileName << std::endl;
  std::cout << "Extra header info required for .mhd:" << std::endl;

  std::cout << "UltrasoundImageOrientation = MATT FIXME" << std::endl;
  std::cout << "UltrasoundImageType = BRIGHTNESS" << std::endl;

  std::string oneZero = "0";
  std::string twoZero = "00";
  std::string threeZero = "000";

  for (unsigned int i = 0; i < images.size(); i++)
  {
    std::ostringstream suffix;
    if (i < 10)
    {
      suffix << threeZero << i;
    }
    else if (i < 100)
    {
      suffix << twoZero << i;
    }
    else if (i < 1000)
    {
      suffix << oneZero << i;
    }
    else
    {
      suffix << i;
    }
    std::cout << "Seq_Frame" << suffix.str() << "_FrameNumber = " << i << std::endl;
    std::cout << "Seq_Frame" << suffix.str() << "_UnfilteredTimestamp = " << i << std::endl;
    std::cout << "Seq_Frame" << suffix.str() << "_Timestamp = " << i << std::endl;
    std::cout << "Seq_Frame" << suffix.str() << "_ToolToTrackerTransformTransformStatus = OK" << std::endl;
    std::cout << "Seq_Frame" << suffix.str() << "_ToolToTrackerTransform =";
    for (int r = 0; r < 4; r++)
    {
      for (int c = 0; c < 4; c++)
      {
        std::cout << " " << matrices[i].at<double>(r, c);
      }
    }
    std::cout << std::endl;
  }
}


//-----------------------------------------------------------------------------
} // end namespace
