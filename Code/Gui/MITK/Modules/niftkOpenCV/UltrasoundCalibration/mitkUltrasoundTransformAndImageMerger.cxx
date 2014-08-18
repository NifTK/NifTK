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
#include <mitkImagePixelReadAccessor.h>
#include <mitkTrackingMatrixTimeStamps.h>
#include <niftkFileHelper.h>
#include <niftkVTKFunctions.h>
#include <mitkIOUtil.h>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

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
    const std::string& outputImageFileName,
    const std::string& outputDataFileName
    )
{

  std::ofstream fout(outputDataFileName.c_str());
  if ( !fout )
  {
    std::ostringstream errorMessage;
    errorMessage << "Could not open " << outputDataFileName << " for output!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  cv::Matx44d identityMatrix;
  mitk::MakeIdentity(identityMatrix);

  std::vector< cv::Mat > matrices;

  std::vector<std::string> matrixFiles = niftk::GetFilesInDirectory(inputMatrixDirectory);
  std::sort(matrixFiles.begin(), matrixFiles.end());

  mitk::TrackingMatrixTimeStamps trackingTimeStamps = FindTrackingTimeStamps(inputMatrixDirectory);

  std::vector<std::string> imageFiles = niftk::GetFilesInDirectory(inputImageDirectory);
  std::sort(imageFiles.begin(), imageFiles.end());

  matrices = LoadMatricesFromDirectory (inputMatrixDirectory);

  // Load all images.
  std::vector<mitk::Image::Pointer> images;
  for (int i = 0; i < imageFiles.size(); i++)
  {
    images.push_back(mitk::IOUtil::LoadImage(imageFiles[i]));
  }

  fout << "Number of matrices=" << matrixFiles.size() << std::endl;
  fout << "Number of images=" << imageFiles.size() << std::endl;

  if (matrixFiles.size() < imageFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "Loaded " << matrices.size() << " matrices, and loaded a difference number of images " << images.size() << ", and number of images must be less than number of matrices." << std::endl;
    mitkThrow() << errorMessage.str();
  }

  if (matrixFiles.size() != matrices.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "Retrieved " << matrixFiles.size() << " file names for matrices, but could only load " << matrices.size() << " matrices!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  if (imageFiles.size() != images.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "Retrieved " << imageFiles.size() << " file names for images, but could only load " << images.size() << " images!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  // Now generate output.

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
  itk::Index<3> inputImageIndex;
  itk::Index<3> outputImageIndex;

  for (unsigned int i = 0; i < images.size(); i++)
  {
    mitk::ImagePixelReadAccessor<unsigned char, 3> readAccess(images[i], images[i]->GetVolumeData(0));

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

        outputImage->SetPixel(outputImageIndex, readAccess.GetPixelByIndex(inputImageIndex));
      }
    }
  }

  // Now we write a volume. We just output the extra header info required.
  // The user can manually append it to the file if necessary.
  itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
  writer->SetFileName(outputImageFileName);
  writer->SetInput(outputImage);
  writer->Update();

  std::cout << "Written meta-data to " << outputDataFileName << std::endl;

  fout << "UltrasoundImageOrientation = UF" << std::endl;
  fout << "UltrasoundImageType = BRIGHTNESS" << std::endl;

  std::string oneZero = "0";
  std::string twoZero = "00";
  std::string threeZero = "000";

  fout.precision(10);
  boost::regex timeStampFilter ( "([0-9]{19})(.)*");
  boost::cmatch what;
  std::string timeStampAsString;
  unsigned long long timeStamp, before, after;
  double proportion;
  int indexBefore, indexAfter;
  cv::Mat interpolatedMatrix( 4, 4, CV_64F );

  std::cout << "Index BeforeTimeStamp BeforeIndex AfterTimeStamp AfterIndex x y z " << std::endl;

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
    fout << "Seq_Frame" << suffix.str() << "_FrameNumber = " << i << std::endl;
    fout << "Seq_Frame" << suffix.str() << "_UnfilteredTimestamp = " << i << std::endl;
    fout << "Seq_Frame" << suffix.str() << "_Timestamp = " << i << std::endl;
    fout << "Seq_Frame" << suffix.str() << "_ProbeToTrackerTransform =";
    fout << std::endl;

    // So, we may have different number of tracking matrices (normally much larger)
    // than the corresponding number of images.

    std::string nameToMatch = niftk::Basename(imageFiles[i]);
    if ( boost::regex_match( nameToMatch.c_str(), what, timeStampFilter) )
    {
      timeStampAsString = nameToMatch.substr(0, 19);
      timeStamp = boost::lexical_cast<unsigned long long>(timeStampAsString);
      proportion = trackingTimeStamps.GetBoundingTimeStamps(timeStamp, before, after);
      indexBefore = trackingTimeStamps.GetFrameNumber(before);
      indexAfter = trackingTimeStamps.GetFrameNumber(after);

      if (indexBefore != -1 && indexAfter != -1)
      {
        mitk::InterpolateTransformationMatrix(matrices[indexBefore], matrices[indexAfter], proportion, interpolatedMatrix);

        for (int r = 0; r < 4; r++)
        {
          for (int c = 0; c < 4; c++)
          {
            fout << " " << interpolatedMatrix.at<double>(r, c);
          }
        }

        std::cout << i << " " << timeStampAsString << " " << before << " " << indexBefore << " " << after << " " << indexAfter << " " << interpolatedMatrix.at<double>(0, 3) << " " << interpolatedMatrix.at<double>(1, 3) << " " << interpolatedMatrix.at<double>(2, 3) << std::endl;

        fout << "Seq_Frame" << suffix.str() << "_ProbeToTrackerTransformStatus = OK" << std::endl;
        fout << "Seq_Frame" << suffix.str() << "_ReferenceToTrackerTransform =";
        for (int r = 0; r < 4; r++)
        {
          for (int c = 0; c < 4; c++)
          {
            // We are not actually tracking a reference object.
            // This is just so that I can get data into fCal.
            fout << " " << identityMatrix(r, c);
          }
        }
        fout << std::endl;
        fout << "Seq_Frame" << suffix.str() << "_ReferenceToTrackerTransformStatus = OK" << std::endl;
        fout << "Seq_Frame" << suffix.str() << "_StylusToTrackerTransform =";
        for (int r = 0; r < 4; r++)
        {
          for (int c = 0; c < 4; c++)
          {
            // We are not actually tracking a stylus object.
            // This is just so that I can get data into fCal.
            fout << " " << identityMatrix(r, c);
          }
        }
        fout << std::endl;
        fout << "Seq_Frame" << suffix.str() << "_StylusToTrackerTransformStatus = OK" << std::endl;
      }
      else
      {
        std::ostringstream errorMessage;
        errorMessage << "Image[" << i << "], before=" << before << ", after=" << after << ", indexBefore=" << indexBefore << ", indexAfter=" << indexAfter << " and neither should be -1" << std::endl;
        mitkThrow() << errorMessage.str();
      }
    }
    else
    {
      std::ostringstream errorMessage;
      errorMessage << "Image " << imageFiles[i] << " does not look like it contains a time-stamp.";
      mitkThrow() << errorMessage.str();
    }
  }

  fout.close();
}


//-----------------------------------------------------------------------------
} // end namespace
