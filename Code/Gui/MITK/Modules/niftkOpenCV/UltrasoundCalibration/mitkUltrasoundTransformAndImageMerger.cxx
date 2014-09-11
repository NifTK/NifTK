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
#include <mitkTrackingAndTimeStampsContainer.h>
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
void UltrasoundTransformAndImageMerger::Merge(const std::string& inputMatrixDirectory,
    const std::string& inputImageDirectory,
    const std::string& outputImageFileName,
    const std::string& imageOrientation)
{
  cv::Matx44d identityMatrix;
  mitk::MakeIdentity(identityMatrix);

  mitk::TrackingAndTimeStampsContainer trackingTimeStamps;
  trackingTimeStamps.LoadFromDirectory(inputMatrixDirectory);

  std::vector<std::string> imageFiles = niftk::GetFilesInDirectory(inputImageDirectory);
  std::sort(imageFiles.begin(), imageFiles.end());

  // Load all images. OK, so this will eventually run out of memory, but its ok for now.
  std::vector<mitk::Image::Pointer> images;
  for (int i = 0; i < imageFiles.size(); i++)
  {
    images.push_back(mitk::IOUtil::LoadImage(imageFiles[i]));
  }

  std::cout << "Number of matrices=" << trackingTimeStamps.GetSize() << std::endl;
  std::cout << "Number of images=" << imageFiles.size() << std::endl;

  if (trackingTimeStamps.GetSize() < imageFiles.size())
  {
    std::ostringstream errorMessage;
    errorMessage << "Loaded " << trackingTimeStamps.GetSize() << " matrices, and loaded a difference number of images " << images.size() << ", and number of images must be less than number of matrices." << std::endl;
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
  std::string outputImgFile = outputImageFileName + ".mhd";
  itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();
  writer->SetFileName(outputImgFile);
  writer->SetInput(outputImage);
  writer->Update();

  std::cout << "Written image data to " << outputImgFile << std::endl;

  // Read .mhd header file.
  std::vector<std::string> linesFromMhdFile;
  std::ifstream fin(outputImgFile.c_str());
  if ( !fin )
  {
    std::ostringstream errorMessage;
    errorMessage << "Could not open " << outputImgFile << " for reading!" << std::endl;
    mitkThrow() << errorMessage.str();
  }
  char lineOfText[256];
  do {
    fin.getline(lineOfText,256);
    if (fin.good())
    {
      linesFromMhdFile.push_back(std::string(lineOfText));
      std::cout << "Read:" << lineOfText << std::endl;
    }
  } while (fin.good());
  fin.close();

  // Now, re-open file .mhd file to add meta-data.
  std::ofstream fout(outputImgFile.c_str(), std::ios::out | std::ios::app);
  if ( !fout )
  {
    std::ostringstream errorMessage;
    errorMessage << "Could not open " << outputImgFile << " for text output!" << std::endl;
    mitkThrow() << errorMessage.str();
  }

  // Write everything except the last string of the existing header.
  for (unsigned int i = 0; i < linesFromMhdFile.size() - 1; i++)
  {
    fout << linesFromMhdFile[i] << std::endl;
  }

  fout << "UltrasoundImageOrientation = " << imageOrientation << std::endl;
  fout << "UltrasoundImageType = BRIGHTNESS" << std::endl;

  std::string oneZero = "0";
  std::string twoZero = "00";
  std::string threeZero = "000";

  fout.precision(10);

  boost::regex timeStampFilter ( "([0-9]{19})(.)*");
  boost::cmatch what;
  std::string timeStampAsString;
  unsigned long long timeStamp;
  unsigned long long timingError;
  unsigned long long timeStampFirstFrame = 0;
  double timeStampInSeconds = 0;
  cv::Matx44d interpolatedMatrix;

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

    std::string nameToMatch = niftk::Basename(imageFiles[i]);
    if ( boost::regex_match( nameToMatch.c_str(), what, timeStampFilter) )
    {
      timeStampAsString = nameToMatch.substr(0, 19);
      timeStamp = boost::lexical_cast<unsigned long long>(timeStampAsString);
      if (timeStampFirstFrame == 0)
      {
        timeStampFirstFrame = timeStamp;
      }
      interpolatedMatrix = trackingTimeStamps.InterpolateMatrix(timeStamp, timingError);
      timeStampInSeconds = (timeStamp - timeStampFirstFrame)/static_cast<double>(1000000000);

      fout << "Seq_Frame" << suffix.str() << "_FrameNumber = " << i << std::endl;
      fout << "Seq_Frame" << suffix.str() << "_UnfilteredTimestamp = " << timeStampInSeconds << std::endl;
      fout << "Seq_Frame" << suffix.str() << "_Timestamp = " << timeStampInSeconds << std::endl;
      fout << "Seq_Frame" << suffix.str() << "_ProbeToTrackerTransform =";

      for (int r = 0; r < 4; r++)
      {
        for (int c = 0; c < 4; c++)
        {
          fout << " " << interpolatedMatrix(r, c);
        }
      }
      fout << std::endl;

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
      errorMessage << "Image " << imageFiles[i] << " does not look like it contains a time-stamp." << std::endl;
      mitkThrow() << errorMessage.str();
    }
  }

  fout << linesFromMhdFile[linesFromMhdFile.size() - 1];
  fout.close();

  std::cout << "Written meta-data to " << outputImgFile << std::endl;
}


//-----------------------------------------------------------------------------
} // end namespace
