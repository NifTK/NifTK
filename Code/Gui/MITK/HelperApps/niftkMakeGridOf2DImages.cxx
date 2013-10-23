/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <niftkFileHelper.h>
#include <niftkMakeGridOf2DImagesCLP.h>
#include <cv.h>
#include <highgui.h>

/*!
 * \file niftkMakeGridOf2DImages.cxx
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // Validate command line args
  if (outputImage.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  if (gridDimensions.size() != 2)
  {
    std::cerr << "ERROR: the grid size did not contain two comma separated numbers!" << std::endl;
    return EXIT_FAILURE;
  }
  if (imageSize.size() != 2)
  {
    std::cerr << "ERROR: the image size did not contain two comma separated numbers!" << std::endl;
    return EXIT_FAILURE;
  }

  // Scan directory for any files.
  // We will try and load as many as possible, up until
  // we have enough images to tile the output image with.
  std::vector<std::string> files = niftk::GetFilesInDirectory(directoryName);
  if (files.size() == 0)
  {
    std::cerr << "ERROR: No files found in:" << directoryName << std::endl;
    return EXIT_FAILURE;
  }
  if (files.size() < gridDimensions[0]*gridDimensions[1])
  {
    std::cerr << "ERROR: There are not enough files for the requested number of tiles!" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat firstImage;
  firstImage = cv::imread(files[0], CV_LOAD_IMAGE_COLOR);

  if (gridDimensions[0]*firstImage.cols != imageSize[0])
  {
    std::cerr << "ERROR: Incorrect image widths!" << std::endl;
    return EXIT_FAILURE;
  }

  if (gridDimensions[1]*firstImage.rows != imageSize[1])
  {
    std::cerr << "ERROR: Incorrect image heights!" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> images;
  for (int i = 0; i < files.size(); i++)
  {
    cv::Mat image;
    image = cv::imread(files[i], CV_LOAD_IMAGE_COLOR);

    if(image.data )
    {
      std::cout << "Adding image:" << files[i] << std::endl;
      images.push_back(image);
    }
  }

  if (images.size() < gridDimensions[0]*gridDimensions[1])
  {
    std::cerr << "ERROR: Not enough images for " << gridDimensions[0] << " x " << gridDimensions[1] << std::endl;
    return EXIT_FAILURE;
  }

  // create output image.
  cv::Mat output = cvCreateMat (imageSize[0], imageSize[1], CV_8UC1);

  for (int y = 0; y < gridDimensions[1]; y++)
  {
    for (int x = 0; x < gridDimensions[0]; x++)
    {
      int imageNumber = y*gridDimensions[0] + x;

      for (int j = 0; j < firstImage.rows; j++)
      {
        for (int i = 0; i < firstImage.cols; i++)
        {
          int outputPixelX = x*firstImage.cols + i;
          int outputPixelY = y*firstImage.rows + j;
          output.at<uchar>(outputPixelX, outputPixelY, 0) = images[imageNumber].at<uchar>(i, j, 0);
        }
      }
    }
  }

  // Save image
  cv::imwrite( outputImage, output );

  return EXIT_SUCCESS;
}
