/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMakeGridOf2DImages.h"

#include <cv.h>
#include <highgui.h>
#include <mitkExceptionMacro.h>

#include <niftkFileHelper.h>

namespace mitk {

//-----------------------------------------------------------------------------
MakeGridOf2DImages::MakeGridOf2DImages()
{

}


//-----------------------------------------------------------------------------
MakeGridOf2DImages::~MakeGridOf2DImages()
{

}


//-----------------------------------------------------------------------------
void MakeGridOf2DImages::MakeGrid(const std::string &inputDirectory,
                                  const std::vector<int>& imageSize,
                                  const std::vector<int>& gridDimensions,
                                  const std::string &outputImageFile
                                 )
{

  if (gridDimensions.size() != 2)
  {
    mitkThrow() << "ERROR: the grid size did not contain two comma separated numbers!";
  }
  if (imageSize.size() != 2)
  {
    mitkThrow() << "ERROR: the image size did not contain two comma separated numbers!";
  }

  // Scan directory for any files.
  // We will try and load as many as possible, up until
  // we have enough images to tile the output image with.
  std::vector<std::string> files = niftk::GetFilesInDirectory(inputDirectory);
  if (files.size() == 0)
  {
    mitkThrow() << "ERROR: No files found in:" << inputDirectory;
  }
  if (files.size() < gridDimensions[0]*gridDimensions[1])
  {
    mitkThrow() << "ERROR: There are not enough files for the requested number of tiles!";
  }

  cv::Mat firstImage;
  firstImage = cv::imread(files[0], CV_LOAD_IMAGE_COLOR);

  std::cout << "First image has size cols=" << firstImage.cols << " x rows=" << firstImage.rows << std::endl;

  if (gridDimensions[0]*firstImage.cols != imageSize[0])
  {
    mitkThrow() << "ERROR: Incorrect image widths!";
  }

  if (gridDimensions[1]*firstImage.rows != imageSize[1])
  {
    mitkThrow() << "ERROR: Incorrect image heights!";
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
    mitkThrow() << "ERROR: Not enough images for " << gridDimensions[0] << " x " << gridDimensions[1];
  }

  // create output image.
  cv::Mat output = cvCreateMat (imageSize[1], imageSize[0], CV_8UC1);

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
          output.at<uchar>(outputPixelY, outputPixelX, 0) = images[imageNumber].at<uchar>(j, i, 0);
        }
      }
    }
  }

  // Save image
  cv::imwrite( outputImageFile, output );
}

} // end namespace
