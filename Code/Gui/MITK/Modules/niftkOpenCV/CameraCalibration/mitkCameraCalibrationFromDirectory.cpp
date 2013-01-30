/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkCameraCalibrationFromDirectory.h"
#include "mitkCameraCalibrationFacade.h"
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include "FileHelper.h"

namespace mitk {

//-----------------------------------------------------------------------------
CameraCalibrationFromDirectory::CameraCalibrationFromDirectory()
{

}


//-----------------------------------------------------------------------------
CameraCalibrationFromDirectory::~CameraCalibrationFromDirectory()
{

}


//-----------------------------------------------------------------------------
bool CameraCalibrationFromDirectory::Calibrate(const std::string& fullDirectoryName,
    const int& numberCornersX,
    const int& numberCornersY,
    const float& sizeSquareMillimeters,
    const std::string& outputFile
    )
{
  bool isSuccessful = false;
  int width = 0;
  int height = 0;

  try
  {
    std::vector<IplImage*> images;
    std::vector<std::string> fileNames;

    std::vector<IplImage*> successfullImages;
    std::vector<std::string> successfullFileNames;

    CvMat *imagePoints = NULL;
    CvMat *objectPoints = NULL;
    CvMat *pointCounts = NULL;

    CvMat *intrinsicMatrix = cvCreateMat(3,3,CV_32FC1);
    CvMat *distortionCoeffs = cvCreateMat(5, 1, CV_32FC1);
    CvMat *rotationMatrix = cvCreateMat(3, 3,CV_32FC1);
    CvMat *translationVector = cvCreateMat(3, 1, CV_32FC1);

    LoadChessBoardsFromDirectory(fullDirectoryName, images, fileNames);
    CheckConstImageSize(images, width, height);

    CvSize imageSize = cvGetSize(images[0]);

    ExtractChessBoardPoints(images, fileNames, numberCornersX, numberCornersY, true, successfullImages, successfullFileNames, imagePoints, objectPoints, pointCounts);

    double projectionError = CalibrateSingleCameraIntrinsicParameters(*objectPoints, *imagePoints, *pointCounts, imageSize, *intrinsicMatrix, *distortionCoeffs);
    CalibrateSingleCameraExtrinsicParameters(*objectPoints, *imagePoints, *intrinsicMatrix, *distortionCoeffs, *rotationMatrix, *translationVector);
    //double projectionError = CalibrateSingleCameraParameters(successfullImages.size(), *objectPoints, *imagePoints, *pointCounts, imageSize, *intrinsicMatrix, *distortionCoeffs, *rotationMatrix, *translationVector);

    ostream *os = NULL;
    std::ostringstream oss;
    std::ofstream fs;

    if (outputFile.size() > 0)
    {
      std::string fullOutputFileName = niftk::ConcatenatePath(fullDirectoryName, outputFile);
      fs.open(fullOutputFileName.c_str(), ios::out);
      if (!fs.fail())
      {
        os = &fs;
        std::cout << "Writing to " << fullOutputFileName << std::endl;
      }
      else
      {
        std::cerr << "ERROR: Writing calibration data to file " << fullOutputFileName << " failed!" << std::endl;
      }
    }
    else
    {
      os = &oss;
    }

    // Output Calibration Data.
    os->precision(6);
    *os << CV_MAT_ELEM(*intrinsicMatrix, float, 0, 0) << "," << CV_MAT_ELEM(*intrinsicMatrix, float, 0, 1) << "," << CV_MAT_ELEM(*intrinsicMatrix, float, 0, 2) << std::endl;
    *os << CV_MAT_ELEM(*intrinsicMatrix, float, 1, 0) << "," << CV_MAT_ELEM(*intrinsicMatrix, float, 1, 1) << "," << CV_MAT_ELEM(*intrinsicMatrix, float, 1, 2) << std::endl;
    *os << CV_MAT_ELEM(*intrinsicMatrix, float, 2, 0) << "," << CV_MAT_ELEM(*intrinsicMatrix, float, 2, 1) << "," << CV_MAT_ELEM(*intrinsicMatrix, float, 2, 2) << std::endl;
    *os << CV_MAT_ELEM(*distortionCoeffs, float, 0, 0) << "," << CV_MAT_ELEM(*distortionCoeffs, float, 1, 0) << "," << CV_MAT_ELEM(*distortionCoeffs, float, 2, 0) << "," << CV_MAT_ELEM(*distortionCoeffs, float, 3, 0) << "," << CV_MAT_ELEM(*distortionCoeffs, float, 4, 0) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrix, float, 0, 0) << "," << CV_MAT_ELEM(*rotationMatrix, float, 0, 1) << "," << CV_MAT_ELEM(*rotationMatrix, float, 0, 2) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrix, float, 1, 0) << "," << CV_MAT_ELEM(*rotationMatrix, float, 1, 1) << "," << CV_MAT_ELEM(*rotationMatrix, float, 1, 2) << std::endl;
    *os << CV_MAT_ELEM(*rotationMatrix, float, 2, 0) << "," << CV_MAT_ELEM(*rotationMatrix, float, 2, 1) << "," << CV_MAT_ELEM(*rotationMatrix, float, 2, 2) << std::endl;
    *os << CV_MAT_ELEM(*translationVector, float, 0, 0) << "," << CV_MAT_ELEM(*translationVector, float, 1, 0) << "," << CV_MAT_ELEM(*translationVector, float, 2, 0) << std::endl;

    *os << projectionError << std::endl;
    *os << width << " " << height << std::endl;
    *os << numberCornersX << " " << numberCornersY << std::endl;

    *os << successfullFileNames.size() << std::endl;

    // Also output files actually used.
    for (unsigned int i = 0; i < successfullFileNames.size(); i++)
    {
      *os << successfullFileNames[i] << std::endl;
    }

    // Tidy up.
    if(fs.is_open())
    {
      fs.close();
    }
    cvReleaseMat(&imagePoints);
    cvReleaseMat(&objectPoints);
    cvReleaseMat(&pointCounts);
    cvReleaseMat(&intrinsicMatrix);
    cvReleaseMat(&distortionCoeffs);
    cvReleaseMat(&rotationMatrix);
    cvReleaseMat(&translationVector);

    for (unsigned int i = 0; i < images.size(); i++)
    {
      cvReleaseImage(&images[i]);
    }

    isSuccessful = true;
  }
  catch(std::logic_error e)
  {
    std::cerr << "CameraCalibrationFromDirectory::Calibrate: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace
