/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <niftkAverageTrackingMatricesCLP.h>
#include <niftkFileHelper.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkFileIOUtils.h>
#include <mitkExceptionMacro.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( inputDirectory.length() == 0 || outputMatrixFile.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    std::vector<std::string> files = niftk::GetFilesInDirectory(inputDirectory);
    if (files.size() == 0)
    {
      std::ostringstream errorMessage;
      errorMessage << "No files in directory:" << inputDirectory << std::endl;
      mitkThrow() << errorMessage.str();
    }

    std::vector<double> rx(files.size());
    std::vector<double> ry(files.size());
    std::vector<double> rz(files.size());
    std::vector<double> tx(files.size());
    std::vector<double> ty(files.size());
    std::vector<double> tz(files.size());

    cv::Mat trackingMatrix = cv::Mat(4,4, CV_64FC1);
    cv::Mat rotationMatrix = cv::Mat(3,3, CV_64FC1);
    cv::Mat rotationVector = cv::Mat(1,3, CV_64FC1);
    cv::Mat translationVector = cv::Mat(1,3, CV_64FC1);
    unsigned long int numberOfFiles = 0;

    for (unsigned long int i = 0; i < files.size(); i++)
    {
      if (   niftk::FilenameHasPrefixAndExtension(files[i], "", "txt")
          || niftk::FilenameHasPrefixAndExtension(files[i], "", "mat")
          || niftk::FilenameHasPrefixAndExtension(files[i], "", "4x4"))
      {
        bool isSuccessfullyRead = mitk::ReadTrackerMatrix(files[i], trackingMatrix);
        if (isSuccessfullyRead)
        {
          for (int j = 0; j < 3; j++)
          {
            for (int k = 0; k < 3; k++)
            {
              rotationMatrix.at<double>(j, k) = trackingMatrix.at<double>(j, k);
            }
          }
          cv::Rodrigues(rotationMatrix, rotationVector);

          rx[i] = (rotationVector.at<double>(0,0));
          ry[i] = (rotationVector.at<double>(0,1));
          rz[i] = (rotationVector.at<double>(0,2));
          tx[i] = trackingMatrix.at<double>(0,3);
          ty[i] = trackingMatrix.at<double>(1,3);
          tz[i] = trackingMatrix.at<double>(2,3);

          numberOfFiles++;
        }
      }
    }

    std::cout << "Read " << numberOfFiles << " tracking matrices." << std::endl;

    if (numberOfFiles < 2)
    {
      std::ostringstream errorMessage;
      errorMessage << "Not enough tracking matrices, n=" << numberOfFiles << std::endl;
      mitkThrow() << errorMessage.str();
    }

    assert(rx.size() == numberOfFiles);
    assert(ry.size() == numberOfFiles);
    assert(rz.size() == numberOfFiles);
    assert(tx.size() == numberOfFiles);
    assert(ty.size() == numberOfFiles);
    assert(tz.size() == numberOfFiles);

    std::sort(rx.begin(), rx.end());
    std::sort(ry.begin(), ry.end());
    std::sort(rz.begin(), rz.end());
    std::sort(tx.begin(), tx.end());
    std::sort(ty.begin(), ty.end());
    std::sort(tz.begin(), tz.end());

    if (numberOfFiles%2 == 0)
    {
      int indexHigh = numberOfFiles / 2;
      int indexLow = indexHigh - 1;

      rotationVector.at<double>(0,0) = (rx[indexLow] + rx[indexHigh])/2.0;
      rotationVector.at<double>(0,1) = (ry[indexLow] + ry[indexHigh])/2.0;
      rotationVector.at<double>(0,2) = (rz[indexLow] + rz[indexHigh])/2.0;
      translationVector.at<double>(0,0) = (tx[indexLow] + tx[indexHigh])/2.0;
      translationVector.at<double>(0,1) = (ty[indexLow] + ty[indexHigh])/2.0;
      translationVector.at<double>(0,2) = (tz[indexLow] + tz[indexHigh])/2.0;

      std::cout << "Calculating median from average of indexes " << indexLow << " and " << indexHigh << std::endl;
    }
    else
    {
      int indexOfMiddleElement = (numberOfFiles - 1) / 2;

      rotationVector.at<double>(0,0) = rx[indexOfMiddleElement];
      rotationVector.at<double>(0,1) = ry[indexOfMiddleElement];
      rotationVector.at<double>(0,2) = rz[indexOfMiddleElement];
      translationVector.at<double>(0,0) = tx[indexOfMiddleElement];
      translationVector.at<double>(0,1) = ty[indexOfMiddleElement];
      translationVector.at<double>(0,2) = tz[indexOfMiddleElement];

      std::cout << "Calculating median from index " << indexOfMiddleElement << std::endl;
    }


    cv::Rodrigues(rotationVector, rotationMatrix);

    vtkSmartPointer<vtkMatrix4x4> outputMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    outputMatrix->Identity();

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        outputMatrix->SetElement(i, j, rotationMatrix.at<double>(i,j));
      }
      outputMatrix->SetElement(i, 3, translationVector.at<double>(0,i));
    }

    bool isSuccessfullyWritten = mitk::SaveVtkMatrix4x4ToFile(outputMatrixFile, *outputMatrix);
    if (!isSuccessfullyWritten)
    {
      std::ostringstream errorMessage;
      errorMessage << "Failed to write:" << outputMatrixFile << std::endl;
      mitkThrow() << errorMessage.str();
    }

    // Extra output
    std::cerr << "Written to:" << outputMatrixFile << std::endl;
    for (int i = 0; i < 4; i++)
    {
      std::cout << outputMatrix->GetElement(i, 0) << " " << outputMatrix->GetElement(i, 1) << " " << outputMatrix->GetElement(i, 2) << " " << outputMatrix->GetElement(i, 3) << std::endl;
    }

    // Done
    returnStatus = EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception:";
    returnStatus = -2;
  }
  return returnStatus;
}
