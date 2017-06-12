/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkAtracsysClientCLP.h>
#include <niftkAtracsysTracker.h>
#include <niftkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkException.h>
#include <igtlTimeStamp.h>
#include <iostream>
#include <cv.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <niftkMITKMathsUtils.h>
#include <mitkOpenCVMaths.h>
#include <memory>

int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  // Early exit if main outpt file not specified.
  if (outputFile.size() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    std::ofstream opf;
    opf.open(outputFile);
    if (opf.is_open())
    {
      mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
      niftk::AtracsysTracker::Pointer tracker = niftk::AtracsysTracker::New(dataStorage.GetPointer(), toolStorage);

      unsigned long int counter = 0;
      igtl::TimeStamp::Pointer t = igtl::TimeStamp::New();

      std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > markers;
      std::vector<mitk::Point3D> balls;

      std::map<std::string, std::unique_ptr<std::vector<cv::Mat> > > matricesToAverage;
      std::map<int, std::unique_ptr<std::vector<mitk::Point3D> > > pointsToAverage;
      vtkSmartPointer<vtkMatrix4x4> tmpVTKMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      cv::Mat tmpOpenCVMatrix = cvCreateMat(4, 4, CV_64FC1);

      do
      {
        t->GetTime();
        tracker->GetMarkersAndBalls(markers, balls);

        std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >::const_iterator iter;
        for (iter = markers.begin(); iter != markers.end(); ++iter)
        {
          opf <<  t->GetTimeStampInNanoseconds() << " "
              << (*iter).first << " "
              << (*iter).second.first[0] << " "
              << (*iter).second.first[1] << " "
              << (*iter).second.first[2] << " "
              << (*iter).second.first[3] << " "
              << (*iter).second.second[0] << " "
              << (*iter).second.second[1] << " "
              << (*iter).second.second[2] << " "
              << std::endl;
          if (average)
          {
            // A bit round-the-houses.
            niftk::ConvertRotationAndTranslationToMatrix((*iter).second.first, (*iter).second.second, *tmpVTKMatrix);
            mitk::CopyToOpenCVMatrix(*tmpVTKMatrix, tmpOpenCVMatrix);

            // Creates initial vector with nothing in it.
            if (matricesToAverage.find((*iter).first) == matricesToAverage.end())
            {
              std::unique_ptr<std::vector<cv::Mat> > tmp(new std::vector<cv::Mat>());
              matricesToAverage.insert(std::pair<std::string, std::unique_ptr<std::vector<cv::Mat> > >((*iter).first, std::move(tmp)));
            }
            // Appends data to vector.
            matricesToAverage[(*iter).first]->push_back(tmpOpenCVMatrix);
          }
        }

        for (int i = 0; i < balls.size(); i++)
        {
          opf << t->GetTimeStampInNanoseconds() << " "
              << i << " " 
              << balls[i][0] << " "
              << balls[i][1] << " "
              << balls[i][2] << " "
              << std::endl;

          // Creates initial vector with nothing in it.
          if (pointsToAverage.find(i) == pointsToAverage.end())
          {
            std::unique_ptr<std::vector<mitk::Point3D> > tmp(new std::vector<mitk::Point3D>());
            pointsToAverage.insert(std::pair<int, std::unique_ptr<std::vector<mitk::Point3D> > >(i, std::move(tmp)));
          }
          // Appends data to vector.
          pointsToAverage[i]->push_back(balls[i]);
        }
        if (balls.size() > 0 || markers.size() > 0)
        {
          counter++;
        }
      } while (counter < numberSamples);

      returnStatus = EXIT_SUCCESS;

      opf.close();

      if (average)
      {
        std::map<std::string, std::unique_ptr<std::vector<cv::Mat> > >::const_iterator mIter;
        for (mIter = matricesToAverage.begin(); mIter != matricesToAverage.end(); ++mIter)
        {
          cv::Mat ave = mitk::AverageMatrices(*((*mIter).second));
          std::cout << "Average of:" << (*mIter).first << std::endl;
          for (int r = 0; r < 4; r++)
          {
            std::cout << "  "
                      << ave.at<double>(r, 0) << " "
                      << ave.at<double>(r, 1) << " "
                      << ave.at<double>(r, 2) << " "
                      << ave.at<double>(r, 3) << std::endl;
          }
        }

        std::map<int, std::unique_ptr<std::vector<mitk::Point3D> > >::const_iterator pIter;
        for (pIter = pointsToAverage.begin(); pIter != pointsToAverage.end(); pIter++)
        {
          mitk::Point3D ave;
          ave.Fill(0);
          for (int i = 0; i < (*((*pIter).second)).size(); i++)
          {
            ave[0] += (*((*pIter).second))[i][0];
            ave[1] += (*((*pIter).second))[i][1];
            ave[2] += (*((*pIter).second))[i][2];
          }
          unsigned int num = (*((*pIter).second)).size();
          ave[0] /= static_cast<double>(num);
          ave[1] /= static_cast<double>(num);
          ave[2] /= static_cast<double>(num);
          std::cout << "Average of point:" << (*pIter).first << std::endl;
          std::cout << "  " << ave[0] << " " << ave[1] << " " << ave[2] << std::endl;
        }
      }
    }
    else
    {
      std::cerr << "Failed to open file:" << outputFile << std::endl;
      returnStatus = EXIT_SUCCESS + 1;
    }
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
