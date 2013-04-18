/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkHandeyeCalibrate.h>

/**
 * Runs ICP registration a known data set and checks the error
 */

int mitkHandeyeCalibrationTest ( int argc, char * argv[] )
{
  std::string inputExtrinsic = argv[1];
  std::string inputTracking = argv[2];
  std::string sort = argv[3];
  std::string result = argv[4];

  mitk::HandeyeCalibrate::Pointer Calibrator = mitk::HandeyeCalibrate::New();
  
  std::vector<cv::Mat> ExtMatrices = Calibrator->LoadMatricesFromExtrinsicFile(inputExtrinsic);
  std::vector<cv::Mat> TrackMatrices = Calibrator->LoadMatricesFromDirectory(inputTracking);

  std::vector<cv::Mat> FlippedTrackMatrices = Calibrator->FlipMatrices(TrackMatrices);

  std::vector<int> indexes;

  if ( sort == "Distances" ) 
  {
    indexes = Calibrator->SortMatricesByDistance(FlippedTrackMatrices);
    std::cout << "Sorted by distances " << std::endl;
    for ( unsigned int i = 0 ; i < indexes.size() ; i++ )
    {
      std::cout << indexes[i] << " " ;
    }
    std::cout << std::endl;
  }
  else 
  {
    if ( sort == "Angles" )
    {
      indexes = Calibrator->SortMatricesByAngle(FlippedTrackMatrices);
      std::cout << "Sorted by Angles " << std::endl;
      for ( unsigned int i = 0 ; i < indexes.size() ; i++ )
      {
        std::cout << indexes[i] << " " ;
      }
      std::cout << std::endl;
    }
    else
    {
      for ( unsigned int i = 0 ; i < FlippedTrackMatrices.size() ; i ++ )
      {
        indexes.push_back(i);
      }
      std::cout << "No Sorting" << std::endl;
      for ( unsigned int i = 0 ; i < indexes.size() ; i++ )
      {
        std::cout << indexes[i] << " " ;
      }
      std::cout << std::endl;
    }
  }

  std::vector<cv::Mat> SortedExtMatrices;
  std::vector<cv::Mat> SortedFlippedTrackMatrices;

  for ( unsigned int i = 0 ; i < indexes.size() ; i ++ )
  {
    SortedExtMatrices.push_back(ExtMatrices[indexes[i]]);
    SortedFlippedTrackMatrices.push_back(FlippedTrackMatrices[indexes[i]]);
  }



  cv::Mat CamToMarker = Calibrator->Calibrate( SortedFlippedTrackMatrices, SortedExtMatrices);
  std::cout << "Camera to Marker Matrix = " << std::endl << CamToMarker << std::endl;
  return EXIT_SUCCESS;
}
