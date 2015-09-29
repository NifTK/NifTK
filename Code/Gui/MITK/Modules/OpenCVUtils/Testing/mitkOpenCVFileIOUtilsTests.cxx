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

#include <niftkFileHelper.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkOpenCVMaths.h>
#include <cmath>

/**
 * \file Tests for some of the functions in openCVFileIOUtils.
 */

void LoadTimeStampedPointsTest(std::string dir)
{
  std::string pointdir = dir + "pins";
  std::string matrixdir = dir + "Aurora_2/1";

  std::vector < std::pair<unsigned long long, cv::Point3d> > timeStampedPoints = mitk::LoadTimeStampedPoints(pointdir);
  
  MITK_TEST_CONDITION ( timeStampedPoints.size() == 102 , "Testing 102 points were loaded. " << timeStampedPoints.size() );
  MITK_TEST_CONDITION ( timeStampedPoints[0].first == 1429793163701888600 , "Testing first time stamp " <<  timeStampedPoints[0].first);
  MITK_TEST_CONDITION ( timeStampedPoints[101].first == 1429793858654637600 , "Testing last time stamp " <<  timeStampedPoints[101].first);
  
  MITK_TEST_CONDITION ( mitk::NearlyEqual (timeStampedPoints[0].second,cv::Point3d (317,191.0,0.0),1e-6), "Testing first point value " <<  timeStampedPoints[0].second);
  MITK_TEST_CONDITION ( mitk::NearlyEqual (timeStampedPoints[101].second, cv::Point3d (345.0, 162.0, 0.0),1e-6), "Testing last time stamp " <<  timeStampedPoints[87].second);

}

void TestLoadPickedObject ( char * filename )
{
  MITK_INFO << "Attemting to open " << filename;
  std::vector <mitk::PickedObject> p1;
  std::ifstream stream;
  stream.open(filename);
  if ( stream )
  {
    LoadPickedObjects ( p1, stream ); 
  }
  else
  {
    MITK_ERROR << "Failed to open " << filename;
  }
  MITK_TEST_CONDITION ( p1.size() == 13 , "Testing that 13 picked objects were read : " << p1.size());
  MITK_TEST_CONDITION ( p1[3].m_TimeStamp == 1421406496123439600, "Testing point 4 time stamp is 1421406496123439600 : " << p1[4].m_TimeStamp );
  MITK_TEST_CONDITION ( p1[1].m_Id == 2, "Testing point 1 ID is 2 : " << p1[2].m_Id );
  MITK_TEST_CONDITION ( p1[8].m_FrameNumber == 0, "Testing point 8 frame number  is 0 : " << p1[9].m_FrameNumber );
  MITK_TEST_CONDITION ( p1[9].m_Channel ==  "left", "Testing point 9 channel  is left : " << p1[10].m_Channel );
  MITK_TEST_CONDITION ( p1[0].m_IsLine , "Testing point 0 is line " << p1[0].m_IsLine );
  MITK_TEST_CONDITION ( ! p1[4].m_IsLine, "Testing point 4 is not a line " << p1[4].m_IsLine );
  MITK_TEST_CONDITION ( p1[11].m_Points.size() == 3 , "Testing there are 3 points in object 11" << p1[11].m_Points.size());
  MITK_TEST_CONDITION ( mitk::NearlyEqual(p1[12].m_Points[1], cv::Point3d ( 262, 98, 0), 1e-8 ) , "Testing value of point 2 in object 13" << p1[12].m_Points[1]);

}

void TestLoadPickedPointListFromDirectory ( char * directory )
{
  mitk::PickedPointList::Pointer ppl = mitk::LoadPickedPointListFromDirectory ( directory  );

  MITK_TEST_CONDITION ( ppl->GetListSize() == 9 , "Testing that there are 9 picked objects in the list : " << ppl->GetListSize() );
}

int mitkOpenCVFileIOUtilsTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVFileIOUtilsTests");

  LoadTimeStampedPointsTest(argv[1]);
  TestLoadPickedObject(argv[2]);
  TestLoadPickedPointListFromDirectory(argv[3]);
  MITK_TEST_END();
}



