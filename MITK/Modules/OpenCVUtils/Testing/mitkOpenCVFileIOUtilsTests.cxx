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
#include <mitkIOUtil.h>
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

void TestLoadPickedPointListFromDirectoryOfMPSFiles ( char * directory )
{
  mitk::PickedPointList::Pointer ppl = mitk::LoadPickedPointListFromDirectoryOfMPSFiles ( directory  );

  MITK_TEST_CONDITION ( ppl->GetListSize() == 9 , "Testing that there are 9 picked objects in the list : " << ppl->GetListSize() );
  MITK_TEST_CONDITION ( ppl->GetNumberOfPoints() == 5, "Testing that there are 5 picked points in the list : " << ppl->GetNumberOfPoints() );
  MITK_TEST_CONDITION ( ppl->GetNumberOfLines() == 4, "Testing that there are 4 picked lines in the list : " << ppl->GetNumberOfLines() );

  std::vector <mitk::PickedObject> pickedObjects = ppl->GetPickedObjects();

  bool point_0_found = false;
  bool point_3_found = false;
  bool point_5_found = false;
  bool line_4_found = false;
  for ( std::vector<mitk::PickedObject>::const_iterator it = pickedObjects.begin() ; it < pickedObjects.end() ; ++it )
  {
    if ( ( it->m_IsLine == false ) && ( it->m_Id == 0 ) )
    {
      point_0_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(-108.62, -35.3123, 1484.7), 1e-6) ,
          "Testing Value of point 0 = " << it->m_Points[0] );
    }
    if ( ( it->m_IsLine == false ) && ( it->m_Id == 5 ) )
    {
      point_5_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(-2.38586, -82.0263, 1509.76), 1e-6) ,
          "Testing Value of point 5 = " << it->m_Points[0] );
    }
    if ( ( it->m_IsLine == true ) && ( it->m_Id == 4 ) )
    {
      line_4_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(-0.270583, -85.0786, 1510.05), 1e-6) ,
          "Testing Value of point 0 in line 4 = " << it->m_Points[0] );
    }
  }
  MITK_TEST_CONDITION ( point_0_found , "Testing that point 0 was found" );
  MITK_TEST_CONDITION ( ! point_3_found , "Testing that point 3 was not found" );
  MITK_TEST_CONDITION ( point_5_found , "Testing that point 5 was found" );
  MITK_TEST_CONDITION ( line_4_found , "Testing that line 4 was found" );

  //the base directory contains point lists in MITK's legacy format. Lets check that we get the same result with MITK's new format
  mitk::PickedPointList::Pointer ppl_v2 = mitk::LoadPickedPointListFromDirectoryOfMPSFiles ( directory + niftk::GetFileSeparator() + "v2" );
  MITK_TEST_CONDITION ( ppl_v2->GetListSize() == 6 , "Testing that there are 6 picked objects in the new formatt list : " << ppl_v2->GetListSize() );
  MITK_TEST_CONDITION ( ppl_v2->GetNumberOfPoints() == 5, "Testing that there are 5 picked points in the list : " << ppl_v2->GetNumberOfPoints() );
  MITK_TEST_CONDITION ( ppl_v2->GetNumberOfLines() == 1, "Testing that there are 1 picked lines in the list : " << ppl_v2->GetNumberOfLines() );

  pickedObjects = ppl_v2->GetPickedObjects();

  point_0_found = false;
  point_3_found = false;
  point_5_found = false;
  line_4_found = false;
  for ( std::vector<mitk::PickedObject>::const_iterator it = pickedObjects.begin() ; it < pickedObjects.end() ; ++it )
  {
    if ( ( it->m_IsLine == false ) && ( it->m_Id == 0 ) )
    {
      point_0_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(-108.62, -35.3123, 1484.7), 1e-6) ,
          "Testing Value of point 0 = " << it->m_Points[0] );
    }
    if ( ( it->m_IsLine == false ) && ( it->m_Id == 5 ) )
    {
      point_5_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(-2.38586, -82.0263, 1509.76), 1e-6) ,
          "Testing Value of point 5 = " << it->m_Points[0] );
    }
    if ( ( it->m_IsLine == true ) && ( it->m_Id == 4 ) )
    {
      line_4_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(-0.270583, -85.0786, 1510.05), 1e-6) ,
          "Testing Value of point 0 in line 4 = " << it->m_Points[0] );
    }
  }
  MITK_TEST_CONDITION ( point_0_found , "Testing that point 0 was found" );
  MITK_TEST_CONDITION ( ! point_3_found , "Testing that point 3 was not found" );
  MITK_TEST_CONDITION ( point_5_found , "Testing that point 5 was found" );
  MITK_TEST_CONDITION ( line_4_found , "Testing that line 4 was found" );

  //and what happens when we move it
  mitk::PickedPointList::Pointer ppl_v2_moved = mitk::LoadPickedPointListFromDirectoryOfMPSFiles ( directory + niftk::GetFileSeparator() + "v2_moved" );
  MITK_TEST_CONDITION ( ppl_v2_moved->GetListSize() == 6 , "Testing that there are 6 picked objects in the new formatt list : " << ppl_v2_moved->GetListSize() );
  MITK_TEST_CONDITION ( ppl_v2_moved->GetNumberOfPoints() == 5, "Testing that there are 5 picked points in the list : " << ppl_v2_moved->GetNumberOfPoints() );
  MITK_TEST_CONDITION ( ppl_v2_moved->GetNumberOfLines() == 1, "Testing that there are 1 picked lines in the list : " << ppl_v2_moved->GetNumberOfLines() );

  pickedObjects = ppl_v2_moved->GetPickedObjects();

  point_0_found = false;
  point_3_found = false;
  point_5_found = false;
  line_4_found = false;
  for ( std::vector<mitk::PickedObject>::const_iterator it = pickedObjects.begin() ; it < pickedObjects.end() ; ++it )
  {
    if ( ( it->m_IsLine == false ) && ( it->m_Id == 0 ) )
    {
      point_0_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d( -88.6200, 691.7687, 1403.4441 ), 1e-3) ,
          "Testing Value of point 0 = " << it->m_Points[0] );
    }
    if ( ( it->m_IsLine == false ) && ( it->m_Id == 5 ) )
    {
      point_5_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(17.6141, 663.8431, 1448.5037), 1e-3) ,
          "Testing Value of point 5 = " << it->m_Points[0] );
    }
    if ( ( it->m_IsLine == true ) && ( it->m_Id == 4 ) )
    {
      line_4_found = true;
      MITK_TEST_CONDITION (mitk::NearlyEqual(it->m_Points[0], cv::Point3d(19.7294, 661.3448, 1450.2810), 1e-3) ,
          "Testing Value of point 0 in line 4 = " << it->m_Points[0] );
    }
  }
  MITK_TEST_CONDITION ( point_0_found , "Testing that point 0 was found" );
  MITK_TEST_CONDITION ( ! point_3_found , "Testing that point 3 was not found" );
  MITK_TEST_CONDITION ( point_5_found , "Testing that point 5 was found" );
  MITK_TEST_CONDITION ( line_4_found , "Testing that line 4 was found" );

}

void TestLoadMPSAndConvertToOpenCVVector ( char * directory )
{
  mitk::PointSet::Pointer pointSet = mitk::IOUtil::LoadPointSet ( directory + niftk::GetFileSeparator() + "points.mps");

  std::vector < cv::Point3d > pointVector = mitk::PointSetToVector ( pointSet );

  MITK_TEST_CONDITION ( pointVector.size() == pointSet->GetSize() , "Testing that point vector size == point set size " <<
      pointVector.size() << " == " << pointSet->GetSize() );
  MITK_TEST_CONDITION (mitk::NearlyEqual(pointVector[0], cv::Point3d(-108.62, -35.3123, 1484.7), 1e-6) ,
          "Testing Value of point 0 = " << pointVector[0] );

  MITK_TEST_CONDITION (mitk::NearlyEqual(pointVector[4], cv::Point3d(-2.38586, -82.0263, 1509.76), 1e-6) ,
          "Testing Value of point 4 = " << pointVector[4] );

  //with new format
  pointSet = mitk::IOUtil::LoadPointSet ( directory + niftk::GetFileSeparator() +
      "v2" + niftk::GetFileSeparator() + "points.mps");

  pointVector = mitk::PointSetToVector ( pointSet );

  MITK_TEST_CONDITION ( pointVector.size() == pointSet->GetSize() , "Testing that point vector size == point set size " <<
      pointVector.size() << " == " << pointSet->GetSize() );
  MITK_TEST_CONDITION (mitk::NearlyEqual(pointVector[0], cv::Point3d(-108.62, -35.3123, 1484.7), 1e-6) ,
          "Testing Value of point 0 = " << pointVector[0] );

  MITK_TEST_CONDITION (mitk::NearlyEqual(pointVector[4], cv::Point3d(-2.38586, -82.0263, 1509.76), 1e-6) ,
          "Testing Value of point 4 = " << pointVector[4] );

  //with new format after move
  pointSet = mitk::IOUtil::LoadPointSet ( directory + niftk::GetFileSeparator() +
      "v2_moved" + niftk::GetFileSeparator() + "points.mps");

  pointVector = mitk::PointSetToVector ( pointSet );

  MITK_TEST_CONDITION ( pointVector.size() == pointSet->GetSize() , "Testing that point vector size == point set size " <<
      pointVector.size() << " == " << pointSet->GetSize() );
  MITK_TEST_CONDITION (mitk::NearlyEqual(pointVector[0], cv::Point3d( -88.6200, 691.7687, 1403.4441 ), 1e-3) ,
          "Testing Value of point 0 = " << pointVector[0] );

  MITK_TEST_CONDITION (mitk::NearlyEqual(pointVector[4], cv::Point3d(17.6141, 663.8431, 1448.5037), 1e-3) ,
          "Testing Value of point 4 = " << pointVector[4] );

}

int mitkOpenCVFileIOUtilsTests(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkOpenCVFileIOUtilsTests");

  LoadTimeStampedPointsTest(argv[1]);
  TestLoadPickedObject(argv[2]);
  TestLoadPickedPointListFromDirectoryOfMPSFiles(argv[3]);
  TestLoadMPSAndConvertToOpenCVVector ( argv[3] );
  MITK_TEST_END();
}



