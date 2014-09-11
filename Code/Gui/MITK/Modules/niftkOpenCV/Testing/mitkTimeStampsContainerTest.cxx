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
#include <mitkTimeStampsContainer.h>

/**
 * \file Test harness for mitk::TimeStampsContainer.
 */
int mitkTimeStampsContainerTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkTimeStampsContainerTest");

  mitk::TimeStampsContainer timeStamps;
  unsigned long long result = 0;
  unsigned long long before = 0;
  unsigned long long after = 0;
  long long delta = 0;
  double proportion = 0;
  bool isValid = false;

  // Test GetFrameNumber().
  MITK_TEST_CONDITION (timeStamps.GetFrameNumber(1234) == -1, "GetFrameNumber(): Empty list, expecting -1 and got:" << timeStamps.GetFrameNumber(1234));
  timeStamps.Insert(1);
  MITK_TEST_CONDITION (timeStamps.GetFrameNumber(1234) == -1, "GetFrameNumber(): Single item in list, expecting -1, and got:" << timeStamps.GetFrameNumber(1234));
  timeStamps.Insert(2);
  MITK_TEST_CONDITION (timeStamps.GetFrameNumber(1234) == -1, "GetFrameNumber(): Two item in list, expecting -1, and got:" << timeStamps.GetFrameNumber(1234));
  MITK_TEST_CONDITION (timeStamps.GetFrameNumber(1) == 0, "GetFrameNumber(): Find first item in list, expecting 0, and got:" << timeStamps.GetFrameNumber(1));
  MITK_TEST_CONDITION (timeStamps.GetFrameNumber(2) == 1, "GetFrameNumber(): Find second item in list, expecting 1, and got:" << timeStamps.GetFrameNumber(2));

  // Test GetBoundingTimeStamps();
  timeStamps.Clear();
  isValid = timeStamps.GetBoundingTimeStamps(1, before, after, proportion);
  MITK_TEST_CONDITION (!isValid, "GetBoundingTimeStamps(): Empty list, expecting isValid==false, and got:" << isValid);
  MITK_TEST_CONDITION (before == 0, "GetBoundingTimeStamps(): Empty list, expecting before==0, and got:" << before);
  MITK_TEST_CONDITION (after == 0, "GetBoundingTimeStamps(): Empty list, expecting after==0, and got:" << after);
  MITK_TEST_CONDITION (proportion == 0, "GetBoundingTimeStamps(): Empty list, expecting proportion==0, and got:" << proportion);
  timeStamps.Insert(2);
  timeStamps.Insert(4);
  timeStamps.Insert(6);
  timeStamps.Insert(8);
  isValid = timeStamps.GetBoundingTimeStamps(10, before, after, proportion);
  MITK_TEST_CONDITION (!isValid, "GetBoundingTimeStamps(): Off top end of list, expecting isValid==false, and got:" << isValid);
  MITK_TEST_CONDITION (before == 8, "GetBoundingTimeStamps(): Off top end of list, expecting before==8, and got:" << before);
  MITK_TEST_CONDITION (after == 0, "GetBoundingTimeStamps(): Off top end of list, expecting after==0, and got:" << after);
  MITK_TEST_CONDITION (proportion == 0, "GetBoundingTimeStamps(): Off top end of list, expecting proportion==0, and got:" << proportion);
  isValid = timeStamps.GetBoundingTimeStamps(1, before, after, proportion);
  MITK_TEST_CONDITION (!isValid, "GetBoundingTimeStamps(): Off bottom end of list, expecting isValid==false, and got:" << isValid);
  MITK_TEST_CONDITION (before == 0, "GetBoundingTimeStamps(): Off bottom end of list, expecting before==0, and got:" << before);
  MITK_TEST_CONDITION (after == 2, "GetBoundingTimeStamps(): Off bottom end of list, expecting after==2, and got:" << after);
  MITK_TEST_CONDITION (proportion == 0, "GetBoundingTimeStamps(): Off bottom end of list, expecting proportion==0, and got:" << proportion);
  isValid = timeStamps.GetBoundingTimeStamps(6, before, after, proportion);
  MITK_TEST_CONDITION (isValid, "GetBoundingTimeStamps(): Exact match, expecting isValid==true, and got:" << isValid);
  MITK_TEST_CONDITION (before == 6, "GetBoundingTimeStamps(): Exact match, expecting before==6, and got:" << before);
  MITK_TEST_CONDITION (after == 6, "GetBoundingTimeStamps(): Exact match, expecting after==6, and got:" << after);
  MITK_TEST_CONDITION (proportion == 0, "GetBoundingTimeStamps(): Exact match, expecting proportion==0, and got:" << proportion);
  isValid = timeStamps.GetBoundingTimeStamps(7, before, after, proportion);
  MITK_TEST_CONDITION (isValid, "GetBoundingTimeStamps(): Interpolating match, expecting true==false, and got:" << isValid);
  MITK_TEST_CONDITION (before == 6, "GetBoundingTimeStamps(): Interpolating match, expecting before==6, and got:" << before);
  MITK_TEST_CONDITION (after == 8, "GetBoundingTimeStamps(): Interpolating match, expecting after==8, and got:" << after);
  MITK_TEST_CONDITION (proportion == 0.5, "GetBoundingTimeStamps(): Interpolating match, expecting proportion==0.5, and got:" << proportion);
  isValid = timeStamps.GetBoundingTimeStamps(2, before, after, proportion);
  MITK_TEST_CONDITION (isValid, "GetBoundingTimeStamps(): Exact match first in list, expecting true==true, and got:" << isValid);
  MITK_TEST_CONDITION (before == 2, "GetBoundingTimeStamps(): Exact match first in list, expecting before==2, and got:" << before);
  MITK_TEST_CONDITION (after == 2, "GetBoundingTimeStamps(): Exact match first in list, expecting after==2, and got:" << after);
  MITK_TEST_CONDITION (proportion == 0.0, "GetBoundingTimeStamps(): Exact match first in list, expecting proportion==0.0, and got:" << proportion);
  isValid = timeStamps.GetBoundingTimeStamps(8, before, after, proportion);
  MITK_TEST_CONDITION (isValid, "GetBoundingTimeStamps(): Exact match last in list, expecting true==true, and got:" << isValid);
  MITK_TEST_CONDITION (before == 8, "GetBoundingTimeStamps(): Exact match last in list, expecting before==8, and got:" << before);
  MITK_TEST_CONDITION (after == 8, "GetBoundingTimeStamps(): Exact match last in list, expecting after==8, and got:" << after);
  MITK_TEST_CONDITION (proportion == 0.0, "GetBoundingTimeStamps(): Exact match last in list, expecting proportion==0.0, and got:" << proportion);




  // Test GetNearestTimeStamp();
  timeStamps.Clear();
  delta = -1;
  result = timeStamps.GetNearestTimeStamp(1, NULL);
  MITK_TEST_CONDITION (result == 0, "GetNearestTimeStamp(): Empty list, expecting result==0, and got:" << result);
  MITK_TEST_CONDITION (delta == -1, "GetNearestTimeStamp(): Empty list, expecting delta==-1, as it was not passed, and got:" << delta);
  result = timeStamps.GetNearestTimeStamp(2, &delta);
  MITK_TEST_CONDITION (result == 0, "GetNearestTimeStamp(): Empty list, expecting result==0, and got:" << result);
  MITK_TEST_CONDITION (delta == 0, "GetNearestTimeStamp(): Empty list, expecting delta==0, as it was passed, and got:" << delta);
  timeStamps.Insert(3);
  timeStamps.Insert(6);
  timeStamps.Insert(9);
  timeStamps.Insert(12);
  result = timeStamps.GetNearestTimeStamp(15, &delta);
  MITK_TEST_CONDITION (result == 12, "GetNearestTimeStamp(): Off top end of list, expecting result==12, and got:" << result);
  MITK_TEST_CONDITION (delta == 3, "GetNearestTimeStamp(): Off top end of list, expecting delta==3, and got:" << delta);
  result = timeStamps.GetNearestTimeStamp(1, &delta);
  MITK_TEST_CONDITION (result == 3, "GetNearestTimeStamp(): Off bottom end of list, expecting result==3, and got:" << result);
  MITK_TEST_CONDITION (delta == -2, "GetNearestTimeStamp(): Off bottom end of list, expecting delta==-2, and got:" << delta);
  result = timeStamps.GetNearestTimeStamp(7, &delta);
  MITK_TEST_CONDITION (result == 6, "GetNearestTimeStamp(): Middle of list, expecting result==6, and got:" << result);
  MITK_TEST_CONDITION (delta == 1, "GetNearestTimeStamp(): Middle of list, expecting delta==1, and got:" << delta);
  result = timeStamps.GetNearestTimeStamp(8, &delta);
  MITK_TEST_CONDITION (result == 9, "GetNearestTimeStamp(): Middle of list, expecting result==9, and got:" << result);
  MITK_TEST_CONDITION (delta == -1, "GetNearestTimeStamp(): Middle of list, expecting delta==-1, and got:" << delta);

  MITK_TEST_END();
}


