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
#include <mitkCameraCalibrationFacade.h>


/**
 * This test attempts to answer the question of how important errors in the 
 * hand eye calibration are.
 * Start with a point in world space. p
 * We estimate it's position (p est) in world space using it's measured position 
 * in screen coordinates and T screen to world. Which is 
 * TScreenToLens (intrinsic) * TLensToTracker (handeye) * TTrackerToWorld
 * Shouldn't need intrinsic, we estimate it's position in world space
 * using it's measured position in lens coordinates.
 * The estimated position may be very wrong due to the handeye error, 
 * but we are only interested in the sunsequent error in it's position 
 * relative to the lens.
 * so we now have two p1, (relative to the lens) the actual and the estimated.
 * p1 actual =  p * WorldToTracker * handeye-1 
 * and p1 (est) = p (est) * WorldToTracker * handeye (est) -1
 * and error = p1 actual - p1 (est) - or vice versa 
 *
 * The purpose of this program is to determine the error for a set of
 * increasingly erroreous handeyes, to try and show how handeye is related to 
 * projection error in this case
 * 
 */

cv::Mat WorldToLens (cv::Mat PointInWorldCoordinates, cv::Mat TrackerToWorld,
    cv::Mat TrackerToLens);

cv::Mat LensToWorld (cv::Mat PointInLensCoordinates, cv::Mat TrackerToWorld,
    cv::Mat TrackerToLens);


int mitkHandeyeSensitivityTest ( int argc, char * argv[] )
{
  return 0;
}
