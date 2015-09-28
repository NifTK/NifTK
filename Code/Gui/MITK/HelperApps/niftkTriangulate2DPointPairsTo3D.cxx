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

#include <mitkTriangulate2DPointPairsTo3D.h>
#include <niftkTriangulate2DPointPairsTo3DCLP.h>

#include <boost/lexical_cast.hpp>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( inputPointPairs.length() == 0
  || intrinsicLeft.length() == 0
  || intrinsicRight.length() == 0
  || rightToLeftExtrinsics.length() == 0
  || outputPoints.length() == 0
  )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::Triangulate2DPointPairsTo3D::Pointer triangulator = mitk::Triangulate2DPointPairsTo3D::New();
    triangulator->SetInput2DPointPairsFileName(inputPointPairs);
    triangulator->SetIntrinsicLeftFileName(intrinsicLeft);
    triangulator->SetIntrinsicRightFileName(intrinsicRight);
    triangulator->SetRightToLeftExtrinsics(rightToLeftExtrinsics);
    triangulator->SetOutputFileName(outputPoints);
    
    if ( leftMask.length() != 0 ) 
    {
      triangulator->SetLeftMaskFileName(leftMask);
    }

    if ( rightMask.length() != 0 ) 
    {
      triangulator->SetRightMaskFileName(rightMask);
    }
    triangulator->SetTrackingMatrixFileName(trackerToWorld);
    triangulator->SetHandeyeMatrixFileName(leftLensToTracker);

    if ( outputMaskImagePrefix.length() != 0 )
    {
      triangulator->SetOutputMaskImagePrefix(outputMaskImagePrefix);
    }

    triangulator->SetUndistortBeforeTriangulation(undistort);

    double minDistanceFromLens = - ( std::numeric_limits<double>::infinity() );
    double maxDistanceFromLens =  std::numeric_limits<double>::infinity();

    if ( minimumDistanceFromLens.length () != 0 )
    {
      minDistanceFromLens = boost::lexical_cast<double>(minimumDistanceFromLens);

      MITK_INFO << "Culling points closer than " << minDistanceFromLens;
    }
    if ( maximumDistanceFromLens.length () != 0 )
    {
      maxDistanceFromLens = boost::lexical_cast<double>(maximumDistanceFromLens);
      MITK_INFO << "Culling points further than " << maxDistanceFromLens;
    }
    triangulator->SetMinimumDistanceFromLens( minDistanceFromLens );
    triangulator->SetMaximumDistanceFromLens( maxDistanceFromLens );

    triangulator->Triangulate();

    returnStatus = EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = -2;
  }

  return returnStatus;
}
