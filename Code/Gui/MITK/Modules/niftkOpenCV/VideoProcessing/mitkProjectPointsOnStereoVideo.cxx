/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkProjectPointsOnStereoVideo.h"
#include <cv.h>
#include <highgui.h>

namespace mitk {

//-----------------------------------------------------------------------------
ProjectPointsOnStereoVideo::ProjectPointsOnStereoVideo()
: m_Visualise(false)
, m_SaveVideo(false)
, m_VideoIn("")
, m_VideoOut("")
, m_TrackerIndex(0)
, m_TrackerMatcher(NULL)
, m_DrawLines(false)
, m_InitOK(false)
, m_ProjectOK(false)
, leftIntrinsicMatrix (cvCreateMat(3,3,CV_32FC1))
, leftDistortionVector (cvCreateMat(5,1,CV_32FC1))
, rightIntrinsicMatrix (cvCreateMat(3,3,CV_32FC1))
, rightDistortionVector (cvCreateMat(5,1,CV_32FC1))
, rightToLeftRotationMatrix (cvCreateMat(3,3,CV_32FC1))
, rightToLeftTranslationVector (cvCreateMat(3,1,CV_32FC1))
, leftHandeye (cvCreateMat(4,4,CV_32FC1))
{
}


//-----------------------------------------------------------------------------
ProjectPointsOnStereoVideo::~ProjectPointsOnStereoVideo()
{

}


//-----------------------------------------------------------------------------
void ProjectPointsOnStereoVideo::Project()
{
}

} // end namespace
