/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkOpenCVVideoDataSourceService.h"

namespace niftk
{

//-----------------------------------------------------------------------------
QMutex    OpenCVVideoDataSourceService::s_Lock(QMutex::Recursive);
QSet<int> OpenCVVideoDataSourceService::s_SourcesInUse;

//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::OpenCVVideoDataSourceService()
{

}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::~OpenCVVideoDataSourceService()
{

}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StartCapture()
{

}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StopCapture()
{

}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StartRecording()
{

}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StopRecording()
{

}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SetLagInNanoSeconds(const unsigned long long& nanoseconds)
{

}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SetRecordingLocation(const std::string& pathName)
{

}

} // end namespace
