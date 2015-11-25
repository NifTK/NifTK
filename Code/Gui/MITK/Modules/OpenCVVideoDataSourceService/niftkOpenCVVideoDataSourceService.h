/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkOpenCVVideoDataSourceService_h
#define niftkOpenCVVideoDataSourceService_h

#include "niftkOpenCVVideoDataSourceServiceExports.h"
#include <string>
#include <mitkOpenCVVideoSource.h>

#include <QObject>
#include <QSet>
#include <QMutex>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceService
* \brief Provides a video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKOPENCVVIDEODATASOURCESERVICE_EXPORT OpenCVVideoDataSourceService
{

public:

  virtual void StartCapture();
  virtual void StopCapture();
  virtual void StartRecording();
  virtual void StopRecording();
  virtual void SetLagInNanoSeconds(const unsigned long long& nanoseconds);
  virtual void SetRecordingLocation(const std::string& pathName);

protected:
  OpenCVVideoDataSourceService();
  virtual ~OpenCVVideoDataSourceService();

private:
  OpenCVVideoDataSourceService(const OpenCVVideoDataSourceService&); // deliberately not implemented
  OpenCVVideoDataSourceService& operator=(const OpenCVVideoDataSourceService&); // deliberately not implemented

  mitk::OpenCVVideoSource::Pointer  m_VideoSource;
  int                               m_ChannelNumber;
  static QMutex                     s_Lock;
  static QSet<int>                  s_SourcesInUse;

}; // end class

} // end namespace

#endif
