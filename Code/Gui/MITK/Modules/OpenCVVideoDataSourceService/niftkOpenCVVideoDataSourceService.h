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
#include <niftkIGIDataSource.h>
#include <niftkIGIDataSourceBuffer.h>
#include <niftkIGIDataSourceBackgroundDeleteThread.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include <niftkIGILocalDataSourceI.h>

#include <mitkOpenCVVideoSource.h>

#include <QObject>
#include <QSet>
#include <QMutex>
#include <QObject>

#include <string>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceService
* \brief Provides an OpenCV video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKOPENCVVIDEODATASOURCESERVICE_EXPORT OpenCVVideoDataSourceService
    : public IGIDataSource
    , public IGILocalDataSourceI
    , public QObject
{

public:

  mitkClassMacroItkParent(OpenCVVideoDataSourceService, IGIDataSource);
  mitkNewMacro1Param(OpenCVVideoDataSourceService, mitk::DataStorage::Pointer);

  /**
  * \see IGIDataSourceI::StartCapturing()
  */
  virtual void StartCapturing() override;

  /**
  * \see IGIDataSourceI::StopCapturing()
  */
  virtual void StopCapturing() override;

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(const std::string& path,
                                  niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                  niftk::IGIDataType::IGITimeType* lastTimeStampInStore);

  /**
  * \see IGIDataSourceI::SetLagInMilliseconds()
  */
  virtual void SetLagInMilliseconds(const niftk::IGIDataType::IGITimeType& milliseconds) override;

  /**
  * \see IGIDataSourceI::GetSaveDirectoryName()
  */
  virtual std::string GetSaveDirectoryName() override;

  /**
  * \see IGIDataSourceI::Update()
  */
  virtual std::vector<IGIDataItemInfo> Update(const niftk::IGIDataType::IGITimeType& time) override;

  /**
  * \see niftk::IGIDataSource::SaveItem()
  */
  virtual void SaveItem(niftk::IGIDataType::Pointer item) override;

  /**
  * \see niftk::IGIDataSource::CleanBuffer()
  */
  virtual void CleanBuffer() override;

  /**
  * \see niftk::IGILocalDataSourceI::GrabData()
  */
  virtual void GrabData() override;

protected:

  OpenCVVideoDataSourceService(mitk::DataStorage::Pointer dataStorage);
  virtual ~OpenCVVideoDataSourceService();

private:

  OpenCVVideoDataSourceService(const OpenCVVideoDataSourceService&); // deliberately not implemented
  OpenCVVideoDataSourceService& operator=(const OpenCVVideoDataSourceService&); // deliberately not implemented

  static int GetNextChannelNumber();

  mitk::OpenCVVideoSource::Pointer                m_VideoSource;
  int                                             m_ChannelNumber;
  static QMutex                                   s_Lock;
  static QSet<int>                                s_SourcesInUse;
  niftk::IGIDataType::IGIIndexType                m_FrameId;
  niftk::IGIDataSourceBuffer::Pointer             m_Buffer;
  niftk::IGIDataSourceBackgroundDeleteThread*     m_BackgroundDeleteThread;
  niftk::IGIDataSourceGrabbingThread*             m_DataGrabbingThread;

}; // end class

} // end namespace

#endif
