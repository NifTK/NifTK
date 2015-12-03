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
  mitkNewMacro3Param(OpenCVVideoDataSourceService, std::string, const IGIDataSourceProperties&, mitk::DataStorage::Pointer);

  /**
  * \see IGIDataSourceI::StartCapturing()
  */
  virtual void StartCapturing() override;

  /**
  * \see IGIDataSourceI::StopCapturing()
  */
  virtual void StopCapturing() override;

  /**
  * \see  IGIDataSourceI::StartPlayback()
  */
  virtual void StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                             niftk::IGIDataType::IGITimeType lastTimeStamp) override;

  /**
  * \see IGIDataSourceI::PlaybackData()
  */
  void PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp) override;

  /**
  * \see IGIDataSourceI::StopPlayback()
  */
  virtual void StopPlayback() override;

  /**
  * \see IGIDataSourceI::SetLagInMilliseconds()
  */
  virtual void SetLagInMilliseconds(const niftk::IGIDataType::IGITimeType& milliseconds) override;

  /**
  * \see IGIDataSourceI::GetRecordingDirectoryName()
  */
  virtual std::string GetRecordingDirectoryName() override;

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

  /**
  * \see IGIDataSourceI::ProbeRecordedData()
  */
  bool ProbeRecordedData(const std::string& path,
                         niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                         niftk::IGIDataType::IGITimeType* lastTimeStampInStore) override;

  /**
  * \brief IGIDataSourceI::SetProperties()
  */
  virtual void SetProperties(const IGIDataSourceProperties& properties) override;

  /**
  * \brief IGIDataSourceI::GetProperties()
  */
  virtual IGIDataSourceProperties GetProperties() const override;

protected:

  OpenCVVideoDataSourceService(std::string factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                               );
  virtual ~OpenCVVideoDataSourceService();

private:

  OpenCVVideoDataSourceService(const OpenCVVideoDataSourceService&); // deliberately not implemented
  OpenCVVideoDataSourceService& operator=(const OpenCVVideoDataSourceService&); // deliberately not implemented

  static int GetNextChannelNumber();

  static QMutex                                   s_Lock;
  static QSet<int>                                s_SourcesInUse;

  QMutex                                          m_Lock;
  mitk::OpenCVVideoSource::Pointer                m_VideoSource;
  int                                             m_ChannelNumber;
  niftk::IGIDataType::IGIIndexType                m_FrameId;
  niftk::IGIDataSourceBuffer::Pointer             m_Buffer;
  niftk::IGIDataSourceBackgroundDeleteThread*     m_BackgroundDeleteThread;
  niftk::IGIDataSourceGrabbingThread*             m_DataGrabbingThread;
  std::set<niftk::IGIDataType::IGITimeType>       m_PlaybackIndex;

}; // end class

} // end namespace

#endif
