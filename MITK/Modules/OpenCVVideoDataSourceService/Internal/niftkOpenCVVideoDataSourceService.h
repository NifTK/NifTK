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

#include <niftkSingleFrameDataSourceService.h>
#include <niftkIGIDataSourceGrabbingThread.h>
#include "niftkOpenCVVideoDataType.h"

#include <mitkOpenCVVideoSource.h>
#include <niftkOpenCVVideoDataSourceServiceExports.h>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceService
* \brief Provides an OpenCV video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should be thrown as mitk::Exception or sub-classes thereof.
*/
class OpenCVVideoDataSourceService : public SingleFrameDataSourceService
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(OpenCVVideoDataSourceService, SingleFrameDataSourceService)
  mitkNewMacro3Param(OpenCVVideoDataSourceService, QString, const IGIDataSourceProperties&, mitk::DataStorage::Pointer)

protected:

  OpenCVVideoDataSourceService(QString factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                               );
  virtual ~OpenCVVideoDataSourceService();

  /**
   * \see niftk::SingleFrameDataSourceService::GrabImage().
   */
  virtual std::unique_ptr<niftk::IGIDataType> GrabImage() override;

  /**
   * \see niftk::SingleFrameDataSourceService::RetrieveImage()
   */
  virtual mitk::Image::Pointer RetrieveImage(const niftk::IGIDataSourceI::IGITimeType& requested,
                                             niftk::IGIDataSourceI::IGITimeType& actualTime,
                                             unsigned int& outputNumberOfBytes) override;

  /**
   * \see niftk::SingleFrameDataSourceService::SaveImage().
   */
  virtual void SaveImage(const std::string& filename, niftk::IGIDataType& item) override;

  /**
   * \see niftk::SingleFrameDataSourceService::LoadImage().
   */
  virtual std::unique_ptr<niftk::IGIDataType> LoadImage(const std::string& filename) override;

private slots:

  void OnErrorFromThread(QString);

private:

  OpenCVVideoDataSourceService(const OpenCVVideoDataSourceService&); // deliberately not implemented
  OpenCVVideoDataSourceService& operator=(const OpenCVVideoDataSourceService&); // deliberately not implemented

  mitk::OpenCVVideoSource::Pointer    m_VideoSource;
  niftk::IGIDataSourceGrabbingThread* m_DataGrabbingThread;
  niftk::OpenCVVideoDataType          m_CachedImage;

}; // end class

} // end namespace

#endif
