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

#include <niftkSingleVideoFrameDataSourceService.h>

#include <niftkIGIDataSourceGrabbingThread.h>
#include <mitkOpenCVVideoSource.h>

namespace niftk
{

/**
* \class OpenCVVideoDataSourceService
* \brief Provides an OpenCV video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class OpenCVVideoDataSourceService : public SingleVideoFrameDataSourceService
{

public:

  mitkClassMacroItkParent(OpenCVVideoDataSourceService, SingleVideoFrameDataSourceService)
  mitkNewMacro3Param(OpenCVVideoDataSourceService, QString, const IGIDataSourceProperties&, mitk::DataStorage::Pointer)

protected:

  OpenCVVideoDataSourceService(QString factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                               );
  virtual ~OpenCVVideoDataSourceService();

  /**
   * \see niftk::SingleVideoFrameDataSourceService::GrabImage().
   */
  virtual niftk::IGIDataType::Pointer GrabImage() override;

  /**
   * \see niftk::SingleVideoFrameDataSourceService::SaveImage().
   */
  virtual void SaveImage(const std::string& filename, niftk::IGIDataType::Pointer item) override;

  /**
   * \see niftk::SingleVideoFrameDataSourceService::LoadImage().
   */
  virtual niftk::IGIDataType::Pointer LoadImage(const std::string& filename) override;

  /**
   * \see niftk::SingleVideoFrameDataSourceService::ConvertImage().
   */
  virtual mitk::Image::Pointer ConvertImage(niftk::IGIDataType::Pointer inputImage,
                                            unsigned int& outputNumberOfBytes) override;

private:

  OpenCVVideoDataSourceService(const OpenCVVideoDataSourceService&); // deliberately not implemented
  OpenCVVideoDataSourceService& operator=(const OpenCVVideoDataSourceService&); // deliberately not implemented

  mitk::OpenCVVideoSource::Pointer    m_VideoSource;
  niftk::IGIDataSourceGrabbingThread* m_DataGrabbingThread;

}; // end class

} // end namespace

#endif
