/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkQtCameraVideoDataSourceService_h
#define niftkQtCameraVideoDataSourceService_h

#include <niftkSingleVideoFrameDataSourceService.h>

class QCamera;
class CameraFrameGrabber;

namespace niftk
{

/**
* \class QtCameraVideoDataSourceService
* \brief Provides an QtCamera video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class QtCameraVideoDataSourceService : public SingleVideoFrameDataSourceService
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(QtCameraVideoDataSourceService,
                          SingleVideoFrameDataSourceService)

  mitkNewMacro3Param(QtCameraVideoDataSourceService, QString,
                     const IGIDataSourceProperties&,
                     mitk::DataStorage::Pointer)

protected:

  QtCameraVideoDataSourceService(QString factoryName,
                               const IGIDataSourceProperties& properties,
                               mitk::DataStorage::Pointer dataStorage
                               );
  virtual ~QtCameraVideoDataSourceService();

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

private slots:

  void OnFrameAvailable(const QImage &image);

private:

  QtCameraVideoDataSourceService(const QtCameraVideoDataSourceService&); // deliberately not implemented
  QtCameraVideoDataSourceService& operator=(const QtCameraVideoDataSourceService&); // deliberately not implemented

  QCamera*             m_Camera;
  CameraFrameGrabber*  m_CameraFrameGrabber;
  mutable QImage*      m_TemporaryWrapper;
}; // end class

} // end namespace

#endif
