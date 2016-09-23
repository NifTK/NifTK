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

#include <niftkQImageDataSourceService.h>

class QCamera;
class CameraFrameGrabber;

namespace niftk
{

/**
* \class QtCameraVideoDataSourceService
* \brief Provides a QtCamera video feed, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class QtCameraVideoDataSourceService : public QImageDataSourceService
{

  Q_OBJECT

public:

  mitkClassMacroItkParent(QtCameraVideoDataSourceService,
                          QImageDataSourceService)

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
   * \see niftk::SingleFrameDataSourceService::GrabImage().
   */
  virtual niftk::IGIDataType::Pointer GrabImage() override;

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
