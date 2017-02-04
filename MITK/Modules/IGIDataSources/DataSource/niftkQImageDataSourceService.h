/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkQImageDataSourceService_h
#define niftkQImageDataSourceService_h

#include <niftkSingleFrameDataSourceService.h>
#include <niftkIGIDataSourcesExports.h>
#include "niftkQImageDataType.h"

namespace niftk
{

/**
* \class QImageDataSourceService
* \brief Base class for SingleFrameDataSourceService based on QImage.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT QImageDataSourceService : public SingleFrameDataSourceService
{

public:

  mitkClassMacroItkParent(QImageDataSourceService,
                          SingleFrameDataSourceService)

protected:

  QImageDataSourceService(QString deviceName,
                          QString factoryName,
                          unsigned int framesPerSecond,
                          unsigned int bufferSize,
                          const IGIDataSourceProperties& properties,
                          mitk::DataStorage::Pointer dataStorage
                         );
  virtual ~QImageDataSourceService();

  /**
   * \brief Derived classes implement this to grab a new image.
   */
  virtual std::unique_ptr<niftk::IGIDataType> GrabImage() override = 0;

  /**
   * \see niftk::SingleFrameDataSourceService::RetrieveImage()
   */
  virtual mitk::Image::Pointer RetrieveImage(const niftk::IGIDataSourceI::IGITimeType& requested,
                                             niftk::IGIDataSourceI::IGITimeType& actualTime,
                                             unsigned int& outputNumberOfBytes) override;

  /**
   * \see niftk::SingleVideoFrameDataSourceService::SaveImage().
   */
  virtual void SaveImage(const std::string& filename, niftk::IGIDataType& item) override;

  /**
   * \see niftk::SingleVideoFrameDataSourceService::LoadImage().
   */
  virtual std::unique_ptr<niftk::IGIDataType> LoadImage(const std::string& filename) override;

private:

  QImageDataSourceService(const QImageDataSourceService&); // deliberately not implemented
  QImageDataSourceService& operator=(const QImageDataSourceService&); // deliberately not implemented

  niftk::QImageDataType m_CachedImage;

}; // end class

} // end namespace

#endif
