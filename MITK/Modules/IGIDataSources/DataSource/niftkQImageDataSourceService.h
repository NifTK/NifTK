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
                          const IGIDataSourceProperties& properties,
                          mitk::DataStorage::Pointer dataStorage
                         );
  virtual ~QImageDataSourceService();

  /**
   * \brief Derived classes implement this to grab a new image.
   */
  virtual niftk::IGIDataType::Pointer GrabImage() override = 0;

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

  QImageDataSourceService(const QImageDataSourceService&); // deliberately not implemented
  QImageDataSourceService& operator=(const QImageDataSourceService&); // deliberately not implemented

}; // end class

} // end namespace

#endif
