/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkQImageDataSourceService.h"
#include <niftkQImageDataType.h>
#include <niftkQImageConversion.h>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
QImageDataSourceService::QImageDataSourceService(
    QString deviceName,
    QString factoryName,
    unsigned bufferSize,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: SingleFrameDataSourceService(deviceName, factoryName, bufferSize, properties, dataStorage)
{
  this->SetStatus("Initialising");
  this->SetDescription("QImageDataSourceService");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
QImageDataSourceService::~QImageDataSourceService()
{
}


//-----------------------------------------------------------------------------
void QImageDataSourceService::SaveImage(const std::string& filename,
                                        niftk::IGIDataType& data)
{
  niftk::QImageDataType* dataType = dynamic_cast<niftk::QImageDataType*>(&data);
  if (dataType == nullptr)
  {
    mitkThrow() << "Failed to save QImageDataType as the data received was the wrong type!";
  }

  const QImage* imageFrame = dataType->GetImage();
  if (imageFrame == nullptr)
  {
    mitkThrow() << "Failed to save QImageDataType as the image frame was NULL!";
  }

  bool success = imageFrame->save(QString::fromStdString(filename));
  if (!success)
  {
    mitkThrow() << "Failed to save QImageDataType to file:" << filename;
  }

  data.SetIsSaved(true);
}


//-----------------------------------------------------------------------------
std::unique_ptr<niftk::IGIDataType> QImageDataSourceService::LoadImage(const std::string& filename)
{
  QImage *image = new QImage();
  bool success = image->load(QString::fromStdString(filename));
  if (!success)
  {
    mitkThrow() << "Failed to load image:" << filename;
  }

  std::unique_ptr<niftk::IGIDataType> result(new niftk::QImageDataType(image));
  return result;
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer QImageDataSourceService::RetrieveImage(const niftk::IGIDataSourceI::IGITimeType& requestedTime,
                                                            niftk::IGIDataSourceI::IGITimeType& actualTime,
                                                            unsigned int& outputNumberOfBytes)
{
  bool gotFromBuffer = m_Buffer.CopyOutItem(requestedTime, m_CachedImage);
  if (!gotFromBuffer)
  {
    MITK_INFO << "QImageDataSourceService: Failed to find data for time:" << requestedTime;
    return nullptr;
  }

  const QImage* img = m_CachedImage.GetImage();
  if (img == nullptr)
  {
    this->SetStatus("Failed");
    mitkThrow() << "Failed to extract QImage!";
  }

  mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(img, outputNumberOfBytes);
  return convertedImage;
}

} // end namespace
