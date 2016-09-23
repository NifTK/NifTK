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
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: SingleFrameDataSourceService(deviceName, factoryName, properties, dataStorage)
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
                                        niftk::IGIDataType::Pointer data)
{
  niftk::QImageDataType::Pointer dataType = static_cast<niftk::QImageDataType*>(data.GetPointer());
  if (dataType.IsNull())
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

  data->SetIsSaved(true);
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::Pointer QImageDataSourceService::LoadImage(const std::string& filename)
{
  QImage image;
  bool success = image.load(QString::fromStdString(filename));
  if (!success)
  {
    mitkThrow() << "Failed to load image:" << filename;
  }

  niftk::QImageDataType::Pointer wrapper = niftk::QImageDataType::New();
  wrapper->ShallowCopy(image);

  return wrapper.GetPointer();
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer QImageDataSourceService::ConvertImage(niftk::IGIDataType::Pointer inputImage,
                                                           unsigned int& outputNumberOfBytes)
{
  niftk::QImageDataType::Pointer dataType = static_cast<niftk::QImageDataType*>(inputImage.GetPointer());
  if (dataType.IsNull())
  {
    this->SetStatus("Failed");
    mitkThrow() << "Failed to extract QImageDataType!";
  }

  const QImage* img = dataType->GetImage();
  if (img == nullptr)
  {
    this->SetStatus("Failed");
    mitkThrow() << "Failed to extract QImage!";
  }

  mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(img, outputNumberOfBytes);
  return convertedImage;
}

} // end namespace
