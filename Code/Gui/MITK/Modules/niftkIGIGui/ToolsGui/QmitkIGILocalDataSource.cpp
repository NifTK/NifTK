/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGILocalDataSource.h"
#include "mitkITKImageImport.h"

//-----------------------------------------------------------------------------
QmitkIGILocalDataSource::QmitkIGILocalDataSource()
{
}


//-----------------------------------------------------------------------------
QmitkIGILocalDataSource::~QmitkIGILocalDataSource()
{
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer QmitkIGILocalDataSource::CreateMitkImage(const IplImage* image) const
{
  ImportFilterType::Pointer importFilter = ImportFilterType::New();
  mitk::ITKImageImport<ItkImage>::Pointer mitkFilter = mitk::ITKImageImport<ItkImage>::New();

  ImportFilterType::SizeType size;
  size[0] = image->width;
  size[1] = image->height;
  size[2] = 1;

  ImportFilterType::IndexType start;
  start.Fill( 0 );

  ImportFilterType::RegionType region;
  region.SetIndex( start );
  region.SetSize(  size  );

  double origin[ 3 ];
  origin[0] = 0.0;    // X coordinate
  origin[1] = 0.0;    // Y coordinate
  origin[2] = 0.0;    // Z coordinate

  double spacing[ 3 ];
  spacing[0] = 1.0;    // along X direction
  spacing[1] = 1.0;    // along Y direction
  spacing[2] = 1.0;    // along Z direction

  const unsigned int numberOfPixels = size[0] * size[1];

  importFilter->SetRegion( region );
  importFilter->SetOrigin( origin );
  importFilter->SetSpacing( spacing );

  // This creates a new buffer, and copies this image into it.
  // Without this, if you use the OpenCV buffer directly, you can see
  // artefacts as the framebuffer is written to.
  const unsigned int numberOfBytes = numberOfPixels * sizeof( UCRGBPixelType );
  UCRGBPixelType* localBuffer = new UCRGBPixelType[numberOfPixels];
  memcpy(localBuffer, image->imageData, numberOfBytes);

  // This will tell the importFilter to use the supplied buffer,
  // and the output of this filter, references the same buffer.
  // We don't let the import filter take control of the buffer
  // because if you do, the importFilter destructor will destroy the
  // buffer when it goes out of scope.  We want the output image to
  // remain after the filter is out of scope.
  bool importFilterWillTakeControlOfBuffer = false;
  importFilter->SetImportPointer( localBuffer, numberOfPixels, importFilterWillTakeControlOfBuffer);
  importFilter->Update();

  // We then need a stand-alone ITK image, that survives after a pipeline.
  ItkImage::Pointer itkOutput = importFilter->GetOutput();
  itkOutput->DisconnectPipeline();

  // We then convert it to MITK, and this conversion takes responsibility for the memory.
  mitk::Image::Pointer mitkImage = mitk::ImportItkImage(itkOutput);

  // Delete local buffer, as even though the output object is disconnected, it does not manage data.
  delete [] localBuffer;

  return mitkImage;
}
