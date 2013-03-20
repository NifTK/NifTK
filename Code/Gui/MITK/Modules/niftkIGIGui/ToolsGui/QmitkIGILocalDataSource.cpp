/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGILocalDataSource.h"
#include "QmitkIGILocalDataSourceGrabbingThread.h"
#include "mitkITKImageImport.h"

//-----------------------------------------------------------------------------
QmitkIGILocalDataSource::QmitkIGILocalDataSource()
: m_GrabbingThread(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkIGILocalDataSource::~QmitkIGILocalDataSource()
{
  if (m_GrabbingThread != NULL)
  {
    delete m_GrabbingThread;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGILocalDataSource::InitializeAndRunGrabbingThread(const int& intervalInMilliseconds)
{
  // Only do this once, as m_GrabbingThread initialised to NULL in constructor.
  if (m_GrabbingThread == NULL)
  {
    m_GrabbingThread = new QmitkIGILocalDataSourceGrabbingThread(this, this);
    m_GrabbingThread->SetInterval(intervalInMilliseconds);
    m_GrabbingThread->start();
  }
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer QmitkIGILocalDataSource::CreateMitkImage(const IplImage* image) const
{
  // the buffer copy below breaks otherwise
  assert(image->nChannels == 3);

  ImportFilterType::Pointer importFilter = ImportFilterType::New();
  mitk::ITKImageImport<ItkImage>::Pointer mitkFilter = mitk::ITKImageImport<ItkImage>::New();

  ImportFilterType::SizeType size;
  size[0] = image->width;
  size[1] = image->height;

  ImportFilterType::IndexType start;
  start.Fill( 0 );

  ImportFilterType::RegionType region;
  region.SetIndex( start );
  region.SetSize(  size  );

  double origin[ 2 ];
  origin[0] = 0.0;    // X coordinate
  origin[1] = 0.0;    // Y coordinate

  double spacing[ 2 ];
  spacing[0] = 1.0;    // along X direction
  spacing[1] = 1.0;    // along Y direction

  const unsigned int numberOfPixels = size[0] * size[1];

  importFilter->SetRegion( region );
  importFilter->SetOrigin( origin );
  importFilter->SetSpacing( spacing );

  // This creates a new buffer, and copies this image into it.
  // Without this, if you use the OpenCV buffer directly, you can see
  // artefacts as the framebuffer is written to.
  const unsigned int numberOfBytes = numberOfPixels * sizeof( UCRGBPixelType );
  UCRGBPixelType* localBuffer = new UCRGBPixelType[numberOfPixels];
  // if the image pitch is the same as its width then everything is peachy
  //  but if not we need to take care of that
  const unsigned int numberOfBytesPerLine = image->width * image->nChannels;
  if (numberOfBytesPerLine == static_cast<unsigned int>(image->widthStep))
  {
    memcpy(localBuffer, image->imageData, numberOfBytes);
  }
  else
  {
    // if that is not true then something is seriously borked
    assert(image->widthStep >= numberOfBytesPerLine);

    // "slow" path: copy line by line
    for (int y = 0; y < image->height; ++y)
    {
      // widthStep is in bytes while width is in pixels
      std::memcpy(&(((char*) localBuffer)[y * numberOfBytesPerLine]), &(image->imageData[y * image->widthStep]), numberOfBytesPerLine); 
    }
  }

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
  mitk::Image::Pointer mitkImage;
  mitk::CastToMitkImage(itkOutput, mitkImage);

  // Delete local buffer, as even though the output object is disconnected, it does not manage data.
  delete [] localBuffer;

  return mitkImage;
}
