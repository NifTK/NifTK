/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkQImageToMitkImageFilter.h"

#include <itkImportImageFilter.h>
#include <itkRGBToLuminanceImageFilter.h>
#include <mitkITKImageImport.txx>

//-----------------------------------------------------------------------------
QmitkQImageToMitkImageFilter::QmitkQImageToMitkImageFilter()
: m_QImage(0), m_Image(0)
{
}


//-----------------------------------------------------------------------------
QmitkQImageToMitkImageFilter::~QmitkQImageToMitkImageFilter()
{
}


//-----------------------------------------------------------------------------
void QmitkQImageToMitkImageFilter::SetQImage(const QImage* image)
{
  this->m_QImage = image;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::ImageSource::DataObjectPointer QmitkQImageToMitkImageFilter::MakeOutput( unsigned int idx )
{
  return Superclass::MakeOutput(idx);
}


//-----------------------------------------------------------------------------
mitk::ImageSource::OutputImageType* QmitkQImageToMitkImageFilter::GetOutput()
{
  return m_Image;
}


//-----------------------------------------------------------------------------
void QmitkQImageToMitkImageFilter::GenerateData()
{
  if(m_QImage == 0)
  {
    MITK_WARN << "Cannot not start filter. QImage not set.";
    return;
  }

  if (m_QImage->format() == QImage::Format_RGB888)
  {
    m_Image = ConvertQImageToMitkImage< UCRGBPixelType, 3>( m_QImage );
  }
  else
  {
		if ( m_QImage->format() == QImage::Format_Indexed8 )
      m_Image = Convert8BitQImageToMitkImage < unsigned char, 3>(m_QImage);
		else
		{
			QImage tmpImage = m_QImage->convertToFormat(QImage::Format_RGB888);
      m_Image = ConvertQImageToMitkImage< UCRGBPixelType, 3>( &tmpImage );
		}
  }
}


//-----------------------------------------------------------------------------
template <typename TPixel, unsigned int VImageDimension>
mitk::Image::Pointer QmitkQImageToMitkImageFilter::ConvertQImageToMitkImage( const QImage* input)
{

  typedef itk::Image< TPixel, VImageDimension > ItkImage;
  typedef itk::Image <unsigned char, VImageDimension> OutputItkImage;
  typedef itk::ImportImageFilter< TPixel, VImageDimension >  ImportFilterType;

  mitk::Image::Pointer mitkImage = mitk::Image::New();

  typename ImportFilterType::Pointer importFilter = ImportFilterType::New();
  typename ImportFilterType::SizeType  size;

  size[0]  = input->width();
  size[1]  = input->height();
	size[2]  = 1;

  typename ImportFilterType::IndexType start;
  start.Fill( 0 );

  typename ImportFilterType::RegionType region;
  region.SetIndex( start );
  region.SetSize(  size  );

  importFilter->SetRegion( region );

  double origin[ VImageDimension ];
  origin[0] = 0.0;    // X coordinate
  origin[1] = 0.0;    // Y coordinate
	origin[2] = 0.0;    // Z coordinate

  importFilter->SetOrigin( origin );

  double spacing[ VImageDimension ];
  spacing[0] = 1.0;    // along X direction
  spacing[1] = 1.0;    // along Y direction
  spacing[2] = 1.0;    // along Z direction

  importFilter->SetSpacing( spacing );

  const unsigned int numberOfPixels = size[0] * size[1] * size[2];
  const unsigned int numberOfBytes = numberOfPixels * sizeof( TPixel );

  TPixel * localBuffer = new TPixel[numberOfPixels];
  memcpy(localBuffer, input->bits(), numberOfBytes);

  importFilter->SetImportPointer( localBuffer, numberOfPixels, false);
  importFilter->Update();


  typename itk::RGBToLuminanceImageFilter<ItkImage, OutputItkImage>::Pointer converter = itk::RGBToLuminanceImageFilter<ItkImage, OutputItkImage>::New();
  converter->SetInput(importFilter->GetOutput());
  converter->Update();

  typename OutputItkImage::Pointer output = converter->GetOutput();
  output->DisconnectPipeline();

  mitkImage = mitk::ImportItkImage( output );


  return mitkImage;
}
	
template <typename TPixel, unsigned int VImageDimension>
mitk::Image::Pointer QmitkQImageToMitkImageFilter::Convert8BitQImageToMitkImage( const QImage* input)
{

  typedef itk::Image< TPixel, VImageDimension > ItkImage;
  typedef itk::Image <unsigned char, VImageDimension> OutputItkImage;
  typedef itk::ImportImageFilter< TPixel, VImageDimension >  ImportFilterType;

  mitk::Image::Pointer mitkImage = mitk::Image::New();

  typename ImportFilterType::Pointer importFilter = ImportFilterType::New();
  typename ImportFilterType::SizeType  size;

  size[0]  = input->width();
  size[1]  = input->height();
	size[2]  = 1;

  typename ImportFilterType::IndexType start;
  start.Fill( 0 );

  typename ImportFilterType::RegionType region;
  region.SetIndex( start );
  region.SetSize(  size  );

  importFilter->SetRegion( region );

  double origin[ VImageDimension ];
  origin[0] = 0.0;    // X coordinate
  origin[1] = 0.0;    // Y coordinate
  origin[2] = 0.0;    // Z coordinate

  importFilter->SetOrigin( origin );

  double spacing[ VImageDimension ];
  spacing[0] = 1.0;    // along X direction
  spacing[1] = 1.0;    // along Y direction
  spacing[2] = 1.0;    // along Z direction

  importFilter->SetSpacing( spacing );

  const unsigned int numberOfPixels = size[0] * size[1] * size[2];
  const unsigned int numberOfBytes = numberOfPixels * sizeof( TPixel );

  TPixel * localBuffer = new TPixel[numberOfPixels];
  memcpy(localBuffer, input->bits(), numberOfBytes);

  importFilter->SetImportPointer( localBuffer, numberOfPixels, false);
  importFilter->Update();
/*	if ( image.format() == QImage::Indexed_8)
	{
		QVector<QRgb> colors=QVector<QRgb> (256);
		for ( int i = 0 ; i < 256 ; i ++)
		  colors[i] = qRgb(i,i,i);
			image.setColorTable(colors);
	}*/

  typename OutputItkImage::Pointer output = importFilter->GetOutput();
  output->DisconnectPipeline();

	mitkImage = mitk::ImportItkImage( output );
	
  return mitkImage;
}


