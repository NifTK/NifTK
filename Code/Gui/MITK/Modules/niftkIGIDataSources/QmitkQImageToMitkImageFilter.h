/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkQImageToMitkImageFilter_h
#define QmitkQImageToMitkImageFilter_h

#include <mitkCommon.h>
#include <mitkImageSource.h>
#include <itkMacro.h>
#include <itkImage.h>
#include <itkRGBPixel.h>
#include <itkRGBAPixel.h>
#include <QImage>

#include "niftkIGIGuiExports.h"

/**
 * \class QmitkQImageToMitkImageFilter
 * \brief A basic interface to produce a 2D Mitk image from a 2D QImage.
 * Currently only supports a QImage with format
 */
class NIFTKIGIGUI_EXPORT QmitkQImageToMitkImageFilter : public mitk::ImageSource
{
  public:
    typedef itk::RGBPixel< unsigned char > UCRGBPixelType;
    typedef itk::RGBPixel< unsigned short > USRGBPixelType;
    typedef itk::RGBPixel< float > FloatRGBPixelType;
    typedef itk::RGBPixel< double > DoubleRGBPixelType;

    typedef itk::RGBAPixel< unsigned char > UCRGBAPixelType;
    typedef itk::RGBAPixel< unsigned short > USRGBAPixelType;
    typedef itk::RGBAPixel< float > FloatRGBAPixelType;
    typedef itk::RGBAPixel< double > DoubleRGBAPixelType;

    mitkClassMacro(QmitkQImageToMitkImageFilter, mitk::ImageSource);
    itkNewMacro(QmitkQImageToMitkImageFilter);

    void SetQImage(const QImage* image);
    itkGetMacro(QImage, const QImage*);

    OutputImageType* GetOutput();

		/**
		 * \brief If set the image geometry will be copied from Geometry Image
		 */
		void SetGeometryImage ( mitk::Image::Pointer GeomImage);

protected:

    QmitkQImageToMitkImageFilter(); // purposely hidden
    virtual ~QmitkQImageToMitkImageFilter();

    virtual DataObjectPointer MakeOutput(unsigned int idx);

    virtual void GenerateData();

protected:
    const QImage* m_QImage;
    mitk::Image::Pointer m_Image;
		mitk::Image::Pointer m_GeomImage;

private:

    template <typename TPixel, unsigned int VImageDimension>
    static mitk::Image::Pointer ConvertQImageToMitkImage( const QImage* input, const mitk::Image::Pointer GeomImage);
    template <typename TPixel, unsigned int VImageDimension>
    static mitk::Image::Pointer Convert8BitQImageToMitkImage( const QImage* input, const mitk::Image::Pointer GeomImage);
};

#endif
