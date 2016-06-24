/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkLargestConnectedComponentFilter_h
#define itkLargestConnectedComponentFilter_h

#include <itkImageToImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkImage.h>
#include <map>
#include <algorithm>


namespace itk
{

/**
 * \class LargestConnectedComponentFilter
 * \brief Does connected component analysis and outputs a
 * binary volume of the largest connected component.
 *
 * Alternatively, use itkMIDASMorphologicalSegmentorLargestConnectedComponentImageFilter as it is faster.
 *
 * Both input and output types should be integer types, and the
 * output type is binary, so should be unsigned char.
 *
 */

  template <class TInputImage, class TOutputImage>
  class ITK_EXPORT LargestConnectedComponentFilter : public ImageToImageFilter<TInputImage, TOutputImage>
  {
  public:
    /** Standard class typedefs */
    typedef LargestConnectedComponentFilter          Self;
    typedef ImageToImageFilter<TInputImage, TOutputImage> SuperClass;
    typedef SmartPointer<Self>                            Pointer;
    typedef SmartPointer<const Self>                      ConstPointer;

    /** Method for creation through the object factory */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(LargestConnectedComponentFilter, ImageToImageFilter);

    /** Some additional typedefs */
    typedef TInputImage                               InputImageType;
    typedef typename InputImageType::PixelType        InputImagePixelType;
    typedef typename InputImageType::IndexType        InputImageIndexType;
    typedef typename InputImageType::SizeType         InputImageSizeType;
    typedef typename InputImageType::ConstPointer     InputImageConstPointer;
    typedef typename InputImageType::RegionType       InputImageRegionType;

    typedef TOutputImage                              OutputImageType;
    typedef typename OutputImageType::PixelType       OutputImagePixelType;
    typedef typename OutputImageType::Pointer         OutputImagePointer;
    typedef typename OutputImageType::RegionType      OutputImageRegionType;

    itkStaticConstMacro(ImageDimension, unsigned int,
                        TInputImage::ImageDimension);

    typedef short                                                                 InternalPixelType;
    typedef itk::Image<InternalPixelType, itkGetStaticConstMacro(ImageDimension)> InternalImageType;
    typedef typename InternalImageType::Pointer                                   InternalImagePointer;
    typedef typename InternalImageType::RegionType                                InternalImageRegionType;

    /** Set/Get methods to set the value on the input image that is considered background. Default 0. */
    itkSetMacro(InputBackgroundValue, InputImagePixelType);
    itkGetConstMacro(InputBackgroundValue, InputImagePixelType);

    /** Set/Get methods to set the output value for outside the largest region. Default 0. */
    itkSetMacro(OutputBackgroundValue, OutputImagePixelType);
    itkGetConstMacro(OutputBackgroundValue, OutputImagePixelType);

    /** Set/Get methods to set the output value for inside the largest region. Default 1. */
    itkSetMacro(OutputForegroundValue, OutputImagePixelType);
    itkGetConstMacro(OutputForegroundValue, OutputImagePixelType);

  protected:
    LargestConnectedComponentFilter();
    virtual ~LargestConnectedComponentFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** The main method to implement the connected component labeling in this single-threaded class */
    virtual void GenerateData();

    typedef typename itk::CastImageFilter<TInputImage, InternalImageType> CastImageFilterType;
    typedef typename itk::ConnectedComponentImageFilter<InternalImageType, InternalImageType> ConnectedComponentFilterType;

  private:
    LargestConnectedComponentFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    InputImagePixelType  m_InputBackgroundValue;
    OutputImagePixelType m_OutputBackgroundValue;
    OutputImagePixelType m_OutputForegroundValue;

    typename CastImageFilterType::Pointer m_CastFilter;
    typename ConnectedComponentFilterType::Pointer m_ConnectedFilter;
  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLargestConnectedComponentFilter.txx"
#endif

#endif
