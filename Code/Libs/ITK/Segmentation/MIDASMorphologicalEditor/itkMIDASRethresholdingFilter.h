/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASRethresholdingFilter_h
#define itkMIDASRethresholdingFilter_h

#include "itkImageToImageFilter.h"
#include "itkMIDASDownSamplingFilter.h"
#include "itkMIDASUpSamplingFilter.h"
#include "itkMIDASMeanIntensityWithinARegionFilter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkAndImageFilter.h"
#include "itkOrImageFilter.h"

namespace itk
{

/**
 * \class MIDASRethresholdingFilter
 * \brief Performs the re-thresholding, described in step 5 of
 * "Interactive Algorithms for the segmentation and quantification of 3-D MRI scans"
 * Freeborough et. al. CMPB 53 (1997) 15-25.
 *
 * The image is downsampled by an integer amount using the parameter SetDownSamplingFactor.
 * It is eroded using the standard ITK erosion with a simple binary cross structuring element.
 * It is then upsampled to match the original size.
 *
 * The net effect is that the interior region has holes filled. This filter is implemented as a composite
 * filter to hide the memory management of different size images and the use of multiple filters.
 *
 * \ingroup midas_morph_editor
 */

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  class ITK_EXPORT MIDASRethresholdingFilter : public ImageToImageFilter<TInputImage1, TOutputImage>
  {
  public:
    /** Standard class typedefs */
    typedef MIDASRethresholdingFilter                      Self;
    typedef ImageToImageFilter<TInputImage1, TOutputImage> SuperClass;
    typedef SmartPointer<Self>                             Pointer;
    typedef SmartPointer<const Self>                       ConstPointer;

    /** Method for creation through the object factory */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(MIDASRethresholdingFilter, ImageToImageFilter);

    /** Typedef to describe the type of pixel for the first image, which should be the grey scale image. */
    typedef typename TInputImage1::PixelType PixelType1;

    /** Typedef to describe the type of pixel for the second image, which should be a binary mask image. */
    typedef typename TInputImage2::PixelType PixelType2;

    /** Some additional typedefs */
    typedef TInputImage1                              InputMainImageType;
    typedef typename InputMainImageType::Pointer      InputMainImagePointer;
    typedef typename InputMainImageType::SizeType     InputMainImageSizeType;
    typedef typename InputMainImageType::RegionType   InputMainImageRegionType;

    typedef TInputImage2                              InputMaskImageType;
    typedef typename InputMaskImageType::Pointer      InputMaskImagePointer;
    typedef typename InputMaskImageType::SizeType     InputMaskImageSizeType;
    typedef typename InputMaskImageType::RegionType   InputMaskImageRegionType;
    typedef typename InputMaskImageType::IndexType    InputMaskImageIndexType;

    typedef TInputImage2                              OutputImageType;
    typedef typename OutputImageType::Pointer         OutputImagePointer;
    typedef typename OutputImageType::RegionType      OutputImageRegionType;
    typedef typename OutputImageType::SizeType        OutputImageSizeType;
    typedef typename OutputImageType::IndexType       OutputImageIndexType;

    typedef typename itk::MIDASDownSamplingFilter<InputMaskImageType, InputMaskImageType> DownSamplingFilterType;
    typedef typename DownSamplingFilterType::Pointer DownSamplingFilterPointer;
    typedef typename itk::MIDASUpSamplingFilter<InputMaskImageType, InputMaskImageType> UpSamplingFilterType;
    typedef typename UpSamplingFilterType::Pointer UpSamplingFilterPointer;

    typedef typename itk::BinaryCrossStructuringElement<typename InputMaskImageType::PixelType, InputMaskImageType::ImageDimension > StructuringElementType;
    typedef typename itk::BinaryErodeImageFilter<InputMaskImageType, InputMaskImageType, StructuringElementType> ErosionFilterType;
    typedef typename ErosionFilterType::Pointer ErosionFilterPointer;

    typedef typename itk::MIDASMeanIntensityWithinARegionFilter<InputMainImageType, InputMaskImageType, OutputImageType> MeanFilterType;
    typedef typename MeanFilterType::Pointer MeanFilterPointer;

    typedef typename itk::BinaryThresholdImageFilter<InputMainImageType, InputMaskImageType> ThresholdFilterType;
    typedef typename ThresholdFilterType::Pointer ThresholdFilterPointer;

    typedef typename itk::AndImageFilter<InputMaskImageType, InputMaskImageType> AndFilterType;
    typedef typename AndFilterType::Pointer AndFilterPointer;

    typedef typename itk::OrImageFilter<InputMaskImageType, InputMaskImageType> OrFilterType;
    typedef typename OrFilterType::Pointer OrFilterPointer;

    /** Set the first input, for the grey scale image. */
    void SetGreyScaleImageInput(const InputMainImageType* image);

    /** Set the second input, which is the binary mask, that will be down sampled, eroded and up-sampled. */
    void SetBinaryImageInput(const InputMaskImageType* image);

    /** Set/Get methods to set the output value for inside the region. Default 1. */
    itkSetMacro(InValue, PixelType2);
    itkGetConstMacro(InValue, PixelType2);

    /** Set/Get methods to set the output value for outside the region. Default 0. */
    itkSetMacro(OutValue, PixelType2);
    itkGetConstMacro(OutValue, PixelType2);

    /** Set/Get methods to set the radius of the structuring element. Default to 2. */
    itkSetMacro(StructuringElementRadius, unsigned int );
    itkGetConstMacro(StructuringElementRadius, unsigned int );

    /** Set/Get methods to set the down-sampling factor, which defaults to 1 which passes the mask image straight through. */
    itkSetMacro(DownSamplingFactor, unsigned int);
    itkGetConstMacro(DownSamplingFactor, unsigned int);

    /** Set/Get methods to set the low percentage threshold, which defaults to 50% */
    itkSetMacro(LowPercentageThreshold, unsigned int);
    itkGetConstMacro(LowPercentageThreshold, unsigned int);

    /** Set/Get methods to set the high percentage threshold, which defaults to 150% */
    itkSetMacro(HighPercentageThreshold, unsigned int);
    itkGetConstMacro(HighPercentageThreshold, unsigned int);

    /** Set/Get methods to set a flag indicating whether we skip the intersection with the thresholded mean image (y_m in equation 3). Default false. */
    itkSetMacro(SkipIntersectionWithMeanMask, bool);
    itkGetConstMacro(SkipIntersectionWithMeanMask, bool);

  protected:
    MIDASRethresholdingFilter();
    virtual ~MIDASRethresholdingFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

  private:
    MIDASRethresholdingFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    void CopyImageToOutput(OutputImageType* image);

    /** Parameters to pass to the contained filters. */
    unsigned int m_DownSamplingFactor;
    unsigned int m_StructuringElementRadius;
    unsigned int m_LowPercentageThreshold;
    unsigned int m_HighPercentageThreshold;
    PixelType2 m_InValue;
    PixelType2 m_OutValue;
    bool m_SkipIntersectionWithMeanMask;

    /** Additional filters to implement composite filter pattern. */
    DownSamplingFilterPointer m_DownSamplingFilter;
    ErosionFilterPointer      m_ErosionFilter;
    UpSamplingFilterPointer   m_UpSamplingFilter;
    MeanFilterPointer         m_MeanFilter;
    ThresholdFilterPointer    m_ThresholdFilter;
    AndFilterPointer          m_AndFilter;
    OrFilterPointer           m_OrFilter;

  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASRethresholdingFilter.txx"
#endif

#endif
