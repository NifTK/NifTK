/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASUpSamplingFilter_h
#define itkMIDASUpSamplingFilter_h

#include "itkImageToImageFilter.h"

namespace itk
{

/**
 * \class MIDASUpSamplingFilter
 * \brief Performs the up sampling described in step 5 of
 * "Interactive Algorithms for the segmentation and quantification of 3-D MRI scans"
 * Freeborough et. al. CMPB 53 (1997) 15-25.
 *
 * <pre>
 * MIDASUpSamplingFilter requires two inputs in this order:
 *   0. The down sized image that is to be upsampled to match the original sized image.
 *   1. The original sized image
 * </pre>
 *
 * \ingroup midas_morph_editor
 */

  template <class TInputImage, class TOutputImage>
  class ITK_EXPORT MIDASUpSamplingFilter : public ImageToImageFilter<TInputImage, TOutputImage>
  {
  public:
    /** Standard class typedefs */
    typedef MIDASUpSamplingFilter                         Self;
    typedef ImageToImageFilter<TInputImage, TOutputImage> SuperClass;
    typedef SmartPointer<Self>                            Pointer;
    typedef SmartPointer<const Self>                      ConstPointer;

    /** Method for creation through the object factory */
    itkNewMacro(Self);

    /** Typedef to describe the type of pixel. */
    typedef typename TInputImage::PixelType PixelType;

    /** Typedef to describe the type of pixel index. */
    typedef typename TInputImage::IndexType IndexType;

    /** Typedef to describe the size. */
    typedef typename TInputImage::SizeType SizeType;

    /** Run-time type information (and related methods) */
    itkTypeMacro(MIDASUpSamplingFilter, ImageToImageFilter);

    /** Some additional typedefs */
    typedef TInputImage                               InputImageType;
    typedef typename InputImageType::ConstPointer     InputImageConstPointer;
    typedef typename InputImageType::Pointer          InputImagePointer;
    typedef typename InputImageType::SizeType         InputImageSizeType;
    typedef typename InputImageType::IndexType        InputImageIndexType;
    typedef typename InputImageType::RegionType       InputImageRegionType;

    typedef TOutputImage                              OutputImageType;
    typedef typename OutputImageType::Pointer         OutputImagePointer;
    typedef typename OutputImageType::SizeType        OutputImageSizeType;
    typedef typename OutputImageType::IndexType       OutputImageIndexType;
    typedef typename OutputImageType::RegionType      OutputImageRegionType;

    /** Set/Get methods to get/set the Upsample factor */
    itkSetMacro(UpSamplingFactor, unsigned int);
    itkGetConstMacro(UpSamplingFactor, unsigned int);

    /** Set/Get methods to set the output value for inside the region. Default 1. */
    itkSetMacro(InValue, PixelType);
    itkGetConstMacro(InValue, PixelType);

    /** Set/Get methods to set the output value for inside the region. Default 0. */
    itkSetMacro(OutValue, PixelType);
    itkGetConstMacro(OutValue, PixelType);

    /** Override these two methods to prepare the input and output images for upsampling */
    virtual void GenerateInputRequestedRegion();
    virtual void GenerateOutputInformation();

  protected:
    MIDASUpSamplingFilter();
    virtual ~MIDASUpSamplingFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** The main method to upsample the input image in this single-threaded class */
    virtual void GenerateData();
    
  private:
    MIDASUpSamplingFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    /** The up sampling factor, which is the number of times the number of voxels are increased by. */
    unsigned int m_UpSamplingFactor;

    PixelType m_InValue;
    PixelType m_OutValue;
  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASUpSamplingFilter.txx"
#endif

#endif
