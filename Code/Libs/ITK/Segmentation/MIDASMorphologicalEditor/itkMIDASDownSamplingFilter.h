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

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASDownSamplingFilter_h
#define itkMIDASDownSamplingFilter_h

#include "itkImageToImageFilter.h"

namespace itk
{

/**
 * \class MIDASDownSamplingFilter
 * \brief Performs the down sampling described in step 5 of
 * "Interactive Algorithms for the segmentation and quantification of 3-D MRI scans"
 * Freeborough et. al. CMPB 53 (1997) 15-25.
 *
 * \ingroup midas_morph_editor
 */

  template <class TInputImage, class TOutputImage>
  class ITK_EXPORT MIDASDownSamplingFilter : public ImageToImageFilter<TInputImage, TOutputImage>
  {
  public:
    /** Standard class typedefs */
    typedef MIDASDownSamplingFilter                       Self;
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
    itkTypeMacro(MIDASDownSamplingFilter, ImageToImageFilter);

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

    /** Set/Get methods to get/set the downsample factor */
    itkSetMacro(DownSamplingFactor, unsigned int);
    itkGetConstMacro(DownSamplingFactor, unsigned int);

    /** Set/Get methods to set the output value for inside the region. Default 1. */
    itkSetMacro(InValue, PixelType);
    itkGetConstMacro(InValue, PixelType);

    /** Set/Get methods to set the output value for inside the region. Default 0. */
    itkSetMacro(OutValue, PixelType);
    itkGetConstMacro(OutValue, PixelType);

    /** Override these two methods to prepare the input and output images for subsampling */
    virtual void GenerateInputRequestedRegion();
    virtual void GenerateOutputInformation();
 
  protected:
    MIDASDownSamplingFilter();
    virtual ~MIDASDownSamplingFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** The main method to down sample the input image in this single-threaded class */
    virtual void GenerateData();
    
  private:
    MIDASDownSamplingFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    /** The down sampling factor, which is the number of times the number of voxels are reduced by. */
    unsigned int m_DownSamplingFactor;

    PixelType m_InValue;
    PixelType m_OutValue;
  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASDownSamplingFilter.txx"
#endif

#endif
