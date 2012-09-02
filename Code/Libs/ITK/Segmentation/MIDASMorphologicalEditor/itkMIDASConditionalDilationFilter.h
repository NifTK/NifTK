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
#ifndef itkMIDASConditionalDilationFilter_h
#define itkMIDASConditionalDilationFilter_h

#include "itkMIDASBaseConditionalMorphologyFilter.h"
#include "itkMIDASMeanIntensityWithinARegionFilter.h"

namespace itk
{

/**
 * \class MIDASConditionalDilationFilter
 * \brief Performs the conditional dilation, described in step 4 of
 * "Interactive Algorithms for the segmentation and quantification of 3-D MRI scans"
 * Freeborough et. al. CMPB 53 (1997) 15-25.
 *
 * From the paper, the parameter m is the number of dilations, which is set using
 * the method SetNumberOfIterations(m), and also the lower (p_lo) and upper (p_high)
 * threshold percentage values, which are percentages of the mean grey intensity,
 * set using SetLowerThreshold and SetUpperThreshold respectively.
 *
 * \ingroup midas_morph_editor
 */

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  class ITK_EXPORT MIDASConditionalDilationFilter : public MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>
  {
  public:
    /** Standard class typedefs */
    typedef MIDASConditionalDilationFilter                                                 Self;
    typedef MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage> SuperClass;
    typedef SmartPointer<Self>                                                             Pointer;
    typedef SmartPointer<const Self>                                                       ConstPointer;

    /** Method for creation through the object factory */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(MIDASConditionalDilationFilter, MIDASBaseConditionalMorphologyFilter);

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
    typedef typename InputMaskImageType::PixelType    InputMaskImagePixelType;
    typedef typename InputMaskImageType::IndexType    InputMaskImageIndexType;

    typedef TInputImage2                              OutputImageType;
    typedef typename OutputImageType::Pointer         OutputImagePointer;
    typedef typename OutputImageType::RegionType      OutputImageRegionType;
    typedef typename OutputImageType::SizeType        OutputImageSizeType;
    typedef typename OutputImageType::IndexType       OutputImageIndexType;
    typedef typename itk::ConstNeighborhoodIterator<OutputImageType>::RadiusType  OutputImageRadiusType;

    typedef typename itk::MIDASMeanIntensityWithinARegionFilter<TInputImage1, TInputImage2, TOutputImage> MeanFilterType;
    typedef typename MeanFilterType::Pointer MeanFilterPointer;

    /** Set/Get methods to set the lower threshold, as percentages of the mean intensity over the input region. */
    itkSetMacro(LowerThreshold, unsigned int);
    itkGetConstMacro(LowerThreshold, unsigned int);

    /** Set/Get methods to set the upper threshold, as percentages of the mean intensity over the input region. */
    itkSetMacro(UpperThreshold, unsigned int);
    itkGetConstMacro(UpperThreshold, unsigned int);

    /** Sets the connection breaker image, so it can be applied at each iteration. */
    void SetConnectionBreakerImage(const InputMaskImageType* image);

  protected:
    MIDASConditionalDilationFilter();
    virtual ~MIDASConditionalDilationFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Called by GenerateData() in MIDASBaseConditionalMorphologyFilter. */
    void DoFilter(InputMainImageType* inGrey, OutputImageType* inMask, OutputImageType *out);

  private:
    MIDASConditionalDilationFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    /** A pixel is next to a region/object if it's value is 0, and at least one of the 6 connected (in 3D) neighbourhood is a 1. */
    bool IsNextToObject(OutputImageIndexType &voxelIndex, OutputImageType* inMask);

    /** Calculates the mean value of the input. */
    MeanFilterPointer m_MeanFilter;

    /** The upper and lower thresholds, as a percentage of the mean grey value of the input region. */
    unsigned int m_LowerThreshold;
    unsigned int m_UpperThreshold;
  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASConditionalDilationFilter.txx"
#endif

#endif
