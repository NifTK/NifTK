/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASConditionalErosionFilter_h
#define itkMIDASConditionalErosionFilter_h

#include "itkMIDASBaseConditionalMorphologyFilter.h"

namespace itk
{

/**
 * \class MIDASConditionalErosionFilter
 * \brief Performs the conditional erosion, described in step 3 of
 * "Interactive Algorithms for the segmentation and quantification of 3-D MRI scans"
 * Freeborough et. al. CMPB 53 (1997) 15-25.
 *
 * From the paper, the parameter n is the number of erosions, which is specified
 * using SetNumberOfIterations(n), and the parameter s_i, which is the upper limit
 * in grey value, above which values are not eroded, which is set using SetUpperThreshold(value).
 *
 * \ingroup midas_morph_editor
 */

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  class ITK_EXPORT MIDASConditionalErosionFilter : public MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage>
  {
  public:
    /** Standard class typedefs */
    typedef MIDASConditionalErosionFilter                                                  Self;
    typedef MIDASBaseConditionalMorphologyFilter<TInputImage1, TInputImage2, TOutputImage> SuperClass;
    typedef SmartPointer<Self>                                                             Pointer;
    typedef SmartPointer<const Self>                                                       ConstPointer;

    /** Method for creation through the object factory */
    itkNewMacro(Self);

    /** Run-time type information (and related methods) */
    itkTypeMacro(MIDASConditionalErosionFilter, MIDASBaseConditionalMorphologyFilter);

    /** Typedef to describe the type of pixel for the first image, which should be the binary mask image. */
    typedef typename TInputImage1::PixelType PixelType1;

    /** Typedef to describe the type of pixel for the second image, which should be a grey scale image. */
    typedef typename TInputImage2::PixelType PixelType2;

    /** Some additional typedefs */
    typedef TInputImage1                              InputMaskImageType;
    typedef typename InputMaskImageType::Pointer      InputMaskImagePointer;
    typedef typename InputMaskImageType::SizeType     InputMaskImageSizeType;
    typedef typename InputMaskImageType::RegionType   InputMaskImageRegionType;
    typedef typename InputMaskImageType::IndexType    InputMaskImageIndexType;

    typedef TInputImage2                              InputMainImageType;
    typedef typename InputMainImageType::Pointer      InputMainImagePointer;
    typedef typename InputMainImageType::SizeType     InputMainImageSizeType;
    typedef typename InputMainImageType::RegionType   InputMainImageRegionType;

    typedef TOutputImage                              OutputImageType;
    typedef typename OutputImageType::Pointer         OutputImagePointer;
    typedef typename OutputImageType::RegionType      OutputImageRegionType;
    typedef typename OutputImageType::SizeType        OutputImageSizeType;
    typedef typename OutputImageType::IndexType       OutputImageIndexType;
    typedef typename itk::ConstNeighborhoodIterator<OutputImageType>::RadiusType  OutputImageRadiusType;

    /** Set/Get the upper threshold. Pixels are only eroded if the grey value is below this threshold. */
    itkSetMacro(UpperThreshold, PixelType2);
    itkGetConstMacro(UpperThreshold, PixelType2);

  protected:
    MIDASConditionalErosionFilter();
    virtual ~MIDASConditionalErosionFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Called by GenerateData() in MIDASBaseConditionalMorphologyFilter. */
    void DoFilter(InputMainImageType* inGrey, OutputImageType* inMask, OutputImageType *out);

  private:
    MIDASConditionalErosionFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    /** A pixel is on the boudary, if it is 1, and at least 1 6 connected neighbour (in 3D) is zero. */
    bool IsOnBoundaryOfObject(const OutputImageIndexType &voxelIndex, const OutputImageType* inMask);

    /** The upper threshold, below which, pixels are not eroded. */
    PixelType2 m_UpperThreshold;
  };

} //end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASConditionalErosionFilter.txx"
#endif

#endif
