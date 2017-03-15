/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMIDASThresholdingRegionGrowingImageFilter_h
#define itkMIDASThresholdingRegionGrowingImageFilter_h

#include <itkMIDASRegionGrowingImageFilter.h>

namespace itk
{

/**
 * \class MIDASThresholdingRegionGrowingImageFilter
 * \brief Implements region growing limited by contours.
 */
template <class TInputImage, class TOutputImage, class TPointSet>
class ITK_EXPORT MIDASThresholdingRegionGrowingImageFilter : public MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>
{

public:
  typedef MIDASThresholdingRegionGrowingImageFilter     Self;
  typedef SmartPointer<const Self>                      ConstPointer;
  typedef SmartPointer<Self>                            Pointer;
  typedef MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet> Superclass;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASThresholdingRegionGrowingImageFilter, MIDASRegionGrowingImageFilter)

  typedef TInputImage                                   InputImageType;
  typedef typename InputImageType::PixelType            InputPixelType;
  typedef TOutputImage                                  OutputImageType;
  typedef typename OutputImageType::RegionType          OutputImageRegionType;
  typedef typename OutputImageType::SizeType            OutputImageSizeType;
  typedef typename OutputImageType::IndexType           OutputImageIndexType;
  typedef typename OutputImageType::Pointer             OutputImagePointerType;
  typedef typename OutputImageType::ConstPointer        OutputImageConstPointerType;
  typedef typename OutputImageType::PixelType           OutputPixelType;
  typedef TPointSet                                     PointSetType;

  typedef itk::ContinuousIndex<double,TInputImage::ImageDimension> ContinuousIndexType;
  typedef itk::PolyLineParametricPath<TInputImage::ImageDimension> ParametricPathType;

  typedef typename ParametricPathType::Pointer          ParametricPathPointer;
  typedef std::vector<ParametricPathPointer>            ParametricPathVectorType;
  typedef typename ParametricPathType::VertexListType   ParametricPathVertexListType;
  typedef typename ParametricPathType::VertexType       ParametricPathVertexType;

  /**
   * \brief Values lower than the LowerThreshold are not grown into.
   */
  itkSetMacro(LowerThreshold, InputPixelType)
  itkGetConstMacro(LowerThreshold, InputPixelType)

  /**
   * \brief Values higher than the UpperThreshold are not grown into.
   */
  itkSetMacro(UpperThreshold, InputPixelType)
  itkGetConstMacro(UpperThreshold, InputPixelType)

protected:

  MIDASThresholdingRegionGrowingImageFilter(); // purposely hidden
  virtual ~MIDASThresholdingRegionGrowingImageFilter() {} // purposely hidden

private:

  MIDASThresholdingRegionGrowingImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  /**
   * \brief The main region growing logic is here, where we decide whether to add the nextImgIdx to a stack.
   * \param[In] r_stack current stack of pixels under consideration, initialised by the available seeds.
   * \param[In] currentImgIdx the current location being considered
   * \param[In] nextImgIdx the next pixel
   * \param[Out] true if the pixel should be added and false otherwise
   */
  virtual void ConditionalAddPixel(
                  std::stack<typename OutputImageType::IndexType> &r_stack,
                  const typename OutputImageType::IndexType &currentImgIdx,
                  const typename OutputImageType::IndexType &nextImgIdx,
                  const bool &isFullyConnected) override;

  InputPixelType                         m_LowerThreshold;
  InputPixelType                         m_UpperThreshold;
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASThresholdingRegionGrowingImageFilter.txx"
#endif

}

#endif
