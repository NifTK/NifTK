/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7522 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKMIDASREGIONGROWINGPROCESSOR_H
#define ITKMIDASREGIONGROWINGPROCESSOR_H

#include "itkObject.h"
#include "itkImage.h"
#include "itkExtractImageFilter.h"
#include "itkPasteImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkPolyLineParametricPath.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionGrowingImageFilter.h"
#include "itkMIDASRegionOfInterestCalculator.h"
#include "itkContinuousIndex.h"

namespace itk
{

/**
 * \class MIDASRegionGrowingProcessor
 * \brief Implements the Region growing (thresholding) for MIDAS.
 * \deprecated This class operates 2D slice by slice, whereas propagate up/down/3D is a 3D region growing.
 *
 * The template types are
 * <pre>
 * TInputImage = Grey scale image
 * TOutputImage = Binary image
 * TPointSet = Seed points for region growing (in millimetre coordinates).
 * </pre>
 *
 * If you provide a region of interest that is a slice, this class will
 * operate on a single slice (i.e. a 2D slice within a 3D volume).
 *
 * If you provide a region of interest that is a volume, this class will
 * iterate through each slice, and repeatedly do region growing in 2D,
 * pasting the output back into the same destination image. In this case,
 * you must specify an orientation, then the seeds are propagated to each slice.
 *
 * In both cases, the seeds must be within the specified 'reference' slice or will be ignored.
 *
 * NOTE: This is not an ITK pipeline. It operates 'in place'.
 * The DestinationImage is written to within the given RegionOfInterest.
 * After running the filter, you must call "GetDestinationImage()" to get the new address.
 */
template <class TInputImage, class TOutputImage, class TPointSet>
class ITK_EXPORT MIDASRegionGrowingProcessor : public Object {

public:

  /** Standard class typedefs */
  typedef MIDASRegionGrowingProcessor Self;
  typedef Object                      Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

  /** Method for creation through the object factory */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASRegionGrowingProcessor, Object);

  /** Additional typedefs */
  typedef TPointSet                                 PointSetType;
  typedef TInputImage                               GreyImageType;
  typedef typename TInputImage::PixelType           GreyImagePixelType;
  typedef typename TInputImage::RegionType          RegionType;
  typedef typename TInputImage::SizeType            SizeType;
  typedef typename TInputImage::IndexType           IndexType;
  typedef typename TInputImage::PointType           PointType;
  typedef TOutputImage                              SegmentationImageType;
  typedef typename SegmentationImageType::PixelType SegmentationImagePixelType;

  typedef itk::PolyLineParametricPath<3>              ParametricPathType;
  typedef typename ParametricPathType::VertexListType ParametricPathVertexListType;
  typedef typename ParametricPathType::VertexType     ParametricPathVertexType;
  typedef typename ParametricPathType::Pointer        ParametricPathPointer;
  typedef std::vector<ParametricPathPointer>          ParametricPathVectorType;

  typedef itk::ExtractImageFilter<GreyImageType, GreyImageType>                               ExtractGreySliceFromGreyImageFilterType;
  typedef itk::CastImageFilter<GreyImageType, SegmentationImageType>                          CastGreySliceToSegmentationSliceFilterType;
  typedef itk::MIDASRegionGrowingImageFilter<GreyImageType, SegmentationImageType, TPointSet> RegionGrowingBySliceFilterType;
  typedef itk::PasteImageFilter<SegmentationImageType, SegmentationImageType>                 PasteRegionFilterType;
  typedef itk::MIDASRegionOfInterestCalculator<SegmentationImagePixelType, ::itk::GetImageDimension<SegmentationImageType>::ImageDimension >    CalculatorType;
  typedef itk::ContinuousIndex<double, ::itk::GetImageDimension<SegmentationImageType>::ImageDimension> ContinuousIndexType;

  /** Set/Get the input grey scale image, that is used for region growing according to intensity. */
  itkSetObjectMacro(GreyScaleImage, GreyImageType);
  itkGetObjectMacro(GreyScaleImage, GreyImageType);

  /** Set/Get the input seeds, in millimetre coordinates. */
  itkSetObjectMacro(Seeds, PointSetType);
  itkGetObjectMacro(Seeds, PointSetType);

  /** Set/Get the input slice number, which is a reference used to control which contours and seeds to process. */
  itkSetMacro(SliceNumber, int);
  itkGetMacro(SliceNumber, int);

  /** Set/Get the input orientation we are interested in (for 2D slicewise propagation). */
  itkSetMacro(Orientation, ORIENTATION_ENUM);
  itkGetMacro(Orientation, ORIENTATION_ENUM);

  /** Set/Get the input lower threshold */
  itkSetMacro(LowerThreshold, GreyImagePixelType);
  itkGetMacro(LowerThreshold, GreyImagePixelType);

  /** Set/Get the input upper threshold */
  itkSetMacro(UpperThreshold, GreyImagePixelType);
  itkGetMacro(UpperThreshold, GreyImagePixelType);

  /** Set/Get the input region of interest, which is the region to process. */
  itkSetMacro(RegionOfInterest, RegionType);
  itkGetMacro(RegionOfInterest, RegionType);

  /** Set/Get the output destination image, that we write to. */
  itkSetObjectMacro(DestinationImage, SegmentationImageType);
  itkGetObjectMacro(DestinationImage, SegmentationImageType);

  /** Set the input contours, in millimetre coordinates. */
  void SetContours(ParametricPathVectorType& contours);

  /** The main method that makes it all happen, and will ultimately write back to the destination image. */
  void Execute();

  /** Set debug flags */
  virtual void SetDebug(bool b) { itk::Object::SetDebug(b); m_RegionOfInterestCalculator->SetDebug(b); }
  virtual void DebugOn() { this->SetDebug(true); }
  virtual void DebugOff() { this->SetDebug(false); }

protected:
  MIDASRegionGrowingProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASRegionGrowingProcessor() {}

private:
  MIDASRegionGrowingProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool RegionOK(RegionType region);
  void PropagatePointList(
      int axis,
      RegionType currentRegion,
      typename PointSetType::Pointer pointsIn,
      typename PointSetType::Pointer pointsOut
      );
  typename ExtractGreySliceFromGreyImageFilterType::Pointer    m_ExtractGreySliceFromReferenceImageFilter;
  typename CastGreySliceToSegmentationSliceFilterType::Pointer m_CastGreySliceToSegmentationSliceFilter;
  typename RegionGrowingBySliceFilterType::Pointer             m_RegionGrowingBySliceFilter;
  typename PasteRegionFilterType::Pointer                      m_PasteRegionFilter;
  typename CalculatorType::Pointer                             m_RegionOfInterestCalculator;

  ORIENTATION_ENUM                          m_Orientation;
  int                                       m_SliceNumber;
  RegionType                                m_RegionOfInterest;
  typename GreyImageType::Pointer           m_GreyScaleImage;
  typename SegmentationImageType::Pointer   m_DestinationImage;
  typename PointSetType::Pointer            m_Seeds;
  ParametricPathVectorType                  m_Contours;
  GreyImagePixelType                        m_LowerThreshold;
  GreyImagePixelType                        m_UpperThreshold;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASRegionGrowingProcessor.txx"
#endif

#endif // ITKMIDASREGIONGROWINGPIPELINE_H
