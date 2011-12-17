/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 12:21:57 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7522 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKIMAGEUPDATESLICEWISEREGIONGROWINGALGORITHM_H
#define ITKIMAGEUPDATESLICEWISEREGIONGROWINGALGORITHM_H

#include "itkPointSet.h"
#include "itkImageUpdateStrategyAlgorithm.h"
#include "itkMIDASRegionGrowingProcessor.h"
#include "itkMIDASRegionOfInterestCalculator.h"

namespace itk
{

/**
 * \class ImageUpdateSlicewiseRegionGrowingAlgorithm
 * \brief Strategy pattern implementation that applies a region growing
 * algorithm slicewise across the region of interest where the orientation etc.
 * must be set before the Execute command is triggered.
 */
template <class TSegmentationPixel, class TGreyscalePixel, class TPointType, unsigned int VImageDimension>
class ITK_EXPORT ImageUpdateSlicewiseRegionGrowingAlgorithm : public ImageUpdateStrategyAlgorithm<TSegmentationPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef ImageUpdateSlicewiseRegionGrowingAlgorithm                                     Self;
  typedef ImageUpdateStrategyAlgorithm<TSegmentationPixel, VImageDimension>              Superclass;
  typedef SmartPointer<Self>                                                             Pointer;
  typedef SmartPointer<const Self>                                                       ConstPointer;

  /** Additional typedefs, as this is the ITK way. */
  typedef TSegmentationPixel                                                             SegmentationPixelType;
  typedef Image<SegmentationPixelType, VImageDimension>                                  SegmentationImageType;
  typedef typename SegmentationImageType::Pointer                                        SegmentationImagePointer;
  typedef typename SegmentationImageType::IndexType                                      IndexType;
  typedef typename SegmentationImageType::SizeType                                       SizeType;
  typedef typename SegmentationImageType::RegionType                                     RegionType;
  typedef TGreyscalePixel                                                                GreyscalePixelType;
  typedef Image<GreyscalePixelType, VImageDimension>                                     GreyscaleImageType;
  typedef typename GreyscaleImageType::Pointer                                           GreyscaleImagePointer;
  typedef TPointType                                                                     PointDataType;
  typedef PointSet<PointDataType, VImageDimension>                                       PointSetType;
  typedef typename PointSetType::Pointer                                                 PointSetPointer;
  typedef MIDASRegionGrowingProcessor<
    GreyscaleImageType, SegmentationImageType, PointSetType>                             RegionGrowingProcessorType;
  typedef typename RegionGrowingProcessorType::Pointer                                   RegionGrowingProcessorPointer;
  typedef itk::MIDASRegionOfInterestCalculator<SegmentationPixelType, VImageDimension >  CalculatorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageUpdateSlicewiseRegionGrowingAlgorithm, ImageUpdateStrategyAlgorithm);

  /** Set/Get the input grey scale image, that is used for region growing according to intensity. */
  void SetGreyScaleImage(GreyscaleImageType *image) { m_RegionGrowingProcessor->SetGreyScaleImage(image); this->Modified(); }
  GreyscaleImageType GetGreyScaleImage() const { return m_RegionGrowingProcessor->GetGreyScaleImage(); }

  /** Set/Get the input seeds, in millimetre coordinates. */
  void SetSeeds(PointSetType *seeds) { m_RegionGrowingProcessor->SetSeeds(seeds); this->Modified(); }
  PointSetType* GetSeeds() const { return m_RegionGrowingProcessor->GetSeeds(); }

  /** Set/Get the input contours, in millimetre coordinates. */
  void SetContours(PointSetType *contours) { m_RegionGrowingProcessor->SetContours(contours); this->Modified(); }
  PointSetType* GetContours() const { return m_RegionGrowingProcessor->GetContours(); }

  /** Set/Get the input region of interest, which is the region to process. */
  void SetRegionOfInterest(RegionType region) { m_RegionGrowingProcessor->SetRegionOfInterest(region); this->Modified(); }
  RegionType GetRegionOfInterest() const { return m_RegionGrowingProcessor->GetRegionOfInterest(); }

  /** Set/Get the input slice number, which is a reference used to control which contours and seeds to process. */
  void SetSliceNumber(int sliceNumber) { m_RegionGrowingProcessor->SetSliceNumber(sliceNumber); this->Modified(); }
  int GetSliceNumber() const { return m_RegionGrowingProcessor->GetSliceNumber(); }

  /** Set/Get the input orientation we are interested in (for 2D slicewise propagation). */
  void SetOrientation(itk::ORIENTATION_ENUM orientation) { m_RegionGrowingProcessor->SetOrientation(orientation); this->Modified(); }
  typename itk::ORIENTATION_ENUM GetOrientation() const { return m_RegionGrowingProcessor->GetOrientation(); }

  /** Set/Get the input lower threshold */
  void SetLowerThreshold(GreyscalePixelType lowerThreshold) { m_RegionGrowingProcessor->SetLowerThreshold(lowerThreshold); this->Modified(); }
  GreyscalePixelType GetLowerThreshold() const { return m_RegionGrowingProcessor->GetLowerThreshold(); }

  /** Set/Get the input upper threshold */
  void SetUpperThreshold(GreyscalePixelType upperThreshold) { m_RegionGrowingProcessor->SetUpperThreshold(upperThreshold); this->Modified(); }
  GreyscalePixelType GetUpperThreshold() const { return m_RegionGrowingProcessor->GetUpperThreshold(); }

  /** Main Virtual Method That All Subclasses Must Implement, but shouldn't call. */
  virtual SegmentationImageType* Execute(SegmentationImageType* imageToBeModified);

  /** Set debug flags. */
  virtual void SetDebug(bool b) { itk::Object::SetDebug(b); m_RegionGrowingProcessor->SetDebug(b); this->Modified(); }
  virtual void DebugOn() { this->SetDebug(true); }
  virtual void DebugOff() { this->SetDebug(false); }

protected:
  ImageUpdateSlicewiseRegionGrowingAlgorithm();
  virtual ~ImageUpdateSlicewiseRegionGrowingAlgorithm() {}

private:
  ImageUpdateSlicewiseRegionGrowingAlgorithm(const Self&); //purposely not implemented
  void PrintSelf(std::ostream& os, Indent indent) const;
  void operator=(const Self&); //purposely not implemented

  RegionGrowingProcessorPointer             m_RegionGrowingProcessor;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageUpdateSlicewiseRegionGrowingAlgorithm.txx"
#endif

#endif // ITKIMAGEUPDATESLICEWISEREGIONGROWINGALGORITHM_H
