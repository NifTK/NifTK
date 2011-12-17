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

#ifndef ITKMIDASPROPAGATEPROCESSOR_H
#define ITKMIDASPROPAGATEPROCESSOR_H

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionProcessor.h"
#include "itkImageUpdateStrategyProcessor.h"
#include "itkImageUpdateSlicewiseRegionGrowingAlgorithm.h"

namespace itk
{

/**
 * \class MIDASPropagateProcessor
 * \brief Base class to support the MIDAS Propagate Up and Down operations found in the MIDAS Irregular volume editor.
 *
 * \sa MIDASRegionProcessor
 * \sa MIDASRegionOfInterestCalculator
 * \sa MIDASPropagateUpProcessor
 * \sa MIDASPropagateDownProcessor
 * \sa ImageUpdateStrategyProcessor
 * \sa ImageUpdateSlicewiseRegionGrowingAlgorithm
 */
template <class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
class ITK_EXPORT MIDASPropagateProcessor : public MIDASRegionProcessor<TSegmentationPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASPropagateProcessor                                   Self;
  typedef MIDASRegionProcessor<TSegmentationPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                                        Pointer;
  typedef SmartPointer<const Self>                                  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASPropagateProcessor, MIDASRegionProcessor);

  /** Additional typedefs */
  typedef itk::Image<TGreyScalePixel, VImageDimension>                                    GreyImageType;
  typedef typename GreyImageType::Pointer                                                 GreyImagePointer;
  typedef typename GreyImageType::PixelType                                               GreyImagePixelType;
  typedef typename GreyImageType::RegionType                                              RegionType;
  typedef typename GreyImageType::SizeType                                                SizeType;
  typedef typename GreyImageType::IndexType                                               IndexType;
  typedef itk::Image<TSegmentationPixel, VImageDimension>                                 SegmentationImageType;
  typedef typename SegmentationImageType::Pointer                                         SegmentationImagePointer;
  typedef itk::PointSet<TPointDataType, VImageDimension>                                  PointSetType;
  typedef typename PointSetType::Pointer                                                  PointSetPointer;
  typedef itk::MIDASRegionOfInterestCalculator<TSegmentationPixel, VImageDimension>       CalculatorType;
  typedef typename CalculatorType::Pointer                                                CalculatorPointer;
  typedef itk::ImageUpdateStrategyProcessor<TSegmentationPixel, VImageDimension>          StrategyProcessorType;
  typedef typename StrategyProcessorType::Pointer                                         StrategyProcessorPointer;
  typedef itk::ImageUpdateSlicewiseRegionGrowingAlgorithm<
    TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension>                 AlgorithmType;
  typedef typename AlgorithmType::Pointer                                                 AlgorithmPointer;

  /** Set/Get the input grey scale image, that is used for region growing according to intensity. */
  void SetGreyScaleImage(GreyImageType *image) { m_Algorithm->SetGreyScaleImage(image); this->Modified(); }
  GreyImageType GetGreyScaleImage() const { return m_Algorithm->GetGreyScaleImage(); }

  /** Set/Get the input seeds, in millimetre coordinates. */
  void SetSeeds(PointSetType *seeds) { m_Algorithm->SetSeeds(seeds); this->Modified(); }
  PointSetType* GetSeeds() const { return m_Algorithm->GetSeeds(); }

  /** Set/Get the input contours, in millimetre coordinates. */
  void SetContours(PointSetType *contours) { m_Algorithm->SetContours(contours); this->Modified(); }
  PointSetType* GetContours() const { return m_Algorithm->GetContours(); }

  /** Set/Get the input lower threshold */
  void SetLowerThreshold(GreyImagePixelType lowerThreshold) { m_Algorithm->SetLowerThreshold(lowerThreshold); this->Modified(); }
  GreyImagePixelType GetLowerThreshold() const { return m_Algorithm->GetLowerThreshold(); }

  /** Set/Get the input upper threshold */
  void SetUpperThreshold(GreyImagePixelType upperThreshold) { m_Algorithm->SetUpperThreshold(upperThreshold); this->Modified(); }
  GreyImagePixelType GetUpperThreshold() const { return m_Algorithm->GetUpperThreshold(); }

  /** This method should set the derived class up, and then call this->Modified(). */
  virtual void SetOrientationAndSlice(itk::ORIENTATION_ENUM orientation, int sliceNumber)
  {
    m_Algorithm->SetSliceNumber(sliceNumber);
    m_Algorithm->SetOrientation(orientation);
    this->Modified();
  }
  int GetSliceNumber() const { return m_Algorithm->GetSliceNumber(); }
  typename itk::ORIENTATION_ENUM GetOrientation() const { return m_Algorithm->GetOrientation(); }

  virtual void SetDebug(bool b) { itk::MIDASRegionProcessor<TSegmentationPixel, VImageDimension>::SetDebug(b); m_Strategy->SetDebug(b); m_Algorithm->SetDebug(b); this->Modified(); }
  virtual void DebugOn() { this->SetDebug(true); }
  virtual void DebugOff() { this->SetDebug(false); }

protected:
  MIDASPropagateProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASPropagateProcessor() {}

  StrategyProcessorPointer m_Strategy;
  AlgorithmPointer         m_Algorithm;

private:
  MIDASPropagateProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASPropagateProcessor.txx"
#endif

#endif // ITKMIDASPROPAGATEPROCESSOR_H
