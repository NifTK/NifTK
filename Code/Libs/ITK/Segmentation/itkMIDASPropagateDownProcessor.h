/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7491 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKMIDASPROPAGATEDOWNPROCESSOR_H
#define ITKMIDASPROPAGATEDOWNPROCESSOR_H

#include "itkImage.h"
#include "itkMIDASRegionOfInterestCalculator.h"
#include "itkMIDASPropagateProcessor.h"

namespace itk
{

/**
 * \class MIDASPropagateDownProcessor
 * \brief Provides the MIDAS Propagate Down operation in the Irregular Volume Editor.
 *
 * \sa MIDASRegionOfInterestCalculator
 * \sa MIDASPropagateProcessor
 */
template <class TSegmentationPixel, class TGreyScalePixel, class TPointDataType, unsigned int VImageDimension>
class ITK_EXPORT MIDASPropagateDownProcessor : public MIDASPropagateProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASPropagateDownProcessor                                                                   Self;
  typedef MIDASPropagateProcessor<TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension> Superclass;
  typedef SmartPointer<Self>                                                                            Pointer;
  typedef SmartPointer<const Self>                                                                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASPropagateDownProcessor, MIDASPropagateProcessor);

  /** Additional typedefs, as this is the ITK way. */
  typedef itk::Image<TGreyScalePixel, VImageDimension>                                    GreyImageType;
  typedef typename GreyImageType::Pointer                                                 GreyImagePointer;
  typedef typename GreyImageType::RegionType                                              RegionType;
  typedef typename GreyImageType::SizeType                                                SizeType;
  typedef typename GreyImageType::IndexType                                               IndexType;
  typedef itk::Image<TSegmentationPixel, VImageDimension>                                 SegmentationImageType;
  typedef typename SegmentationImageType::Pointer                                         SegmentationImagePointer;

  typedef itk::MIDASRegionOfInterestCalculator<TSegmentationPixel, VImageDimension>       CalculatorType;
  typedef typename CalculatorType::Pointer                                                CalculatorPointer;
  typedef itk::ImageUpdateStrategyProcessor<TSegmentationPixel, VImageDimension>          StrategyProcessorType;
  typedef typename StrategyProcessorType::Pointer                                         StrategyProcessorPointer;
  typedef itk::ImageUpdateSlicewiseRegionGrowingAlgorithm<
    TSegmentationPixel, TGreyScalePixel, TPointDataType, VImageDimension>                 AlgorithmType;
  typedef typename AlgorithmType::Pointer                                                 AlgorithmPointer;

  /** Calculates the region for the current slice, and sets up the processor. */
  virtual void SetOrientationAndSlice(itk::ORIENTATION_ENUM orientation, int sliceNumber);

protected:
  MIDASPropagateDownProcessor();
  virtual ~MIDASPropagateDownProcessor() {}

private:
  MIDASPropagateDownProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASPropagateDownProcessor.txx"
#endif

#endif // ITKMIDASPROPAGATEDOWNPROCESSOR_H
