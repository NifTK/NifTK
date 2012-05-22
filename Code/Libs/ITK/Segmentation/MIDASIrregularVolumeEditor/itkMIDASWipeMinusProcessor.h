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

#ifndef ITKMIDASWIPEMINUSPROCESSOR_H
#define ITKMIDASWIPEMINUSPROCESSOR_H

#include "itkImage.h"
#include "itkMIDASHelper.h"
#include "itkMIDASWipeProcessor.h"

namespace itk
{

/**
 * \class MIDASWipeMinusProcessor
 * \brief Provides the MIDAS Wipe Minus operation in the Irregular Volume Editor.
 * \deprecated See MIDASGeneralSegmentorView now uses itk::ImageUpdateCopyRegionProcessor.
 * \sa MIDASWipeSliceProcessor
 * \sa MIDASWipePlusProcessor
 * \sa MIDASRegionOfInterestCalculator
 * \sa ImageUpdateClearRegionProcessor
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASWipeMinusProcessor : public MIDASWipeProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASWipeMinusProcessor                     Self;
  typedef MIDASWipeProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                          Pointer;
  typedef SmartPointer<const Self>                    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASWipeMinusProcessor, MIDASWipeProcessor);

  /** Additional typedefs */
  typedef itk::ImageUpdateClearRegionProcessor<TPixel, VImageDimension> ProcessorType;
  typedef typename ProcessorType::Pointer ProcessorPointer;
  typedef itk::MIDASRegionOfInterestCalculator<TPixel, VImageDimension> CalculatorType;
  typedef typename CalculatorType::Pointer CalculatorPointer;
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  /** Calculates the region for the current slice, and sets up the processor. */
  virtual void SetOrientationAndSlice(itk::ORIENTATION_ENUM orientation, int sliceNumber);

protected:
  MIDASWipeMinusProcessor();
  virtual ~MIDASWipeMinusProcessor() {}

private:
  MIDASWipeMinusProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASWipeMinusProcessor.txx"
#endif

#endif // ITKMIDASWIPEMINUSPROCESSOR_H
