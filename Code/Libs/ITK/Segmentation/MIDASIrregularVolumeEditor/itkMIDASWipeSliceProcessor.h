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

#ifndef ITKMIDASWIPESLICEPROCESSOR_H
#define ITKMIDASWIPESLICEPROCESSOR_H

#include "itkImage.h"
#include "itkMIDASWipeProcessor.h"
#include "itkMIDASHelper.h"

namespace itk
{

/**
 * \class MIDASWipeSliceProcessor
 * \brief Provides the MIDAS Wipe operation in the Irregular Volume Editor.
 * \deprecated See MIDASGeneralSegmentorView now uses itk::ImageUpdateCopyRegionProcessor.
 * \sa MIDASWipeProcessor
 * \sa MIDASWipePlusProcessor
 * \sa MIDASWipeMinusProcessor
 * \sa MIDASRegionOfInterestCalculator
 * \sa ImageUpdateClearRegionProcessor
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASWipeSliceProcessor : public MIDASWipeProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef MIDASWipeSliceProcessor                     Self;
  typedef MIDASWipeProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                          Pointer;
  typedef SmartPointer<const Self>                    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASWipeSliceProcessor, MIDASWipeProcessor);

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
  MIDASWipeSliceProcessor();
  virtual ~MIDASWipeSliceProcessor() {}

private:
  MIDASWipeSliceProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASWipeSliceProcessor.txx"
#endif

#endif // ITKMIDASWIPESLICEPROCESSOR_H
