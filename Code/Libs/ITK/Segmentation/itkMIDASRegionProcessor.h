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

#ifndef ITKMIDASREGIONPROCESSOR_H
#define ITKMIDASREGIONPROCESSOR_H

#include "itkObject.h"
#include "itkImageUpdateProcessor.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionOfInterestCalculator.h"

namespace itk
{

/**
 * \class MIDASWipeProcessor
 * \brief Base class to support the MIDAS Wipe, Wipe+, Wipe-, PropUp and PropDown
 * operations in the Irregular Volume Editor. The template type TPixel should
 * be that of the segmentation image type (eg. unsigned char for instance).
 *
 * \sa MIDASWipeProcessor
 * \sa MIDASPropagateProcessor
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASRegionProcessor : public Object {

public:

  /** Standard class typedefs */
  typedef MIDASRegionProcessor     Self;
  typedef Object                   Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASRegionProcessor, Object);

  /** Additional typedefs */
  typedef TPixel                                                                       SegmentationPixelType;
  typedef Image<SegmentationPixelType, VImageDimension>                                SegmentationImageType;
  typedef itk::ImageUpdateProcessor<SegmentationPixelType, VImageDimension>            ProcessorType;
  typedef typename ProcessorType::Pointer                                              ProcessorPointer;
  typedef itk::MIDASRegionOfInterestCalculator<SegmentationPixelType, VImageDimension> CalculatorType;
  typedef typename CalculatorType::Pointer                                             CalculatorPointer;

  /** Undo and Redo execute the main method by delegating to the processor, which actually supports undo, redo. */
  void Undo();
  void Redo();

  /** Set/Get the destination image, which is passed to the contained processor. */
  void SetDestinationImage(SegmentationImageType* image) { m_Processor->SetDestinationImage(image); this->Modified(); }
  SegmentationImageType* GetDestinationImage() const { return m_Processor->GetDestinationImage(); }

  /** This method should set the derived class up, and then call this->Modified(). */
  virtual void SetOrientationAndSlice(itk::ORIENTATION_ENUM orientation, int sliceNumber) = 0;

  /** Set debug flags. */
  virtual void SetDebug(bool b) { itk::Object::SetDebug(b); m_Processor->SetDebug(b); m_Calculator->SetDebug(b); m_Processor->SetDebug(b); this->Modified(); }
  virtual void DebugOn() { this->SetDebug(true); }
  virtual void DebugOff() { this->SetDebug(false); }

protected:

  MIDASRegionProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASRegionProcessor() {}

  CalculatorPointer m_Calculator;
  ProcessorPointer  m_Processor;

private:
  MIDASRegionProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASRegionProcessor.txx"
#endif

#endif // ITKMIDASWIPEPROCESSOR_H
