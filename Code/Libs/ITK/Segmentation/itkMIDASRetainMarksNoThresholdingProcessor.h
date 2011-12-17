/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-15 07:06:41 +0100 (Sat, 15 Oct 2011) $
 Revision          : $Revision: 7522 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKMIDASRETAINMARKSNOTHRESHOLDINGPROCESSOR_H
#define ITKMIDASRETAINMARKSNOTHRESHOLDINGPROCESSOR_H

#include "itkObject.h"
#include "itkImageUpdateCopyRegionProcessor.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionOfInterestCalculator.h"

namespace itk
{

/**
 * \class MIDASRetainMarksNoThresholdingProcessor
 * \brief Class to support the MIDAS Retain marks operation, specifically when thresholding is off,
 * which means it copies from a given slice in the source image, to a given slice in the destination image.
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASRetainMarksNoThresholdingProcessor : public Object {

public:

  /** Standard class typedefs */
  typedef MIDASRetainMarksNoThresholdingProcessor Self;
  typedef Object                                  Superclass;
  typedef SmartPointer<Self>                      Pointer;
  typedef SmartPointer<const Self>                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASRetainMarksNoThresholdingProcessor, Object);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef itk::ImageUpdateCopyRegionProcessor<TPixel, VImageDimension> ProcessorType;
  typedef typename ProcessorType::Pointer ProcessorPointer;
  typedef itk::MIDASRegionOfInterestCalculator<TPixel, VImageDimension> CalculatorType;
  typedef typename CalculatorType::Pointer CalculatorPointer;
  typedef typename ImageType::RegionType RegionType;

  /** Undo and Redo simply call the contained processor, which must be correctly setup at that point. */
  void Undo() { m_Processor->Undo(); }
  void Redo() { m_Processor->Redo(); }

  /** Set/Get the source image, which is passed to the contained processor. */
  void SetSourceImage(ImageType* image) { m_Processor->SetSourceImage(image); this->Modified(); }
  ImageType* GetSourceImage() const { return m_Processor->GetSourceImage(); }

  /** Set/Get the destination image, which is passed to the contained processor. */
  void SetDestinationImage(ImageType* image) { m_Processor->SetDestinationImage(image); this->Modified(); }
  ImageType* GetDestinationImage() const { return m_Processor->GetDestinationImage(); }

  /** This method sets up the regions of interest from the sourceSliceNumber to the targetSliceNumber. */
  virtual void SetSlices(itk::ORIENTATION_ENUM orientation, int sourceSliceNumber, int targetSliceNumber);

  /** Set debug flags */
  virtual void SetDebug(bool b) { itk::Object::SetDebug(b); m_Processor->SetDebug(b); m_Calculator->SetDebug(b); }
  virtual void DebugOn() { this->SetDebug(true); }
  virtual void DebugOff() { this->SetDebug(false); }

protected:

  MIDASRetainMarksNoThresholdingProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASRetainMarksNoThresholdingProcessor() {}

  ProcessorPointer m_Processor;
  CalculatorPointer m_Calculator;

private:
  MIDASRetainMarksNoThresholdingProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASRetainMarksNoThresholdingProcessor.txx"
#endif

#endif // ITKMIDASRETAINMARKSNOTHRESHOLDINGPROCESSOR_H
