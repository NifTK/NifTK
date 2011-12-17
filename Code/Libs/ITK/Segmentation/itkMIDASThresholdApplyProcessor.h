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

#ifndef ITKMIDASTHRESHOLDAPPLYPROCESSOR_H
#define ITKMIDASTHRESHOLDAPPLYPROCESSOR_H

#include "itkObject.h"
#include "itkImageUpdateCopyRegionProcessor.h"
#include "itkImageUpdateClearRegionProcessor.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionOfInterestCalculator.h"

namespace itk
{

/**
 * \class MIDASThresholdApplyProcessor
 * \brief Class to support the MIDAS Threshold Apply operation, which copies the currently
 * thresholded region (in blue), into the target image (normally green).
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT MIDASThresholdApplyProcessor : public Object {

public:

  /** Standard class typedefs */
  typedef MIDASThresholdApplyProcessor            Self;
  typedef Object                                  Superclass;
  typedef SmartPointer<Self>                      Pointer;
  typedef SmartPointer<const Self>                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MIDASThresholdApplyProcessor, Object);

  /** Additional typedefs */
  typedef TPixel PixelType;
  typedef Image<TPixel, VImageDimension>                                ImageType;
  typedef typename ImageType::Pointer                                   ImagePointer;
  typedef itk::ImageUpdateCopyRegionProcessor<TPixel, VImageDimension>  CopyProcessorType;
  typedef typename CopyProcessorType::Pointer                           CopyProcessorPointer;
  typedef itk::ImageUpdateClearRegionProcessor<TPixel, VImageDimension> ClearProcessorType;
  typedef typename ClearProcessorType::Pointer                          ClearProcessorPointer;
  typedef itk::MIDASRegionOfInterestCalculator<TPixel, VImageDimension> CalculatorType;
  typedef typename CalculatorType::Pointer                              CalculatorPointer;
  typedef typename ImageType::RegionType                                RegionType;

  /** Undoes the command. */
  void Undo();

  /** Executes the command. */
  void Redo();

  /** Set/Get the source image. */
  itkSetObjectMacro(SourceImage, ImageType);
  itkGetObjectMacro(SourceImage, ImageType);

  /** Set/Get the destination image. */
  itkSetObjectMacro(DestinationImage, ImageType);
  itkGetObjectMacro(DestinationImage, ImageType);

  /** This should be called before Undo/Redo, as it sets up the correct (i.e. minimal region of interest). */
  void CalculateRegionOfInterest();

  /** Set debug flags */
  virtual void SetDebug(bool b) { itk::Object::SetDebug(b); m_CopyProcessor->SetDebug(b); m_ClearProcessor->SetDebug(b); m_Calculator->SetDebug(b); }
  virtual void DebugOn() { this->SetDebug(true); }
  virtual void DebugOff() { this->SetDebug(false); }

protected:

  MIDASThresholdApplyProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~MIDASThresholdApplyProcessor() {}

  CopyProcessorPointer m_CopyProcessor;
  ClearProcessorPointer m_ClearProcessor;
  CalculatorPointer m_Calculator;

  ImagePointer m_SourceImage;
  ImagePointer m_DestinationImage;

private:
  MIDASThresholdApplyProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMIDASThresholdApplyProcessor.txx"
#endif

#endif // ITKMIDASTHRESHOLDAPPLYPROCESSOR_H
