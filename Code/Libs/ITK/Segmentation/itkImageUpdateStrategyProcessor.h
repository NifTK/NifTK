/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 12:21:57 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7491 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef ITKIMAGEUPDATESTATEGYPROCESSOR_H
#define ITKIMAGEUPDATESTATEGYPROCESSOR_H

#include "itkImageUpdateProcessor.h"
#include "itkImageUpdateStrategyAlgorithm.h"
#include "itkMIDASRegionOfInterestCalculator.h"

namespace itk
{

/**
 * \class ImageUpdateStrategyProcessor
 * \brief Class to support undo/redo of any image processing
 * algorithm that can be implemented via a Strategy pattern.
 *
 * So the algorithm must be supplied via dependency injection. This means
 * the steps are:
 *
 * <pre>
 * 1. Create subclass of ImageUpdateStrategyAlgorithm, setting any member variables the algorithm needs
 * 2. Create this class.
 * 3. Inject the subclass of ImageUpdateStrategyAlgorithm using SetAlgorithm method into this class.
 * 4. Set the destination image on this class (defined in ImageUpdateProcessor).
 * 5. Set the destination region of interest on this class (defined in ImageUpdateProcessor).
 * 6. Call Redo/Undo, which will backup the before and after image, and call Execute on the algorithm.
 * </pre>
 *
 * \sa ImageUpdateProcessor
 * \sa ImageUpdateStrategyAlgorithm
 * \sa ImageUpdateSliceBasedRegionGrowingAlgorithm
 */
template <class TPixel, unsigned int VImageDimension>
class ITK_EXPORT ImageUpdateStrategyProcessor : public ImageUpdateProcessor<TPixel, VImageDimension> {

public:

  /** Standard class typedefs */
  typedef ImageUpdateStrategyProcessor                  Self;
  typedef ImageUpdateProcessor<TPixel, VImageDimension> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Additional typedefs, as this is the ITK way. */
  typedef itk::ImageUpdateStrategyAlgorithm<TPixel, VImageDimension>          AlgorithmType;
  typedef typename AlgorithmType::Pointer                                     AlgorithmPointer;
  typedef itk::MIDASRegionOfInterestCalculator<TPixel, VImageDimension>       CalculatorType;
  typedef typename CalculatorType::Pointer                                    CalculatorPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageUpdateStrategyProcessor, ImageUpdateProcessor);

  /** Set/Get the actual algorithm that is executed. */
  itkSetObjectMacro(Algorithm, AlgorithmType);
  itkGetObjectMacro(Algorithm, AlgorithmType);

  /** Additional typedefs */
  typedef TPixel                          PixelType;
  typedef Image<TPixel, VImageDimension>  ImageType;
  typedef typename ImageType::Pointer     ImagePointer;
  typedef typename ImageType::RegionType  RegionType;

protected:
  ImageUpdateStrategyProcessor();
  void PrintSelf(std::ostream& os, Indent indent) const;
  virtual ~ImageUpdateStrategyProcessor() {}

  // This class simply delegates to the supplied algorithm, and calls Execute on it.
  virtual void ApplyUpdateToAfterImage();

private:
  ImageUpdateStrategyProcessor(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  AlgorithmPointer m_Algorithm;
  CalculatorPointer m_Calculator;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageUpdateStrategyProcessor.txx"
#endif

#endif // ITKIMAGEUPDATESTATEGYPROCESSOR_H
