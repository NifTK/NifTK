/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMIImageToImageMetric_h
#define __itkMImageToImageMetric_h

#include "itkHistogramSimilarityMeasure.h"

namespace itk
{
/** 
 * \class MIImageToImageMetric
 * \brief Implements Mutual Information of a histogram for a similarity measure.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT MIImageToImageMetric : 
    public HistogramSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef MIImageToImageMetric                                            Self;
  typedef HistogramSimilarityMeasure<TFixedImage, TMovingImage>           Superclass;
  typedef SmartPointer<Self>                                              Pointer;
  typedef SmartPointer<const Self>                                        ConstPointer;
  typedef typename Superclass::FixedImageType::PixelType                  FixedImagePixelType;
  typedef typename Superclass::MovingImageType::PixelType                 MovingImagePixelType;  
  typedef typename Superclass::MeasureType                                MeasureType; 
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(MIImageToImageMetric, HistogramSimilarityMeasure);

  /** MI should be Maximized. */
  bool ShouldBeMaximized() { return true; };
  
protected:
  
  MIImageToImageMetric() {};
  virtual ~MIImageToImageMetric() {};

  /**
   * In this method, we do any final aggregating,
   * which basically means "evaluate the histogram".
   * Filling the histogram is in the base class.
   */
  MeasureType FinalizeCostFunction()
    {
      return this->EntropyFixed() + this->EntropyMoving() - this->JointEntropy();
    }

private:
  MIImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);       // purposefully not implemented  
};

} // end namespace itk

#endif



