/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUCLHistogram_h
#define itkUCLHistogram_h

#include <itkHistogram.h>

namespace itk{
namespace Statistics{

/** 
 * \class UCLHistogram 
 * \brief Extends Histogram to provide standard Entropy functions.
 * 
 * \sa Histogram
 */

template < class TMeasurement, unsigned int VMeasurementVectorSize = 1,
           class TFrequencyContainer = DenseFrequencyContainer2 > 
class ITK_EXPORT UCLHistogram 
  : public Histogram<TMeasurement, TFrequencyContainer>
{
public:

  /** Standard typedefs */
  typedef UCLHistogram                                                         Self;
  typedef Histogram<TMeasurement, TFrequencyContainer> Superclass;
  typedef SmartPointer<Self>                                                   Pointer;
  typedef SmartPointer<const Self>                                             ConstPointer;
  typedef double                                                               MeasureType;
  typedef typename Superclass::FrequencyContainerType                          FrequencyContainerType;
  typedef typename Superclass::AbsoluteFrequencyType                           FrequencyType;
  typedef typename Superclass::IndexType                                       IndexType;
  typedef typename Superclass::SizeType                                        SizeType;
  typedef typename Superclass::ConstIterator                                   IteratorType;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(UCLHistogram, Histogram) ;

  /** standard New() method support */
  itkNewMacro(Self) ;

  /** Dimension of a measurement vector */
  itkStaticConstMacro(MeasurementVectorSize, unsigned int, VMeasurementVectorSize);
 
  /** Returns the mean in the x-direction. */
  MeasureType MeanFixed() const;

  /** Returns the mean in the y-direction. */
  MeasureType MeanMoving() const;

  /** Returns the variance in the x-direction. */
  MeasureType VarianceFixed() const;

  /** Returns the variance in the y-direction. */
  MeasureType VarianceMoving() const;

  /** Returns the co-variance. */
  MeasureType Covariance() const;
  
  /** Returns the entropy in the fixed direction. */
  MeasureType EntropyFixed() const;
  
  /** Returns the entropy in the moving direction. */
  MeasureType EntropyMoving() const;
  
  /** Returns joint entropy. */
  MeasureType JointEntropy() const;

protected:
  UCLHistogram() ;
  virtual ~UCLHistogram() {}
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  UCLHistogram(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
}; 

} // end of namespace Statistics 
} // end of namespace itk 

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkUCLHistogram.txx"
#endif

#endif
