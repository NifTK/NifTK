/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkUCLHistogram_h
#define __itkUCLHistogram_h

#include "itkHistogram.h"

namespace itk{
namespace Statistics{

/** 
 * \class UCLHistogram 
 * \brief Extends Histogram to provide standard Entropy functions.
 * 
 * \sa Histogram
 */

template < class TMeasurement, unsigned int VMeasurementVectorSize = 1,
           class TFrequencyContainer = DenseFrequencyContainer > 
class ITK_EXPORT UCLHistogram 
  : public Histogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
{
public:

  /** Standard typedefs */
  typedef UCLHistogram                                                         Self;
  typedef Histogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer> Superclass;
  typedef SmartPointer<Self>                                                   Pointer;
  typedef SmartPointer<const Self>                                             ConstPointer;
  typedef double                                                               MeasureType;
  typedef typename Superclass::FrequencyContainerType                          FrequencyContainerType;
  typedef typename Superclass::FrequencyType                                   FrequencyType;
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
