/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkHistogramSimilarityMeasure_h
#define __itkHistogramSimilarityMeasure_h

#include "itkUCLHistogram.h"
#include "itkFiniteDifferenceGradientSimilarityMeasure.h"


namespace itk
{
/** 
 * \class HistogramSimilarityMeasure 
 * \brief Computes similarity between two objects to be registered using Histogram.
 * 
 * As of 20090126, provides support for filling histogram in a Parzen window type 
 * approach. However, the code is a bit simple for now, and we assume that the
 * image has the same number of intensity values as the histogram has bins.
 * So you MUST rescale your image to fit the histogram first.
 */
template <class TFixedImage, class TMovingImage>
class ITK_EXPORT HistogramSimilarityMeasure : 
public FiniteDifferenceGradientSimilarityMeasure<TFixedImage, TMovingImage>
{
public:
  
  /** Standard class typedefs. */
  typedef HistogramSimilarityMeasure                      Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage>    Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(HistogramSimilarityMeasure, SimilarityMeasure);
 
  /** Types transferred from the base class */
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  
  /** 
   * Typedefs for histogram. 
   */
  typedef Statistics::UCLHistogram<double, 2>             HistogramType;
  typedef typename HistogramType::Pointer                 HistogramPointer;
  typedef typename HistogramType::SizeType                HistogramSizeType;
  typedef typename HistogramType::MeasurementVectorType   HistogramMeasurementVectorType;
  typedef typename HistogramType::FrequencyType           HistogramFrequencyType;
  typedef typename HistogramType::Iterator                HistogramIteratorType;
  
  /** Initializes the metric. */
  void Initialize() throw (ExceptionObject);

  /** Overload this method, to make it easier for users. */
  void SetHistogramSize(int x, int y)
    {
      // The size of this vector is templated in base class.
      niftkitkDebugMacro(<< "HistogramSimilarityMeasure():Setting size to [" << niftk::ConvertToString((int)x) \
          << "," << niftk::ConvertToString((int)y) << "]");
      
      HistogramSizeType size;
      size[0] = x;
      size[1] = y;
      this->SetHistogramSize(size);
    };

  /**
   * Sets the histogram size. Note this function must be called before
   * \c Initialize().
   */
  itkSetMacro( HistogramSize, HistogramSizeType );

  /** Gets the histogram size. */
  itkGetConstReferenceMacro( HistogramSize, HistogramSizeType );

  /** 
   * Return the joint histogram. This is updated during every call to the 
   * GetValue() method and the GetDerivative() method.  
   */
  itkGetObjectMacro( Histogram, HistogramType );

  /** 
   * If set to true, we fill histogram with a gaussian per sample. Default false.
   * NOTE: For now, for this to work properly, the image must have the same
   * number of intensity values as the histogram. 
   */
  itkSetMacro(UseParzenFilling, bool);
  itkGetMacro(UseParzenFilling, bool);

  /** Returns the Parzen value. */
  double GetParzenValue(double x);
  
  /** Returns the Parzen derivative. */
  double GetParzenDerivative(double x);

protected:
  
  HistogramSimilarityMeasure();
  virtual ~HistogramSimilarityMeasure() {};

  /** The histogram size. */
  HistogramSizeType m_HistogramSize;

  /**
   * Called at the start of each evaluation. 
   */
  void ResetCostFunction() 
    {
      HistogramMeasurementVectorType lowerBounds;
      lowerBounds[0] = this->GetFixedLowerBound();
      lowerBounds[1] = this->GetMovingLowerBound();
      HistogramMeasurementVectorType upperBounds;
      upperBounds[0] = this->GetFixedUpperBound();
      upperBounds[1] = this->GetMovingUpperBound();
      
//      niftkitkDebugMacro(<< "HistogramSimilarityMeasure():Resetting histogram to size:" << this->m_HistogramSize << ", lowerBounds:" << lowerBounds << ", upperBounds:" << upperBounds);
      this->m_Histogram->Initialize( this->m_HistogramSize, lowerBounds, upperBounds );
      this->m_Histogram->SetToZero();
    }

  /** 
   * Use this method to add corresponding pairs of image values,
   * called repeatedly during a single value of the cost function.
   */
  void AggregateCostFunctionPair(FixedImagePixelType fixedValue, MovingImagePixelType movingValue);
  
  /** PrintSelf funtion */
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Returns the mean in the x-direction. */
  MeasureType MeanFixed() const { return this->m_Histogram->MeanFixed(); }

  /** Returns the mean in the y-direction. */
  MeasureType MeanMoving() const { return this->m_Histogram->MeanMoving(); }

  /** Returns the variance in the x-direction. */
  MeasureType VarianceFixed() const { return this->m_Histogram->VarianceFixed(); }

  /** Returns the variance in the y-direction. */
  MeasureType VarianceMoving() const { return this->m_Histogram->VarianceMoving(); }

  /** Returns the co-variance. */
  MeasureType Covariance() const { return this->m_Histogram->Covariance(); }
  
  /** Returns the entropy in the fixed direction. */
  MeasureType EntropyFixed() const { return this->m_Histogram->EntropyFixed(); }
  
  /** Returns the entropy in the moving direction. */
  MeasureType EntropyMoving() const { return this->m_Histogram->EntropyMoving(); }
  
  /** Returns joint entropy. */
  MeasureType JointEntropy() const { return this->m_Histogram->JointEntropy(); }

  /**
   * Pointer to the joint histogram.
   * This is updated during every call to GetValue() 
   */
  HistogramPointer  m_Histogram;

private:
  HistogramSimilarityMeasure(const Self&);  // purposefully not implemented
  void operator=(const Self&);              // purposefully not implemented

  /** Turn Parzen filling on/off. default off.*/
  bool m_UseParzenFilling;
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHistogramSimilarityMeasure.txx"
#endif

#endif // __itkHistogramSimilarityMeasure_h
