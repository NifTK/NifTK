/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkIntegrateStreamlinesFilter_h
#define __itkIntegrateStreamlinesFilter_h

#include <itkImage.h>
#include <itkVector.h>
#include "itkBaseCTEStreamlinesFilter.h"
#include <itkLinearInterpolateImageFunction.h>
#include <itkVectorLinearInterpolateImageFunction.h>

namespace itk {
/** 
 * \class IntegrateStreamlinesFilter
 * \brief Integrates streamlines using Eulers method (Lagrangian framework).
 * 
 * This filter implements step (9) in Jones et al. Human Brain Mapping
 * 11:12-32 (2000). The first input, set using SetScalarImage 
 * should be the output of step (7), which is the output of 
 * LaplacianSolverImageFilter, i.e. a scalar image representing voltage 
 * potentials.  The second input should be the output of step (8) which 
 * is a vector field of normalized gradient, set using SetVectorImage.
 * This filter has a min and a max voltage threshold,
 * and for any voxel that is between those values, it will integrate
 * along the gradient vector in both directions, until it hits the
 * threshold, thus computing the streamline length between the two
 * surfaces (read the paper!). Hence you can set the integration
 * step size, and also a hard limit to force the iterating to stop 
 * once it hits a maximum distance.
 * 
 * \sa BaseStreamlinesFilter
 * \sa LaplacianSolverImageFilter
 * \sa RelaxStreamlinesFilter
 * \sa OrderedTraversalStreamlinesFilter 
 */
template < class TImageType, typename TScalarType=double, unsigned int NDimensions=3 > 
class ITK_EXPORT IntegrateStreamlinesFilter :
  public BaseCTEStreamlinesFilter<TImageType, TScalarType, NDimensions>
{
public:

  /** Standard "Self" typedef. */
  typedef IntegrateStreamlinesFilter                                     Self;
  typedef BaseCTEStreamlinesFilter<TImageType, TScalarType, NDimensions> Superclass;
  typedef SmartPointer<Self>                                             Pointer;
  typedef SmartPointer<const Self>                                       ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(IntegrateStreamlinesFilter, BaseCTEStreamlinesFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Standard typedefs. */
  typedef Vector< TScalarType, NDimensions >                     InputVectorImagePixelType;
  typedef Image< InputVectorImagePixelType, NDimensions >        InputVectorImageType;
  typedef typename InputVectorImageType::Pointer                 InputVectorImagePointer;
  typedef typename InputVectorImageType::ConstPointer            InputVectorImageConstPointer; 
  typedef TScalarType                                            InputScalarImagePixelType;
  typedef Image< InputScalarImagePixelType, NDimensions >        InputScalarImageType;
  typedef typename InputScalarImageType::PointType               InputScalarImagePointType;
  typedef typename InputScalarImageType::Pointer                 InputScalarImagePointer;
  typedef typename InputScalarImageType::IndexType               InputScalarImageIndexType;
  typedef typename InputScalarImageType::ConstPointer            InputScalarImageConstPointer;
  typedef typename InputScalarImageType::RegionType              InputScalarImageRegionType;
  typedef InputScalarImageType                                   OutputImageType;
  typedef typename OutputImageType::PixelType                    OutputImagePixelType;
  typedef typename OutputImageType::Pointer                      OutputImagePointer;
  typedef typename OutputImageType::ConstPointer                 OutputImageConstPointer;
  typedef VectorLinearInterpolateImageFunction<
                                          InputVectorImageType
                                         ,TScalarType
                                              >                  VectorInterpolatorType;
  typedef typename VectorInterpolatorType::Pointer               VectorInterpolatorPointer;
  typedef typename VectorInterpolatorType::PointType             VectorInterpolatorPointType;
  typedef LinearInterpolateImageFunction< 
                                     InputScalarImageType
                                    ,TScalarType 
                                        >                        ScalarInterpolatorType;
  typedef typename ScalarInterpolatorType::Pointer               ScalarInterpolatorPointer;
  typedef typename ScalarInterpolatorType::PointType             ScalarInterpolatorPointType;

  /** Sets the scalar (Laplacian) image, at input 0. */
  void SetScalarImage(const InputScalarImageType *image) {this->SetNthInput(0, const_cast<InputScalarImageType *>(image)); }

  /** Sets the vector image, at input 1. */
  void SetVectorImage(const InputVectorImageType* image) { this->SetNthInput(1, const_cast<InputVectorImageType *>(image)); }
                                        
  /** Set/get the minimum threshold we iterate towards, should really be the same as LowVoltage, but here you can tweak where integration stops. */
  itkSetMacro(MinIterationVoltage, InputScalarImagePixelType);
  itkGetMacro(MinIterationVoltage, InputScalarImagePixelType);

  /** Set/get the maximum threshold we iterate towards, should really be the same as HighVoltage, but here you can tweak where integration stops. */
  itkSetMacro(MaxIterationVoltage, InputScalarImagePixelType);
  itkGetMacro(MaxIterationVoltage, InputScalarImagePixelType);

  /** Set/Get the Step Size for integration, defaults to 0.001. */
  itkSetMacro(StepSize, TScalarType);
  itkGetMacro(StepSize, TScalarType);

  /** Set/Get the maximum length for integration, defaults to 10. Used to force filter to stop iterating. */
  itkSetMacro(MaxIterationLength, TScalarType);
  itkGetMacro(MaxIterationLength, TScalarType);

protected:
  IntegrateStreamlinesFilter();
  ~IntegrateStreamlinesFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  // The main method to implement in derived classes, note, its threaded.
  virtual void ThreadedGenerateData( const InputScalarImageRegionType &outputRegionForThread, int);

private:
  
  /**
   * Prohibited copy and assingment. 
   */
  IntegrateStreamlinesFilter(const Self&); 
  void operator=(const Self&); 

  /** The low voltage that we iterate towards, defaults to 0.  */
  InputScalarImagePixelType m_MinIterationVoltage;
  
  /** The high voltage that we iterate towards, defaults to 10000. */
  InputScalarImagePixelType m_MaxIterationVoltage;
  
  /** The integration step size, defaults 0.01 */
  TScalarType m_StepSize;

  /** To force the integration to stop. Defaults to 10. */
  TScalarType m_MaxIterationLength;

  /** Internal method that actually does the integrating. */
  double GetLengthToThreshold(
                              const InputScalarImagePointType &startingPoint,
                              const InputScalarImagePixelType &initialValue,
                              const InputScalarImagePixelType &threshold,
                              const InputVectorImagePointer   &vectorImage,
                              const InputScalarImagePointer   &scalarImage,
                              const VectorInterpolatorPointer &vectorInterpolator,
                              const ScalarInterpolatorPointer &scalarInterpolator,
                              const double &multiplier, // 1 or -1
                              const bool &debug,
                              bool &maxLengthExceeded
                             );
                              
  
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkIntegrateStreamlinesFilter.txx"
#endif

#endif
