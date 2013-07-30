/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMultiResolutionBlockMatchingMethod_h
#define itkMultiResolutionBlockMatchingMethod_h

#include "itkMultiResolutionImageRegistrationWrapper.h"
#include <ConversionUtils.h>
#include "itkBlockMatchingMethod.h"

namespace itk
{

/** 
 * \class MultiResolutionBlockMatchingMethod
 * \brief Extends MultiResolutionImageRegistrationWrapper to provide setting up block matching method
 * at each resolution level, which currently just means setting the percentage of points to keep.
 * 
 * \sa MultiResolutionImageRegistrationWrapper
 */
template <typename TInputImageType, class TScalarType>
class ITK_EXPORT MultiResolutionBlockMatchingMethod 
  : public MultiResolutionImageRegistrationWrapper<TInputImageType> 
{
public:
  /** Standard class typedefs. */
  typedef MultiResolutionBlockMatchingMethod                        Self;
  typedef MultiResolutionImageRegistrationWrapper<TInputImageType>  Superclass;
  typedef SmartPointer<Self>                                        Pointer;
  typedef SmartPointer<const Self>                                  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionBlockMatchingMethod, MultiResolutionImageRegistrationWrapper);

  /** So we can set things on the block matching method. */
  typedef BlockMatchingMethod<TInputImageType, TScalarType>                        BlockMatchingType;
  typedef BlockMatchingType*                                                       BlockMatchingPointer;
  typedef typename BlockMatchingMethod<TInputImageType, TScalarType>::PointSetType PointSetType;
  
  /** Set/Get the VarianceRejectionInitialPercentage. Default 50% */
  itkSetMacro(VarianceRejectionInitialPercentage, int);
  itkGetMacro(VarianceRejectionInitialPercentage, int);

  /** Set/Get the VarianceRejectionPercentageMultiplier. Default 0.5 */
  itkSetMacro(VarianceRejectionPercentageMultiplier, double);
  itkGetMacro(VarianceRejectionPercentageMultiplier, double);

  /** Set/Get the VarianceRejectionLowerPercentageLimit. Default 20% */
  itkSetMacro(VarianceRejectionLowerPercentageLimit, int);
  itkGetMacro(VarianceRejectionLowerPercentageLimit, int);

  /** Set/Get the DistanceRejectionInitialPercentage. Default 50% */
  itkSetMacro(DistanceRejectionInitialPercentage, int);
  itkGetMacro(DistanceRejectionInitialPercentage, int);

  /** Set/Get the DistanceRejectionPercentageMultiplier. Default 0.5 */
  itkSetMacro(DistanceRejectionPercentageMultiplier, double);
  itkGetMacro(DistanceRejectionPercentageMultiplier, double);

  /** Set/Get the DistanceRejectionLowerPercentageLimit. Default 20% */
  itkSetMacro(DistanceRejectionLowerPercentageLimit, int);
  itkGetMacro(DistanceRejectionLowerPercentageLimit, int);

protected:
  MultiResolutionBlockMatchingMethod();
  virtual ~MultiResolutionBlockMatchingMethod() {};

  /** Set stuff up. */
  virtual void PreparePyramids() throw (ExceptionObject);
  
  /** Called after registration. */
  virtual void AfterSingleResolutionRegistration();
  
private:
  MultiResolutionBlockMatchingMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** Initial percentage of points to keep based on variance. */
  int m_VarianceRejectionInitialPercentage;
  
  /** Reduction factor. */
  double m_VarianceRejectionPercentageMultiplier;
  
  /** The lower limit. */
  int m_VarianceRejectionLowerPercentageLimit;

  /** Initial percentage of points to keep based on distance. */
  int m_DistanceRejectionInitialPercentage;
  
  /** Reduction factor. */
  double m_DistanceRejectionPercentageMultiplier;
  
  /** The lower limit. */
  int m_DistanceRejectionLowerPercentageLimit;

  /** The current distance based percentage. */
  int m_CurrentDistancePercentage;
  
  /** The current variance based percentage. */
  int m_CurrentVariancePercentage;
  
  /** Sets the percentage of blocks to keep on the single res method. */
  void SetVariancePercentage(int percentage);
  
  /** Sets the percentage of points to trim when doing Least Trimmed Squares. */
  void SetDistancePercentage(int percentage);
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiResolutionBlockMatchingMethod.txx"
#endif

#endif



