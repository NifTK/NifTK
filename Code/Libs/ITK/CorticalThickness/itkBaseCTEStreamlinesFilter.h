/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkBaseCTEStreamlinesFilter_h
#define __itkBaseCTEStreamlinesFilter_h

#include "itkImage.h"
#include "itkVector.h"
#include "itkImageToImageFilter.h"
#include "itkVectorInterpolateImageFunction.h"
#include "itkInterpolateImageFunction.h"


namespace itk {
/** 
 * \class BaseCTEStreamlinesFilter
 * \brief Base class for filters that calculate thicknesses based on Laplacian streamlines.
 * 
 * \sa IntegrateStreamlinesFilter
 * \sa RelaxStreamlinesFilter
 * \sa LagrangianInitializedRelaxStreamlinesFilter
 * \sa OrderedTraversalStreamlinesFilter
 */
template < class TImageType, typename TScalarType=double, unsigned int NDimensions=3 > 
class ITK_EXPORT BaseCTEStreamlinesFilter :
  public BaseCTEFilter< TImageType > 
{
public:

  /** Standard "Self" typedef. */
  typedef BaseCTEStreamlinesFilter    Self;
  typedef BaseCTEFilter< TImageType > Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(BaseCTEStreamlinesFilter, BaseCTEFilter);

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
  typedef VectorInterpolateImageFunction<InputVectorImageType
                                         ,TScalarType
                                        >                        VectorInterpolatorType;
  typedef typename VectorInterpolatorType::Pointer               VectorInterpolatorPointer;
  typedef typename VectorInterpolatorType::PointType             VectorInterpolatorPointType;
  typedef InterpolateImageFunction< InputScalarImageType
                                    ,TScalarType >               ScalarInterpolatorType;
  typedef typename ScalarInterpolatorType::Pointer               ScalarInterpolatorPointer;
  typedef typename ScalarInterpolatorType::PointType             ScalarInterpolatorPointType;

  /** Set/Get the Low Voltage threshold, defaults to 0. Only pixels > LowVoltage are solved. */
  itkSetMacro(LowVoltage, InputScalarImagePixelType);
  itkGetMacro(LowVoltage, InputScalarImagePixelType);

  /** Set/Get the High Voltage threshold, defaults to 10000. Only pixels < HighVoltage are solved. */
  itkSetMacro(HighVoltage, InputScalarImagePixelType);
  itkGetMacro(HighVoltage, InputScalarImagePixelType);

  /** 
   * Set the interpolator (so in future, we could have 
   * BSpline vector field interpolation).
   * Default is Nearest Neighbour. 
   */
  itkSetObjectMacro( VectorInterpolator, VectorInterpolatorType );
  itkGetConstObjectMacro( VectorInterpolator, VectorInterpolatorType );

  /** 
   * Set the interpolator for the scalar field, 
   * so we could have Bspline for instance.
   * Default is Nearest Neighbour.
   */
  itkSetObjectMacro( ScalarInterpolator, ScalarInterpolatorType );
  itkGetConstObjectMacro( ScalarInterpolator, ScalarInterpolatorType );
  
protected:
  BaseCTEStreamlinesFilter();
  ~BaseCTEStreamlinesFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** The interpolator for vector field. */
  VectorInterpolatorPointer m_VectorInterpolator;

  /** The interpolator for scalar field. */
  ScalarInterpolatorPointer m_ScalarInterpolator;
  
  /** The low voltage value (see paper), defaults to 0. */
  InputScalarImagePixelType m_LowVoltage;
  
  /** The high voltage value (see paper), defaults to 10000. */
  InputScalarImagePixelType m_HighVoltage;

private:
  
  /**
   * Prohibited copy and assingment. 
   */
  BaseCTEStreamlinesFilter(const Self&); 
  void operator=(const Self&); 

};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBaseCTEStreamlinesFilter.txx"
#endif

#endif
