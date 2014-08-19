/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatEstimationFitMetricForArray_h
#define __itkMammogramFatEstimationFitMetricForArray_h

#include <itkMammogramFatEstimationFitMetric.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

namespace itk {
  
/** \class MammogramFatEstimationFitMetricForArray
 * \brief A metric to compute the similarity between an image and breast fat model.
 *
 * Computes the similarity to a shape model:
 *
 * y = {x < 0: 0}, {0 < x < a: b/a sqrt(a^2 - x^2)}, {x > a: b}
 *
 * \section itkMammogramFatEstimationFitMetricForArrayCaveats Caveats
 * \li None
 */

template <class TInputImage>
class  ITK_EXPORT MammogramFatEstimationFitMetricForArray :
  public MammogramFatEstimationFitMetric
{
//  Software Guide : EndCodeSnippet
public:
  typedef MammogramFatEstimationFitMetricForArray  Self;
  typedef MammogramFatEstimationFitMetric   Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( MammogramFatEstimationFitMetricForArray, MammogramFatEstimationFitMetric );

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  itkStaticConstMacro( ParametricSpaceDimension, unsigned int, 4 );

  virtual unsigned int GetNumberOfParameters(void) const  
  {
    return ParametricSpaceDimension;
  }

  typedef float DistancePixelType;

  /** Set the number of distances. */
  void SetNumberOfDistances( unsigned int nDistances ) {
    m_NumberOfDistances = nDistances;
  }

  /** Connect the input array. */
  void SetInputArray( float *minIntensityVsEdgeDistance ) {
    m_MinIntensityVsEdgeDistance = minIntensityVsEdgeDistance;
  }

  /** Get the maximum distance to the breast edge in mm. */
  DistancePixelType GetMaxDistance( void ) { return m_MaxDistance; }
  /** Get the maximum distance to the breast edge in mm. */
  void SetMaxDistance( DistancePixelType distance ) { m_MaxDistance = distance; }


  MeasureType GetValue( const ParametersType &parameters ) const;

  void WriteIntensityVsEdgeDistToFile( std::string fileOutputIntensityVsEdgeDist );
  void WriteFitToFile( std::string fileOutputFit,
                       const ParametersType &parameters );

  void GenerateFatArray( unsigned int nDistances, float *fatEstimate, 
                         const ParametersType &parameters );

protected:

  MammogramFatEstimationFitMetricForArray();
  virtual ~MammogramFatEstimationFitMetricForArray();
  MammogramFatEstimationFitMetricForArray(const Self &) {}
  void operator=(const Self &) {}
  void PrintSelf(std::ostream & os, Indent indent) const;

  unsigned int m_NumberOfDistances;
  float *m_MinIntensityVsEdgeDistance;

  DistancePixelType      m_MaxDistance;

  double CalculateFit( double d, const ParametersType &parameters );

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramFatEstimationFitMetricForArray.txx"
#endif

#endif
