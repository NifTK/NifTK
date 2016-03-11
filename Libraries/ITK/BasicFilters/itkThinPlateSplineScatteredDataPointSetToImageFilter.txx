/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkThinPlateSplineScatteredDataPointSetToImageFilter_txx
#define __itkThinPlateSplineScatteredDataPointSetToImageFilter_txx

#include <itkThinPlateSplineScatteredDataPointSetToImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkProgressReporter.h>
#include <itkThinPlateSplineKernelTransform.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageDuplicator.h>
#include <itkCastImageFilter.h>
#include <itkNumericTraits.h>

#include <vnl/vnl_math.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/vnl_vector.h>
#include <vcl_limits.h>

namespace itk
{

/**
 * Initialize new instance
 */
template< typename TInputPointSet, typename TOutputImage >
ThinPlateSplineScatteredDataPointSetToImageFilter< TInputPointSet, TOutputImage >
::ThinPlateSplineScatteredDataPointSetToImageFilter()
{

  m_KernelTransform = KernelTransformType::New();

  m_Invert = false;
  m_SplineHeightDimension = ImageDimension - 1;
  m_Stiffness = 1.;
}

/**
 * Print out a description of self
 */
template< typename TInputPointSet, typename TOutputImage >
void
ThinPlateSplineScatteredDataPointSetToImageFilter< TInputPointSet, TOutputImage >
::PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "KernelTransform: " << m_KernelTransform.GetPointer() << std::endl;
}


/**
 * Sub-sample the input displacement field and prepare the KernelBase
 * BSpline
 */
template< typename TInputPointSet, typename TOutputImage >
void
ThinPlateSplineScatteredDataPointSetToImageFilter< TInputPointSet, TOutputImage >
::PrepareKernelBaseSpline()
{

  unsigned int i = 0;
  unsigned int iLandmark = 0;

  typedef typename KernelTransformType::InputPointType  InputPointType;
  typedef typename KernelTransformType::OutputPointType OutputPointType;

  OutputIndexType idxBaseline;  // Index to current output pixel
  InputPointType ptBaseline;   // Coordinates of current output pixel

  OutputImageType *outputPtr = this->GetOutput();

  OutputSizeType outSize = outputPtr->GetLargestPossibleRegion().GetSize();

  for ( i=0; i<ImageDimension; i++ )
  {
    idxBaseline[ i ] = 0;
  }

  outputPtr->TransformIndexToPhysicalPoint( idxBaseline, ptBaseline );

  // Need to generate the 'height = min' point set

  const TInputPointSet *inputPointSet = this->GetInput();

  typename LandmarkPointSetType::Pointer targetPointSet = LandmarkPointSetType::New();
  typename LandmarkPointSetType::Pointer sourcePointSet = LandmarkPointSetType::New();

  typename LandmarkPointSetType::PointsContainerPointer  targetLandmarks = targetPointSet->GetPoints();
  typename LandmarkPointSetType::PointsContainerPointer  sourceLandmarks = sourcePointSet->GetPoints();

  typename LandmarkContainer::Iterator pointDataIterator;

  typename LandmarkContainer::ConstIterator itInputPoints;

  itInputPoints = inputPointSet->GetPoints()->Begin();

  while( itInputPoints != inputPointSet->GetPoints()->End() )
  {
    LandmarkPointType landmark = itInputPoints.Value();

    if ( this->GetDebug() )
    {
      std::cout << "Landmark " << iLandmark << " Target: " << landmark;
    }

    targetLandmarks->InsertElement( iLandmark, landmark );

    landmark[ m_SplineHeightDimension ] = ptBaseline[ m_SplineHeightDimension ];
    sourceLandmarks->InsertElement( iLandmark, landmark );

    if ( this->GetDebug() )
    {
      std::cout << " Source: " << landmark << std::endl;
    }

    iLandmark++;
    itInputPoints++;
  }

  m_KernelTransform->GetModifiableTargetLandmarks()->SetPoints( targetLandmarks );
  m_KernelTransform->GetModifiableSourceLandmarks()->SetPoints( sourceLandmarks );

  m_KernelTransform->SetStiffness( this->m_Stiffness );

  std::cout << "Stiffness: " << m_KernelTransform->GetStiffness() << std::endl;

  itkDebugMacro(<< "Before ComputeWMatrix() ");

  m_KernelTransform->ComputeWMatrix();

  itkDebugMacro(<< "After ComputeWMatrix() ");
}

/**
 * GenerateData
 */
template< typename TInputPointSet, typename TOutputImage >
void
ThinPlateSplineScatteredDataPointSetToImageFilter< TInputPointSet, TOutputImage >
::GenerateData()
{
  unsigned int i;

  // Define a few indices that will be used to translate from an input pixel
  // to an output pixel
  OutputIndexType outputIndex;         // Index to current output pixel

  typedef typename KernelTransformType::InputPointType  InputPointType;
  typedef typename KernelTransformType::OutputPointType OutputPointType;

  InputPointType outputPoint;    // Coordinates of current output pixel

  typedef ImageRegionIteratorWithIndex< TOutputImage > OutputIterator;
  typedef itk::ImageLinearIteratorWithIndex< TOutputImage > LineIteratorType;

  itkDebugMacro(<< "Actually executing");

  // Get the output pointers
  OutputImageType *outputPtr = this->GetOutput();

  for ( i=0; i<ImageDimension; i++ )
  {
    if( this->m_Size[i] == 0 )
    {
      itkExceptionMacro("Size must be specified.");
    }
  }

  outputPtr->SetOrigin(    this->m_Origin );
  outputPtr->SetSpacing(   this->m_Spacing );
  outputPtr->SetDirection( this->m_Direction );
  outputPtr->SetRegions(   this->m_Size );

  outputPtr->Allocate();
  outputPtr->FillBuffer( 0 );

  // First subsample the input displacement field in order to create
  // the KernelBased spline.
  this->PrepareKernelBaseSpline();

  // Create an iterator that will generate the indices for which the
  // height of the spline will be calculated
  OutputImageRegionType region;
  OutputSizeType size;

  region = outputPtr->GetRequestedRegion();
  size = region.GetSize();

  size[ m_SplineHeightDimension ] = 1; // We're only interested in the xy plane (3D) or x (2D)
  region.SetSize( size );

  OutputIterator outIt(outputPtr, region);

  outIt.GoToBegin();

  // Support for progress methods/callbacks
  ProgressReporter progress(this, 0, region.GetNumberOfPixels(), 10);

  // Walk the output region

  OutputPointType interpolatedDisplacement;

  region = outputPtr->GetRequestedRegion();
  size = region.GetSize();

  for ( i=0; i<ImageDimension; i++ )
  {
    if ( i != m_SplineHeightDimension )
    {
      size[ i ] = 1;              // Single column of voxels to set the height in the mask
    }
  }

  region.SetSize( size );

  while ( !outIt.IsAtEnd() )
  {
    // Determine the index of the current output pixel
    outputIndex = outIt.GetIndex();
    outputPtr->TransformIndexToPhysicalPoint( outputIndex, outputPoint );

    if ( this->GetDebug() )
    {
      std::cout << "Index: " << outputIndex << "  Point: " << outputPoint;
    }

    // Compute corresponding spline height
    interpolatedDisplacement = m_KernelTransform->TransformPoint( outputPoint );

    region.SetIndex( outputIndex );
    LineIteratorType itHeight( outputPtr, region );

    itHeight.SetDirection( m_SplineHeightDimension );

    outputPtr->TransformPhysicalPointToIndex( interpolatedDisplacement, outputIndex );
    
    itHeight.GoToBegin();
    itHeight.GoToBeginOfLine();

    if ( this->GetDebug() )
    {
      std::cout << "   Displacement: " << interpolatedDisplacement 
                << "   Index: " << outputIndex << std::endl;
    }

    for ( i=0; 
          i<=outputIndex[ m_SplineHeightDimension ] && ( ! itHeight.IsAtEndOfLine() );
          i++, ++itHeight )
    {
      itHeight.Set( 1 );
    }


    ++outIt;
    progress.CompletedPixel();
  }


  // Invert the mask?

  if ( m_Invert )
  {
    typedef typename itk::InvertIntensityBetweenMaxAndMinImageFilter<OutputImageType> InvertFilterType;
    typename InvertFilterType::Pointer invertFilter = InvertFilterType::New();
    
    invertFilter->SetInput( outputPtr );
    
    invertFilter->Update( );

    this->GraftOutput( invertFilter->GetOutput() );
  }

}


/**
 * Verify if any of the components has been modified.
 */
template< typename TInputPointSet, typename TOutputImage >
ModifiedTimeType
ThinPlateSplineScatteredDataPointSetToImageFilter< TInputPointSet, TOutputImage >
::GetMTime(void) const
{
  ModifiedTimeType latestTime = Object::GetMTime();

  if ( m_KernelTransform )
  {
    if ( latestTime < m_KernelTransform->GetMTime() )
    {
      latestTime = m_KernelTransform->GetMTime();
    }
  }

  return latestTime;
}

} // end namespace itk

#endif
