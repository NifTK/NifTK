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
  unsigned int iLandmark = 0;

  // Need to generate the 'z=0' point set

  const TInputPointSet *inputPointSet = this->GetInput();

  typename LandmarkPointSetType::Pointer targetPointSet = LandmarkPointSetType::New();
  typename LandmarkPointSetType::Pointer sourcePointSet = LandmarkPointSetType::New();

  typename LandmarkPointSetType::PointsContainerPointer  targetLandmarks = targetPointSet->GetPoints();
  typename LandmarkPointSetType::PointsContainerPointer  sourceLandmarks = sourcePointSet->GetPoints();

  typename LandmarkContainer::Iterator pointDataIterator;

  typename LandmarkContainer::ConstIterator It = inputPointSet->GetPoints()->Begin();

  while( It != inputPointSet->GetPoints()->End() )
  {
    LandmarkPointType landmark = It.Value();

    targetLandmarks->InsertElement( iLandmark, landmark );

    landmark[ImageDimension - 1] = 0.;
    sourceLandmarks->InsertElement( iLandmark, landmark );

    iLandmark++;
    It++;
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

  // First subsample the input displacement field in order to create
  // the KernelBased spline.
  this->PrepareKernelBaseSpline();

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

  // Create an iterator that will generate the indices for which the
  // height of the spline will be calculated
  OutputImageRegionType region;
  OutputSizeType size;

  region = outputPtr->GetRequestedRegion();
  size = region.GetSize();

  size[ ImageDimension - 1 ] = 1; // We're only interested in the xy plane (3D) or x (2D)
  region.SetSize( size );

  OutputIterator outIt(outputPtr, region);

  outIt.GoToBegin();

  // Support for progress methods/callbacks
  ProgressReporter progress(this, 0, region.GetNumberOfPixels(), 10);

  // Walk the output region

  OutputPointType interpolatedDisplacement;

  region = outputPtr->GetRequestedRegion();
  size = region.GetSize();

  for ( i=0; i<ImageDimension - 1; i++ )
  {
    size[ i ] = 1;              // Single column of voxels to set the height in the mask
  }

  region.SetSize( size );

  while ( !outIt.IsAtEnd() )
  {
    // Determine the index of the current output pixel
    outputIndex = outIt.GetIndex();
    outputPtr->TransformIndexToPhysicalPoint( outputIndex, outputPoint );

    //std::cout << "Index: " << outputIndex << "  Point: " << outputPoint;

    // Compute corresponding spline height
    interpolatedDisplacement = m_KernelTransform->TransformPoint( outputPoint );

    region.SetIndex( outputIndex );
    LineIteratorType itHeight( outputPtr, region );

    itHeight.SetDirection( ImageDimension - 1 );

    outputPtr->TransformPhysicalPointToIndex( interpolatedDisplacement, outputIndex );
    
    itHeight.GoToBegin();
    itHeight.GoToBeginOfLine();

    //std::cout << "   Displacement: " << interpolatedDisplacement 
    //          << "   Index: " << outputIndex << std::endl;

    for ( i=0; 
          i<=outputIndex[ ImageDimension - 1 ] && ( ! itHeight.IsAtEndOfLine() );
          i++, ++itHeight )
    {
      itHeight.Set( 1 );
    }


    ++outIt;
    progress.CompletedPixel();
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
