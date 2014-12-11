/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkUCLN4BiasFieldCorrectionFilter_txx
#define __itkUCLN4BiasFieldCorrectionFilter_txx

#include "itkUCLN4BiasFieldCorrectionFilter.h"

#include <itkCastImageFilter.h>
#include <itkOtsuThresholdImageFilter.h>
#include <itkShrinkImageFilter.h>
#include <itkN4BiasFieldCorrectionImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkExpandImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkExpImageFilter.h>
#include <itkImageDuplicator.h>

#include <itkExceptionObject.h>

#include <vnl/vnl_math.h>

#include <itkLogHelper.h>


namespace itk
{

// -------------------------------------------------------------------------
// N4BFC_IterationUpdate
// -------------------------------------------------------------------------

template<class TFilter>
class N4BFC_IterationUpdate : public itk::Command
{
public:
  typedef N4BFC_IterationUpdate  Self;
  typedef itk::Command            Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  itkNewMacro( Self );

protected:
  N4BFC_IterationUpdate() {}

public:

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *) caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    const TFilter * filter =
      dynamic_cast< const TFilter * >( object );

    if( typeid( event ) != typeid( itk::IterationEvent ) )
                                        { return; }
    if( filter->GetElapsedIterations() == 1 )
      {
      std::cout << "Current level = " << filter->GetCurrentLevel() + 1
                << std::endl;
      }
    std::cout << "  Iteration " << filter->GetElapsedIterations()
              << " (of "
              << filter->GetMaximumNumberOfIterations()[
      filter->GetCurrentLevel()]
              << ").  ";
    std::cout << " Current convergence value = "
              << filter->GetCurrentConvergenceMeasurement()
              << " (threshold = " << filter->GetConvergenceThreshold()
              << ")" << std::endl;
    }

};


/* ---------------------------------------------------------------------
   Constructor
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
UCLN4BiasFieldCorrectionFilter<TInputImage, TOutputImage>
::UCLN4BiasFieldCorrectionFilter()
{
  m_Subsampling = 4.;
  m_SplineOrder = 3;
  m_NumberOfHistogramBins = 200;
  m_WeinerFilterNoise = 0.01;
  m_BiasFieldFullWidthAtHalfMaximum = 0.15;
  m_MaximumNumberOfIterations = 50;
  m_ConvergenceThreshold = 0.001;
  m_NumberOfFittingLevels = 4;
  m_NumberOfControlPoints = 0;     

  m_Mask = 0;
}
 

/* ---------------------------------------------------------------------
   GenerateData()
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
UCLN4BiasFieldCorrectionFilter<TInputImage, TOutputImage>
::GenerateData()
{
  unsigned int idim;

  // Get the input and output pointers

  typedef itk::ImageDuplicator< InputImageType > DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();

  duplicator->SetInputImage( this->GetInput() );
  duplicator->Update();

  InputImagePointer imOriginal = duplicator->GetOutput();
  

  // Allocate memory for the output

  OutputImagePointer outputPtr = this->GetOutput( );

  outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
  outputPtr->Allocate();


  // Create a mask by thresholding
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  typedef itk::OtsuThresholdImageFilter< InputImageType,
                                         MaskImageType > OtsuThresholdImageFilterType;

  typename OtsuThresholdImageFilterType::Pointer thresholder = OtsuThresholdImageFilterType::New();

  if ( ! m_Mask ) 
  {
    thresholder->SetInput( imOriginal );

    thresholder->SetInsideValue( 0 );
    thresholder->SetOutsideValue( 1 );

    thresholder->SetNumberOfHistogramBins( 200 );

    std::cout << "Thresholding to obtain image mask" << std::endl;
    thresholder->Update();

    m_Mask = thresholder->GetOutput();
  }


  // Shrink the image and the mask
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ShrinkImageFilter <InputImageType, InputImageType> ShrinkImageFilterType;
 
  typename ShrinkImageFilterType::Pointer shrinkFilter = ShrinkImageFilterType::New();

  shrinkFilter->SetInput( imOriginal );

  shrinkFilter->SetShrinkFactor(0, m_Subsampling);
  shrinkFilter->SetShrinkFactor(1, m_Subsampling);
  shrinkFilter->SetShrinkFactor(2, m_Subsampling);

  std::cout << "Shrinking the image by a factor of " << m_Subsampling << std::endl;
  shrinkFilter->Update();


  typedef itk::ShrinkImageFilter <MaskImageType, MaskImageType> ShrinkMaskFilterType;
 
  typename ShrinkMaskFilterType::Pointer maskShrinkFilter = ShrinkMaskFilterType::New();

  maskShrinkFilter->SetInput( m_Mask );

  maskShrinkFilter->SetShrinkFactor(0, m_Subsampling);
  maskShrinkFilter->SetShrinkFactor(1, m_Subsampling);
  maskShrinkFilter->SetShrinkFactor(2, m_Subsampling);

  std::cout << "Shrinking the mask by a factor of " << m_Subsampling << std::endl;
  maskShrinkFilter->Update();


  // Compute the N4 Bias Field Correction
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::N4BiasFieldCorrectionImageFilter <InputImageType, 
                                                 MaskImageType,
                                                 InputImageType> N4BiasFieldCorrectionImageFilterType;
 
  typename N4BiasFieldCorrectionImageFilterType::Pointer 
    biasFieldFilter = N4BiasFieldCorrectionImageFilterType::New();

  biasFieldFilter->SetInput( shrinkFilter->GetOutput() );

  biasFieldFilter->SetMaskImage( maskShrinkFilter->GetOutput() );

  typedef N4BFC_IterationUpdate< N4BiasFieldCorrectionImageFilterType > CommandType;
  typename CommandType::Pointer observer = CommandType::New();
  biasFieldFilter->AddObserver( itk::IterationEvent(), observer );

  biasFieldFilter->SetMaskLabel( 1 );

  biasFieldFilter->SetNumberOfHistogramBins( m_NumberOfHistogramBins );
  biasFieldFilter->SetSplineOrder( m_SplineOrder );
  biasFieldFilter->SetWienerFilterNoise( m_WeinerFilterNoise );
  biasFieldFilter->SetBiasFieldFullWidthAtHalfMaximum( m_BiasFieldFullWidthAtHalfMaximum );
  biasFieldFilter->SetConvergenceThreshold( m_ConvergenceThreshold );

  // handle the number of iterations
  std::vector<unsigned int> numIters;
  numIters.resize( m_NumberOfFittingLevels, m_MaximumNumberOfIterations );

  typename N4BiasFieldCorrectionImageFilterType::VariableSizeArrayType
  maximumNumberOfIterations( numIters.size() );
  for( unsigned int d = 0; d < numIters.size(); d++ )
  {
    maximumNumberOfIterations[d] = numIters[d];
  }
  biasFieldFilter->SetMaximumNumberOfIterations( maximumNumberOfIterations );

  typename N4BiasFieldCorrectionImageFilterType::ArrayType numberOfFittingLevels;
  numberOfFittingLevels.Fill( numIters.size() );
  biasFieldFilter->SetNumberOfFittingLevels( numberOfFittingLevels );

  if ( m_NumberOfControlPoints )
  {
    typename N4BiasFieldCorrectionImageFilterType::ArrayType numberOfControlPoints;
    numberOfControlPoints.Fill( m_NumberOfControlPoints );
    biasFieldFilter->SetNumberOfControlPoints( numberOfControlPoints );
  }

  std::cout << "Computing the bias field" << std::endl;
  biasFieldFilter->UpdateLargestPossibleRegion();


  // Reconstruction of the bias field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Compute the log bias field

  typedef itk::BSplineControlPointImageFilter
    < typename N4BiasFieldCorrectionImageFilterType::BiasFieldControlPointLatticeType, 
      typename N4BiasFieldCorrectionImageFilterType::ScalarImageType > BSplinerType;

  typename BSplinerType::Pointer bspliner = BSplinerType::New();

  bspliner->SetInput( biasFieldFilter->GetLogBiasFieldControlPointLattice() );
  bspliner->SetSplineOrder( biasFieldFilter->GetSplineOrder() );
  bspliner->SetSize( imOriginal->GetLargestPossibleRegion().GetSize() );
  bspliner->SetOrigin( imOriginal->GetOrigin() );
  bspliner->SetDirection( imOriginal->GetDirection() );
  bspliner->SetSpacing( imOriginal->GetSpacing() );

  std::cout << "Computing the log bias field" << std::endl;
  bspliner->UpdateLargestPossibleRegion();

  // And then the exponential

  typedef itk::VectorIndexSelectionCastImageFilter
    < typename N4BiasFieldCorrectionImageFilterType::ScalarImageType, 
    InputImageType > CastImageFilterType;

  typename CastImageFilterType::Pointer caster = CastImageFilterType::New();

  caster->SetInput( bspliner->GetOutput() );
  caster->SetIndex(0);


  typedef itk::ExpImageFilter
    < InputImageType, 
      InputImageType > ExpImageFilterType;

  typename ExpImageFilterType::Pointer expFilter = ExpImageFilterType::New();

  expFilter->SetInput( caster->GetOutput() );

  std::cout << "Computing the exponential of the bias field" << std::endl;
  expFilter->UpdateLargestPossibleRegion();


  // Correct the original input image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename InputImageType::Pointer imBiasField = expFilter->GetOutput();

  typename itk::ImageRegionIterator< InputImageType > 
    iterOriginal( imOriginal, imOriginal->GetLargestPossibleRegion() );

  typename itk::ImageRegionConstIterator< InputImageType > 
    iterBiasField( imBiasField, imBiasField->GetLargestPossibleRegion() );
        
  for ( iterBiasField.GoToBegin(), iterOriginal.GoToBegin();
        ! iterBiasField.IsAtEnd();
        ++iterBiasField, ++iterOriginal)
  {
    if ( iterBiasField.Get() )
    {
      iterOriginal.Set( iterOriginal.Get() / iterBiasField.Get() );
    }
  }    


  this->GraftOutput( imOriginal );
}


/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
UCLN4BiasFieldCorrectionFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}


} // namespace itk

#endif
