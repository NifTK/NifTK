#ifndef __itkDisplacementFieldCompositionFilter_txx
#define __itkDisplacementFieldCompositionFilter_txx
#include "itkDisplacementFieldCompositionFilter.h"

#include <itkProgressAccumulator.h>
#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>

namespace itk
{

/**
 * Default constructor.
 */
template <class TInputImage, class TOutputImage>
DisplacementFieldCompositionFilter<TInputImage,TOutputImage>
::DisplacementFieldCompositionFilter()
{
  // Setup the number of required inputs
  this->SetNumberOfRequiredInputs( 2 );  
  
  // Declare sub filters
  m_Warper = VectorWarperType::New();
  m_Adder = AdderType::New();

  // Setup the default interpolator
  typedef VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
    DisplacementFieldType,double> DefaultFieldInterpolatorType;
  m_Warper->SetInterpolator( DefaultFieldInterpolatorType::New() );

  // Setup the adder to be inplace
  m_Adder->InPlaceOn();
}


/**
 * Standard PrintSelf method.
 */
template <class TInputImage, class TOutputImage>
void
DisplacementFieldCompositionFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Warper: "<<m_Warper<<std::endl;
  os << indent << "Adder: "<<m_Adder<<std::endl;
}

/** 
 * GenerateData()
 */
template <class TInputImage, class TOutputImage>
void 
DisplacementFieldCompositionFilter<TInputImage,TOutputImage>
::GenerateData()
{
  DisplacementFieldConstPointer leftField = this->GetInput(0);
#if ( ITK_VERSION_MAJOR < 3 ) || ( ITK_VERSION_MAJOR == 3 && ITK_VERSION_MINOR < 13 )
  DisplacementFieldPointer rightField = const_cast<DisplacementFieldType*>(this->GetInput(1));
#else
  DisplacementFieldConstPointer rightField = this->GetInput(1);
#endif

  // Sanity checks
  if( !m_Warper )
    {
    itkExceptionMacro(<< "Warper not set");
    }

  // Set up mini-pipeline
  m_Warper->SetInput( leftField );
  m_Warper->SetDeformationField( rightField );
  m_Warper->SetOutputOrigin( rightField->GetOrigin() );
  m_Warper->SetOutputSpacing( rightField->GetSpacing() );
  m_Warper->SetOutputDirection( rightField->GetDirection() );

  m_Adder->SetInput1( m_Warper->GetOutput() );
  m_Adder->SetInput2( rightField );
  
  m_Adder->GetOutput()->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Create a progress accumulator for tracking the progress of minipipeline
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  progress->RegisterInternalFilter(m_Warper, 0.6);
  progress->RegisterInternalFilter(m_Adder, 0.4);

  // Update the pipeline
  m_Adder->Update();
  
  // Region passing stuff
  this->GraftOutput( m_Adder->GetOutput() );
}
  

} // end namespace itk


#endif
