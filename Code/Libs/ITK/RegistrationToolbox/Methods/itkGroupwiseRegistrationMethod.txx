/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkGroupwiseRegistrationMethod_txx
#define _itkGroupwiseRegistrationMethod_txx

#include <itkGroupwiseRegistrationMethod.h>

#include <itkLogHelper.h>

#include <itkImageFileWriter.h>
#include <itkArray.h>
#include <itkEulerAffineTransform.h>

#include <itkLogHelper.h>

namespace itk
{

/* -----------------------------------------------------------------------
 * Constructor
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::GroupwiseRegistrationMethod()
{
  // Prevents destruction of the allocated mean image estimate
  this->ReleaseDataBeforeUpdateFlagOff();

  this->SetNumberOfRequiredInputs( 2 );
  this->SetNumberOfRequiredOutputs( 1 );

  m_FlagInitialSumComputed = false;
  m_FlagInitialised = false;

  m_NumberOfIterations = 5;

  // Create the output which will be the reconstructed volume

  ImagePointer meanOutputImage = 
                 dynamic_cast< ImageType * >( this->MakeOutput(0).GetPointer() );

  this->ProcessObject::SetNthOutput( 0, meanOutputImage.GetPointer() );

#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  this->SetNumberOfThreads( this->GetMultiThreader()->GetNumberOfThreads() );
#else
  this->SetNumberOfThreads( 1 );
  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif
}

  
/* -----------------------------------------------------------------------
   SetInput(const ImageType *input)
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void 
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::SetInput(const ImageType *input)
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(0, const_cast< ImageType * >( input ) );
}


/* -----------------------------------------------------------------------
   SetInput( unsigned int index, const TImageType * image )
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::SetInput( unsigned int index, const TImageType *image ) 
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(index, const_cast< ImageType *>( image ) );
}


/* -----------------------------------------------------------------------
   GetInput()
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
const typename GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>::ImageType *
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::GetInput(void) 
{
  if (this->GetNumberOfInputs() < 1)
    {
    return 0;
    }
  
  return static_cast<const ImageType * >(this->ProcessObject::GetInput(0) );
}
  
/* -----------------------------------------------------------------------
   GetInput(unsigned int idx)
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
const typename GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>::ImageType *
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::GetInput(unsigned int idx)
{
  return static_cast< const ImageType * >(this->ProcessObject::GetInput(idx));
}


/* -----------------------------------------------------------------------
   GenerateInputRequestedRegion()
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::GenerateInputRequestedRegion()
{
  for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
    
    ImagePointer input = const_cast<ImageType *>(this->GetInput(i));
    
    if ( input )
      input->SetRequestedRegionToLargestPossibleRegion();
  }
}


/* -----------------------------------------------------------------------
   ComputeInitialSumOfInputImages()
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::ComputeInitialSumOfInputImages()
{
  if ( ! m_FlagInitialSumComputed ) {

    // Check that the inputs to the sum filter have been set

    if ( m_SumImagesFilter->GetNumberOfInputs() 
         != this->GetNumberOfInputs())

      itkExceptionMacro("Number of sum filter inputs (" 
                        << m_SumImagesFilter->GetNumberOfInputs() 
                        << ") does not equal number of group inputs (" 
                        << this->GetNumberOfInputs()
                        << ").");

    // Sum the input images
    
    niftkitkInfoMacro(<<"Summing input images");
    
    m_SumImagesFilter->Update();

    m_FlagInitialSumComputed = true;
    m_SumImagesFilter->ClearTransformation( );
  }
}


/* -----------------------------------------------------------------------
   GenerateOutputInformation()
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::GenerateOutputInformation()
{
  if ( ! m_FlagInitialSumComputed ) {
    ComputeInitialSumOfInputImages();

    this->GetOutput()->CopyInformation( m_SumImagesFilter->GetOutput() );
  }
}
 

/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );

}


/* -----------------------------------------------------------------------
 * Initialise by setting the interconnects between components. 
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::Initialise( void ) throw (ExceptionObject)
{
  if (m_FlagInitialised) return;

  niftkitkDebugMacro(<<"GroupwiseRegistrationMethod::Initialise()" );

  if ( ! m_SumImagesFilter )
    itkExceptionMacro("Mean images filter not set.");

  if ( m_RegistrationFilters.size() < 2 )
    itkExceptionMacro("Less than 2 registration filters set.");

  if ( ! m_FlagInitialSumComputed )
    ComputeInitialSumOfInputImages();
  
  // Setup the multi-threading
#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif

  this->Modified();
  m_FlagInitialised = true;
}


/* -----------------------------------------------------------------------
   Generate Data
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::GenerateData()
{
  niftkitkDebugMacro(<<"GroupwiseRegistrationMethod::GenerateData()" );

  this->Initialise();
  this->StartOptimization();

  this->GraftOutput( m_SumImagesFilter->GetOutput() );
}


/* -----------------------------------------------------------------------
 * Starts the Optimization process
   ----------------------------------------------------------------------- */

template <typename TImageType, unsigned int Dimension, 
          class TScalarType, typename TDeformationScalar >
void
GroupwiseRegistrationMethod<TImageType, Dimension, TScalarType, TDeformationScalar>
::StartOptimization( void )
{ 
  unsigned int iIteration;
  unsigned int iRegn;

  typedef typename itk::EulerAffineTransform<TScalarType, Dimension, Dimension> TransformType;

  typename TransformType::Pointer transform;

  typename std::vector< ImageRegistrationFilterPointerType >::iterator regnFilter;

  typedef itk::ImageDuplicator< ImageType > DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();

  
  niftkitkInfoMacro(<<"Registering images" );

  try {

    for ( iIteration=0; iIteration<m_NumberOfIterations; iIteration++ ) {

      niftkitkInfoMacro(<<"Iteration: " << iIteration );

      duplicator->SetInputImage( m_SumImagesFilter->GetOutput() );
      duplicator->Update();


      // Do the first set of image registrations

      for ( iRegn=0, regnFilter=m_RegistrationFilters.begin(); 
            regnFilter<m_RegistrationFilters.end(); 
            ++regnFilter, iRegn++ ) {
        
        niftkitkInfoMacro(<<"Invoking registration filter: " << iRegn );
        
        (*regnFilter)->SetFixedImage( duplicator->GetOutput() );
        (*regnFilter)->SetMovingImage( const_cast< ImageType * >( this->GetInput(iRegn) ));
        
        //(*regnFilter)->Print( std::cout );
        
        (*regnFilter)->Update( );
      }
      
      // Sum the transformed images
      
      niftkitkInfoMacro(<<"Summing transformed images");
      
      for (unsigned int i = 0; i<this->GetNumberOfInputs(); i++) 
        
        m_SumImagesFilter->SetInput( i, m_RegistrationFilters.at(i)->GetOutput() );
      
      //m_SumImagesFilter->SetExpandOutputRegion( 0. );
      m_SumImagesFilter->Update();
      
      typedef  itk::ImageFileWriter< ImageType > WriterType;
      typename WriterType::Pointer writer = WriterType::New();
      
      std::string filename("MeanImage_" + niftk::ConvertToString( iIteration ) + std::string( ".gipl.gz" ));

      writer->SetFileName( filename );
      writer->SetInput( m_SumImagesFilter->GetOutput() );
      
      niftkitkInfoMacro(<<"Writing iteration " << iIteration << " mean image to file: " << filename);
      writer->Update();
    }
  }

  catch( ExceptionObject& err ) {

    // Pass exception to caller
    throw err;
  }

  niftkitkDebugMacro(<<"Registration complete");
}



} // end namespace itk


#endif
