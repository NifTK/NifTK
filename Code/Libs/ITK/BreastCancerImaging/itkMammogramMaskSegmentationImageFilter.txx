/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramMaskSegmentationImageFilter_txx
#define __itkMammogramMaskSegmentationImageFilter_txx

#include "itkMammogramMaskSegmentationImageFilter.h"

#include <itkBasicImageFeaturesImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageFileWriter.h>
#include <itkImageDuplicator.h>
#include <itkProgressReporter.h>

#include <niftkConversionUtils.h>

#include <vnl/vnl_double_2x2.h>

#include <itkUCLMacro.h>


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::MammogramMaskSegmentationImageFilter()
{

  // Multi-threaded execution is enabled by default
  m_FlagMultiThreadedExecution = true;

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::~MammogramMaskSegmentationImageFilter()
{
}


/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out) 
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData(void)
{
  typedef itk::BasicImageFeaturesImageFilter< TInputImage, 
                                              TOutputImage > BasicImageFeaturesFilterType;

 
  // Create the basic image features filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename BasicImageFeaturesFilterType::Pointer BIFsFilter = BasicImageFeaturesFilterType::New();

  BIFsFilter->SetEpsilon( 1e-3 );

  BIFsFilter->SetInput( this->GetInput() );     
  BIFsFilter->SetSigma( 2. );
  BIFsFilter->SetSingleThreadedExecution();

  std::cout << "Computing Basic Image Features" << std::endl;
  BIFsFilter->Update();
        
  this->GraftOutput( BIFsFilter->GetOutput() );

  typedef itk::ImageFileWriter< TOutputImage > WriterType;
  typename WriterType::Pointer writer = WriterType::New();

  writer->SetFileName( "BIFS.nii" );
  writer->SetInput( BIFsFilter->GetOutput() );
  
  try
  {
    writer->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
  }                

  
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::GenerateData(void)
{
  // Perform multi-threaded execution by default

  if (m_FlagMultiThreadedExecution) {
    
    niftkitkDebugMacro( "Multi-threaded mammogram mask segmentation");

    Superclass::GenerateData();
  }

  // Single-threaded execution

  else {
  
    niftkitkDebugMacro( "Single-threaded mammogram mask segmentation");

    this->AllocateOutputs();
    this->BeforeThreadedGenerateData();
  
    // Set up the multithreaded processing
    typename ImageSource<TOutputImage>::ThreadStruct str;
    str.Filter = this;
    
    this->GetMultiThreader()->SetNumberOfThreads( 1 );
    this->GetMultiThreader()->SetSingleMethod(this->ThreaderCallback, &str);
    
    // multithread the execution
    this->GetMultiThreader()->SingleMethodExecute();
    
    // Call a method that can be overridden by a subclass to perform
    // some calculations after all the threads have completed
    this->AfterThreadedGenerateData();
  }
}


/* -----------------------------------------------------------------------
   ThreadedGenerateData(const OutputImageRegionType&, int)
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId)
{

}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::AfterThreadedGenerateData(void)
{

}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if (m_FlagMultiThreadedExecution)
    os << indent << "MultiThreadedExecution: ON" << std::endl;
  else
    os << indent << "MultiThreadedExecution: OFF" << std::endl;

}

} // end namespace itk

#endif
