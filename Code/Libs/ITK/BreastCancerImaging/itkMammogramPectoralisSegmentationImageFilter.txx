/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramPectoralisSegmentationImageFilter_txx
#define __itkMammogramPectoralisSegmentationImageFilter_txx

#include "itkMammogramPectoralisSegmentationImageFilter.h"

#include <niftkConversionUtils.h>

#include <vnl/vnl_double_2x2.h>

#include <itkUCLMacro.h>

#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageToHistogramFilter.h>
#include <itkImageMomentsCalculator.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkShrinkImageFilter.h>
#include <itkExpandImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkSubsampleImageFilter.h>

#include <itkForegroundFromBackgroundImageThresholdCalculator.h>


namespace itk
{


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::MammogramPectoralisSegmentationImageFilter()
{
  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::~MammogramPectoralisSegmentationImageFilter()
{
}


/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out) 
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
}


// --------------------------------------------------------------------------
// WriteImageToFile()
// --------------------------------------------------------------------------

template <typename TInputImage, typename TOutputImage>
void
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::WriteImageToFile( const char *fileOutput, const char *description,
		    typename TInputImage::Pointer image )
{
  if ( fileOutput ) {

    typedef itk::ImageFileWriter< TInputImage > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput );
    writer->SetInput( image );

    try
    {
      std::cout << "Writing " << description << " to file: "
                << fileOutput << std::endl;
      writer->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
      exit( EXIT_FAILURE );
    }
  }
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::GenerateData(void)
{
  unsigned int i;

  // Single-threaded execution

  this->AllocateOutputs();

  InputImageConstPointer image = this->GetInput();



  this->GraftOutput( ->GetOutput() );
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
