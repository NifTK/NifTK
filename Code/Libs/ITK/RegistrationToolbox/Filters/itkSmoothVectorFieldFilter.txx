/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSmoothVectorFieldFilter_txx
#define __itkSmoothVectorFieldFilter_txx

#include "itkSmoothVectorFieldFilter.h"
#include "itkImageFileWriter.h"

#include "itkLogHelper.h"

namespace itk {

template<class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions>  
SmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
::SmoothVectorFieldFilter()
{
  this->SetInPlace(false);
}

template<class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
void 
SmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
::WriteVectorImage(std::string filename)
{
  niftkitkDebugMacro(<<"WriteVectorImage():Writing to:" << filename);
  
  // Note: File writer probably wont handle 4D vectors.
  
  typedef ImageFileWriter<OutputImageType> FileWriterType;
  typename FileWriterType::Pointer writer = FileWriterType::New();
  writer->SetFileName(filename);
  writer->SetInput(this->GetOutput());
  writer->Update();
  
  niftkitkDebugMacro(<<"WriteVectorImage():Writing to:" << filename << "....DONE");
}

template<class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
void
SmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions> 
::GenerateInputRequestedRegion() throw(InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method. this should
  // copy the output requested region to the input requested region
  Superclass::GenerateInputRequestedRegion();

  // This filter needs all of the input
  InputImagePointer image = const_cast<InputImageType*>( this->GetInput() );
  
  if( image )
    {
      image->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
    }

}

template<class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
void
SmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions> 
::EnlargeOutputRequestedRegion(DataObject *output)
{
  OutputImageType *out = dynamic_cast<OutputImageType*>(output);

  if (out)
    {
      out->SetRequestedRegion( out->GetLargestPossibleRegion() );
    }
}

template<class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
void
SmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
::GenerateData()
{
  
  niftkitkDebugMacro(<<"GenerateData():Started");
  
  // Make sure the output is generated.
  this->AllocateOutputs();
  
  // Get pointers to input and output.
  InputImagePointer inputImage = const_cast<InputImageType *>(this->GetInput());
  OutputImagePointer outputImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));

  niftkitkDebugMacro(<<"GenerateData():Smoothing in base class" \
    << ", vector dimensions=" << NumberVectorDimensions \
    << ", image dimensions=" << NumberImageDimensions \
    );
    
  // Create ONE temporary image (unfortunately).
  OutputImagePointer tmpImage = OutputImageType::New();
  tmpImage->SetRegions(outputImage->GetLargestPossibleRegion());
  tmpImage->SetSpacing(outputImage->GetSpacing());
  tmpImage->SetDirection(outputImage->GetDirection());
  tmpImage->SetOrigin(outputImage->GetOrigin());
  tmpImage->Allocate();
  
  // The filter is a generic "convolve with a kernel" type thing.
  SmootherFilterPointer filter = SmootherFilterType::New();
  
  // So we alternate between writing to the output image and the tmp image.
  // Then the tmp image gets destroyed when we are done.
  
  unsigned int i = 0;
  
  for (i = 0; i < NumberImageDimensions; i++)
  {
    
    if (i%2 == 0)
    {
      if (i == 0)
      {
        niftkitkDebugMacro(<<"GenerateData():Dimension=" << i << ", input = filter input");
        filter->SetInput(this->GetInput());
      }
      else
      {
        niftkitkDebugMacro(<<"GenerateData():Dimension=" << i << ", input = output buffer");
        filter->SetInput(outputImage);
      }
      
      niftkitkDebugMacro(<<"GenerateData():Dimension=" << i << ", output = tmpImage buffer");
      filter->GraftOutput(tmpImage);
      
    }
    else
    {
      niftkitkDebugMacro(<<"GenerateData():Dimension=" << i << ", input = tmpImage buffer");
      filter->SetInput(tmpImage);
      
      niftkitkDebugMacro(<<"GenerateData():Dimension=" << i << ", output = outputImage");
      filter->GraftOutput(outputImage);
    }
    
    // Get operator from derived class.
    NeighborhoodOperatorType* op = this->CreateOperator(i);

    // And set it on the filter, nearly ready to go now!
    filter->SetOperator(*op);

    // Go, go, go. Filter to the max.
    filter->UpdateLargestPossibleRegion();
    
    // Clean up, as the next iteration, we get a new one.
    delete op;
    
  }
  
  // Just need to make sure the output was in the output buffer, not tmpImage;
  if (i%2 != 0)
  {
    niftkitkDebugMacro(<<"GenerateData():Setting output = tmpImage");
    this->GraftOutput(tmpImage);
  }
  else
  {
    niftkitkDebugMacro(<<"GenerateData():Output already in correct output buffer");
  }
  
  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace

#endif
