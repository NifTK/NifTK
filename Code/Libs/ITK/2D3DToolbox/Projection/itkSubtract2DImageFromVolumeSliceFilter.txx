/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSubtract2DImageFromVolumeSliceFilter_txx
#define __itkSubtract2DImageFromVolumeSliceFilter_txx

#include "itkSubtract2DImageFromVolumeSliceFilter.h"

#include "itkImageRegionIterator.h"
#include "itkProgressReporter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageFileWriter.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::Subtract2DImageFromVolumeSliceFilter()
{
  m_SliceNumber = 0;
}


/* -----------------------------------------------------------------------
   SetInputImage2D()
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::SetInputImage2D( const InputImageType *im2D)
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(0, const_cast<InputImageType *>( im2D ));
}


/* -----------------------------------------------------------------------
   SetInputVolume3D()
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::SetInputVolume3D( const InputProjectionVolumeType *im3D)
{
  // Process object is not const-correct so the const casting is required.
  this->SetNthInput(1, const_cast<InputProjectionVolumeType *>( im3D ));
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::BeforeThreadedGenerateData(void)
{
#if 0

  cout << "DEBUG - Inputs to Subtract2DImageFromVolumeSliceFilter: ";

  // We use dynamic_cast since inputs are stored as DataObjects.  The
  // ImageToImageFilter::GetInput(int) always returns a pointer to a
  // TInputImage1 so it cannot be used for the second input.
  InputImagePointer pInput2D
    = dynamic_cast<InputImageType *>(ProcessObject::GetInput(0));
  InputProjectionVolumePointer pInput3D
    = dynamic_cast<InputProjectionVolumeType *>(ProcessObject::GetInput(1));

  ImageRegionConstIterator<InputImageType> itInput2D(pInput2D, pInput2D->GetLargestPossibleRegion());

  InputImageIndexType index2D;
  InputProjectionVolumeIndexType index3D;

  itInput2D.GoToBegin();

  index3D[2] = m_SliceNumber;

  while( !itInput2D.IsAtEnd() )  {

    index2D = itInput2D.GetIndex();

    index3D[0] = index2D[0];
    index3D[1] = index2D[1];

    cout << index3D[0]  << ", " << index3D[1]  << ", " << index3D[2] 
	 << ": " << itInput2D.Get() << " - " << pInput3D->GetPixel(index3D) << endl;

    ++itInput2D;
  }

  cout << endl;

#endif
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::GenerateData(void)
{
  niftkitkDebugMacro(<<"Multi-threaded subtraction of volume slice");
  Superclass::GenerateData();
}


/* -----------------------------------------------------------------------
   ThreadedGenerateData(const OutputImageRegionType&, int)
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::ThreadedGenerateData( const OutputSubtractedImageRegionType &outputRegionForThread,
                        int threadId)
{
  // We use dynamic_cast since inputs are stored as DataObjects.  The
  // ImageToImageFilter::GetInput(int) always returns a pointer to a
  // TInputImage1 so it cannot be used for the second input.
  InputImagePointer pInput2D
    = dynamic_cast<InputImageType *>(ProcessObject::GetInput(0));
  InputProjectionVolumePointer pInput3D
    = dynamic_cast<InputProjectionVolumeType *>(ProcessObject::GetInput(1));

  OutputSubtractedImagePointer pOutput2D = this->GetOutput(0);
  
  ImageRegionConstIterator<InputImageType> itInput2D(pInput2D, outputRegionForThread);
  ImageRegionIterator<OutputSubtractedImageType> itOutput2D(pOutput2D, outputRegionForThread);

  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  InputImageIndexType index2D;
  InputProjectionVolumeIndexType index3D;

  itInput2D.GoToBegin();
  itOutput2D.GoToBegin();

  index3D[2] = m_SliceNumber;

  while( !itInput2D.IsAtEnd() )  {

    index2D = itInput2D.GetIndex();

    index3D[0] = index2D[0];
    index3D[1] = index2D[1];

    itOutput2D.Set( itInput2D.Get() - pInput3D->GetPixel(index3D) );

    ++itInput2D;
    ++itOutput2D;

    progress.CompletedPixel(); // potential exception thrown here
  }
}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
Subtract2DImageFromVolumeSliceFilter<IntensityType>
::AfterThreadedGenerateData(void)
{
#if 0
  OutputSubtractedImagePointer outImage = this->GetOutput();
  ImageRegionIterator<OutputSubtractedImageType> outputIterator;

  cout << "Output of Subtract2DImageFromVolumeSliceFilter: ";
 
  outputIterator = ImageRegionIterator<OutputSubtractedImageType>(outImage, outImage->GetLargestPossibleRegion());

  for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) {
      cout << outputIterator.Get() << " ";
  }

  cout << endl;
#endif
}


} // end namespace itk


#endif
