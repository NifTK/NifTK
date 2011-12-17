/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkBackwardImageProjector2Dto3D_txx
#define __itkBackwardImageProjector2Dto3D_txx

#include "itkBackwardImageProjector2Dto3D.h"

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
BackwardImageProjector2Dto3D<IntensityType>
::BackwardImageProjector2Dto3D()
{
  // Multi-threaded execution is enabled by default

  m_FlagMultiThreadedExecution = true;

  // Clear the back-projected volume prior to the next back-projection

  m_ClearBackProjectedVolume = true;

  // Set default values for the output image size

  m_OutputImageSize[0]  = 100;  // size along X
  m_OutputImageSize[1]  = 100;  // size along Y
  m_OutputImageSize[2]  = 100;  // size along Z
 
  // Set default values for the output image resolution

  m_OutputImageSpacing[0]  = 1;  // resolution along X axis
  m_OutputImageSpacing[1]  = 1;  // resolution along Y axis
  m_OutputImageSpacing[2]  = 1;  // resolution along Z axis
 
  // Set default values for the output image origin

  m_OutputImageOrigin[0]  = 0.;  // origin in X
  m_OutputImageOrigin[1]  = 0.;  // origin in Y
  m_OutputImageOrigin[2]  = 0.;  // origin in Z

  OutputImagePointer      outputPtr = this->GetOutput();

  outputPtr->SetSpacing(m_OutputImageSpacing);
  outputPtr->SetOrigin(m_OutputImageOrigin);
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>::
PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Output image size: " << m_OutputImageSize << std::endl;
  os << indent << "Output image spacing: " << m_OutputImageSpacing << std::endl;
  os << indent << "Output image origin: " << m_OutputImageOrigin << std::endl;

  if (m_FlagMultiThreadedExecution)
    os << indent << "MultiThreadedExecution: ON" << std::endl;
  else
    os << indent << "MultiThreadedExecution: OFF" << std::endl;

  if (m_ClearBackProjectedVolume)
    os << indent << "ClearBackProjectedVolume: ON" << std::endl;
  else
    os << indent << "ClearBackProjectedVolume: OFF" << std::endl;
}


/* -----------------------------------------------------------------------
   GenerateOutputInformation()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::GenerateOutputInformation()
{
  OutputImageRegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize( m_OutputImageSize );

  OutputImagePointer outputPtr = this->GetOutput();
  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );  

  niftkitkDebugMacro(<< "Back-projection output size: " << outputPtr->GetLargestPossibleRegion().GetSize());
}


/* -----------------------------------------------------------------------
   GenerateInputRequestedRegion()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::GenerateInputRequestedRegion()
{
  // generate everything in the region of interest
  InputImagePointer  inputPtr = const_cast<InputImageType *> (this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();

  niftkitkDebugMacro(<< "Back-projection input size: " << inputPtr->GetLargestPossibleRegion().GetSize());
}


/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion(DataObject *)
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  // call the superclass' implementation of this method
  Superclass::EnlargeOutputRequestedRegion(output);
  
  // generate everything in the region of interest
  this->GetOutput()->SetRequestedRegionToLargestPossibleRegion();
}


/* -----------------------------------------------------------------------
   ClearVolume()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::ClearVolume(void)
{
  if (m_ClearBackProjectedVolume) {
    this->GetOutput()->FillBuffer(0);
    m_ClearBackProjectedVolume = false;

    niftkitkDebugMacro(<< "Back-projected volume has been reset to zero");
  }
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::BeforeThreadedGenerateData(void)
{
  // Do we need to set the volume to zero?
  ClearVolume();
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::GenerateData(void)
{

  // Perform multi-threaded execution by default
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (m_FlagMultiThreadedExecution) {
    
	niftkitkDebugMacro(<< "Multi-threaded back-projection");

    // Call a method that can be overriden by a subclass to allocate
    // memory for the filter's outputs
    this->AllocateOutputs();
  
    // Call a method that can be overridden by a subclass to perform
    // some calculations prior to splitting the main computations into
    // separate threads
    BeforeThreadedGenerateData();
  
    // Set up the multithreaded processing
    BackwardImageProjectorThreadStruct str;
    str.Filter = this;
  
    this->GetMultiThreader()->SetNumberOfThreads(this->GetNumberOfThreads());
    this->GetMultiThreader()->SetSingleMethod(this->BackwardImageProjectorThreaderCallback, &str);
  
    // multithread the execution
    this->GetMultiThreader()->SingleMethodExecute();

    // Call a method that can be overridden by a subclass to perform
    // some calculations after all the threads have completed
    AfterThreadedGenerateData();

  }

  // Single-threaded execution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  else {

	niftkitkDebugMacro(<< "Single-threaded back-projection");

    // Call a method that can be overriden by a subclass to allocate
    // memory for the filter's outputs
    this->AllocateOutputs();
  
    BeforeThreadedGenerateData();

    // Do we need to set the volume to zero?
    ClearVolume();

    // Call ThreadedGenerateData once for this single thread
    ThreadedGenerateData(this->GetInput()->GetRequestedRegion(), 0);

    AfterThreadedGenerateData();
  }
}


/* -----------------------------------------------------------------------
   SplitRequestedRegion()
   ----------------------------------------------------------------------- */

template< class IntensityType>
int 
BackwardImageProjector2Dto3D<IntensityType>
::SplitRequestedRegion(int i, int num, InputImageRegionType& splitRegion)
{
  // Get the input pointer
  InputImagePointer  inputPtr = const_cast<InputImageType *> (this->GetInput());
  InputImageSizeType requestedRegionSize 
    = inputPtr->GetRequestedRegion().GetSize();

  int splitAxis;
  InputImageIndexType splitIndex;
  InputImageSizeType splitSize;

  // Initialize the splitRegion to the input requested region
  splitRegion = inputPtr->GetRequestedRegion();
  splitIndex = splitRegion.GetIndex();
  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  splitAxis = inputPtr->GetImageDimension() - 1;

  while (requestedRegionSize[splitAxis] == 1) {

    --splitAxis;
    if (splitAxis < 0) { // cannot split
      niftkitkDebugMacro(<< "Cannot split region for back-projection");
      return 1;
    }
  }

  // determine the actual number of pieces that will be generated
  typename InputImageSizeType::SizeValueType range = requestedRegionSize[splitAxis];
  int valuesPerThread = (int)::ceil(range/(double)num);
  int maxThreadIdUsed = (int)::ceil(range/(double)valuesPerThread) - 1;

  // Split the region
  if (i < maxThreadIdUsed) {

    splitIndex[splitAxis] += i*valuesPerThread;
    splitSize[splitAxis] = valuesPerThread;
  }

  if (i == maxThreadIdUsed) {
    splitIndex[splitAxis] += i*valuesPerThread;
    // last thread needs to process the "rest" dimension being split
    splitSize[splitAxis] = splitSize[splitAxis] - i*valuesPerThread;
  }
  
  // set the split region ivars
  splitRegion.SetIndex( splitIndex );
  splitRegion.SetSize( splitSize );

  //niftkitkDebugMacro(<< "Back-projection split piece: " << splitRegion);

  return maxThreadIdUsed + 1;
}


/* -----------------------------------------------------------------------
   ThreaderCallback()
   Callback routine used by the threading library. This routine just calls
   the ThreadedGenerateData method after setting the correct region for this
   thread. 
   ----------------------------------------------------------------------- */

template< class IntensityType>
ITK_THREAD_RETURN_TYPE  
BackwardImageProjector2Dto3D<IntensityType>
::BackwardImageProjectorThreaderCallback( void *arg )
{
  BackwardImageProjectorThreadStruct *str;
  int total, threadId, threadCount;

  threadId = ((MultiThreader::ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((MultiThreader::ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (BackwardImageProjectorThreadStruct *)(((MultiThreader::ThreadInfoStruct *)(arg))->UserData);

  // execute the actual method with appropriate output region
  // first find out how many pieces extent can be split into.
  InputImageRegionType splitRegion;
  total = str->Filter->SplitRequestedRegion(threadId, threadCount,
                                            splitRegion);

  if (threadId < total)
    {
    str->Filter->ThreadedGenerateData(splitRegion, threadId);
    }
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a 
  //   few threads idle.
  //   }
  
  return ITK_THREAD_RETURN_VALUE;
}


/* -----------------------------------------------------------------------
   ThreadedGenerateData(const InputImageRegionType&, int)
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::ThreadedGenerateData(const InputImageRegionType& inputRegionForThread,
                       int threadId)
{
  InputImageIndexType inIndex;
  InputImagePointType inPoint;

  ImageRegionConstIterator<InputImageType> inputIterator;
  
  itk::Matrix<double, 4, 4> projMatrix;

 
  // Allocate output

  InputImageConstPointer  inImage  = this->GetInput();
  OutputImagePointer outImage = this->GetOutput();

  // Support progress methods/callbacks

  ProgressReporter progress(this, threadId, inputRegionForThread.GetNumberOfPixels());


  // Create the backprojection ray

  itk::Ray<OutputImageType> ray;
  ray.SetImage( outImage );

  // Calculate the projection matrix (perspective*affine)

  projMatrix = this->m_PerspectiveTransform->GetMatrix();
  projMatrix *= this->m_AffineTransform->GetFullAffineMatrix();

  ray.SetProjectionMatrix(projMatrix);


  // Iterate over pixels in the 2D projection (i.e. input) image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  inputIterator = ImageRegionConstIterator<InputImageType>(inImage, inputRegionForThread);

  for ( inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator) {

    // Determine the coordinate of the input pixel
    inIndex = inputIterator.GetIndex();
    inImage->TransformIndexToPhysicalPoint(inIndex, inPoint);
    
    // Create a ray for this coordinate
    ray.SetRay(inPoint);
    
    // Cast it through the volume. NB. This function will divide by
    // the number of voxels and the ray point spacing to ensure the
    // inverse of Ray::IntegrateAboveThreshold() is performed.
    ray.IncrementRayVoxelIntensities(inputIterator.Get());
	
    progress.CompletedPixel();
  }
}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
BackwardImageProjector2Dto3D<IntensityType>
::AfterThreadedGenerateData(void)
{
  OutputImagePointer     outImage = this->GetOutput();

#if 0
  ImageRegionIterator<OutputImageType> outputIterator;

  cout << "DEBUG - Output of back projection: ";
 
  outputIterator = ImageRegionIterator<OutputImageType>(outImage, outImage->GetLargestPossibleRegion());

  for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) 
      cout << outputIterator.Get() << " ";

  cout << endl;
#endif
}


} // end namespace itk


#endif
