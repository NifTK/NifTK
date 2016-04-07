/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageProjector2D3D_txx
#define __itkImageProjector2D3D_txx

#include "itkImageProjector2D3D.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkProgressReporter.h>

#include <itkCastImageFilter.h>

#include <itkLogHelper.h>


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
ImageProjector2D3D<IntensityType>
::ImageProjector2D3D()
{
  // Initialise the threshold above which intensities are integrated
  
  m_Threshold = 0.;

  // Multi-threaded execution is enabled by default

  m_FlagMultiThreadedExecution = true;

  // Set default values for the output image size

  m_OutputImageSize[0]  = 100;  // size along X
  m_OutputImageSize[1]  = 100;  // size along Y
 
  // Set default values for the output image resolution

  m_OutputImageSpacing[0]  = 1;  // resolution along X axis
  m_OutputImageSpacing[1]  = 1;  // resolution along Y axis
 
  // Set default values for the output image origin

  m_OutputImageOrigin[0]  = 0.;  // origin in X
  m_OutputImageOrigin[1]  = 0.;  // origin in Y

  OutputImagePointer      outputPtr = this->GetOutput();

  outputPtr->SetSpacing(m_OutputImageSpacing);
  outputPtr->SetOrigin(m_OutputImageOrigin);
}


/* -----------------------------------------------------------------------
   PrintSelf(std::ostream&, Indent)
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ImageProjector2D3D<IntensityType>::
PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Output image size: " << m_OutputImageSize << std::endl;
  os << indent << "Output image spacing: " << m_OutputImageSpacing << std::endl;
  os << indent << "Output image origin: " << m_OutputImageOrigin << std::endl;

  os << indent << "Threshold: " << m_Threshold << std::endl;

  if (m_FlagMultiThreadedExecution)
    os << indent << "MultiThreadedExecution: ON" << std::endl;
  else
    os << indent << "MultiThreadedExecution: OFF" << std::endl;
}


/* -----------------------------------------------------------------------
   GenerateOutputInformation()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageProjector2D3D<IntensityType>
::GenerateOutputInformation()
{
  OutputImageRegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize( m_OutputImageSize );

  OutputImagePointer outputPtr = this->GetOutput();
  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );  

  niftkitkDebugMacro(<<"Forward-projection output size: " << outputPtr->GetLargestPossibleRegion().GetSize());
}

/* -----------------------------------------------------------------------
   GenerateInputRequestedRegion()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageProjector2D3D<IntensityType>
::GenerateInputRequestedRegion()
{
  // generate everything in the region of interest
  InputImagePointer  inputPtr = const_cast<InputImageType *> (this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();

  niftkitkDebugMacro(<<"Forward-projection input size: " << inputPtr->GetLargestPossibleRegion().GetSize());
}

/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion(DataObject *)
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageProjector2D3D<IntensityType>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  // call the superclass' implementation of this method
  Superclass::EnlargeOutputRequestedRegion(output);
  
  // generate everything in the region of interest
  this->GetOutput()->SetRequestedRegionToLargestPossibleRegion();
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageProjector2D3D<IntensityType>
::BeforeThreadedGenerateData(void)
{

    // Set the output projection to zero
    this->GetOutput()->FillBuffer(0);


#if 0
  static int n = 0;
  char filename[256];

  // First cast the image from double to float

  typedef itk::Image< float, 3 > OutputImageType;
  typedef itk::CastImageFilter< InputImageType, OutputImageType > CastFilterType;

  typename CastFilterType::Pointer  caster =  CastFilterType::New();

  caster->SetInput( this->GetInput() );

  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  sprintf(filename, "/tmp/ImageProjector2D3D_INPUT_%03d.gipl", ++n );
  writer->SetFileName( filename );
  writer->SetInput( caster->GetOutput() );

  try {
    std::cout << "BeforeThreadedGenerateData: " << filename << std::endl;
    writer->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to write output to file: " << filename << "; " << err << endl;
  }

  if (n >= 21)
    exit(1);
#endif
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageProjector2D3D<IntensityType>
::GenerateData(void)
{
  // Perform multi-threaded execution by default

  if (m_FlagMultiThreadedExecution) {
    
    niftkitkDebugMacro(<<"Multi-threaded forward-projection");

    // Call a method that can be overriden by a subclass to allocate
    // memory for the filter's outputs
    this->AllocateOutputs();

    // Call a method that can be overridden by a subclass to perform
    // some calculations prior to splitting the main computations into
    // separate threads
    this->BeforeThreadedGenerateData();
    
    // Set up the multithreaded processing
    ImageProjectorThreadStruct str;
    str.Filter = this;
  
    this->GetMultiThreader()->SetNumberOfThreads(this->GetNumberOfThreads());
    this->GetMultiThreader()->SetSingleMethod(this->ImageProjectorThreaderCallback, &str);
  
    // multithread the execution
    this->GetMultiThreader()->SingleMethodExecute();

    // Call a method that can be overridden by a subclass to perform
    // some calculations after all the threads have completed
    this->AfterThreadedGenerateData();
  }

  // Single-threaded execution

  else {
  
    niftkitkDebugMacro(<<"Single-threaded forward-projection");

    // Call a method that can be overriden by a subclass to allocate
    // memory for the filter's outputs
    this->AllocateOutputs();
  
    BeforeThreadedGenerateData();

    // Call ThreadedGenerateData once for this single thread
    ThreadedGenerateData(this->GetInput()->GetRequestedRegion(), 0);

    // Call a method that can be overridden by a subclass to perform
    // some calculations after all the threads have completed
    this->AfterThreadedGenerateData();
  }
}



/* -----------------------------------------------------------------------
   SplitRequestedRegion()
   ----------------------------------------------------------------------- */

template< class IntensityType>
int 
ImageProjector2D3D<IntensityType>
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
      niftkitkDebugMacro(<<"Cannot split region for projection");
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

  //niftkitkDebugMacro(<<"Projection split piece: " << splitRegion);

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
ImageProjector2D3D<IntensityType>
::ImageProjectorThreaderCallback( void *arg )
{
  ImageProjectorThreadStruct *str;
  int total, threadId, threadCount;

  threadId = ((MultiThreader::ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((MultiThreader::ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (ImageProjectorThreadStruct *)(((MultiThreader::ThreadInfoStruct *)(arg))->UserData);

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
ImageProjector2D3D<IntensityType>
::ThreadedGenerateData(const InputImageRegionType& inputRegionForThread,
                       ThreadIdType threadId)
{
    double dx, dy, dz;
    double ox, oy, oz;

    double subVoxelProportion;

    double voxelSubSamplingFactor = 5;
    double nSubVoxels;

    InputImageIndexType inIndex3D;
    OutputImageIndexType outIndex2D;

    InputImagePointType inPoint3D;
    OutputImagePointType outPoint2D;

    OutputImagePixelType outValue2D;

    ImageRegionConstIteratorWithIndex<InputImageType> inputIterator;
  
    Vector<double, 4> subPoint3D;
    subPoint3D[3] = 1.;

    Vector<double, 4> transPoint3D;
    Vector<double, 4> projPoint2D;

    Vector<double, 4> normalPosn;
    
    // Allocate output

    InputImageConstPointer inImage3D  = this->GetInput();
    OutputImagePointer     outImage2D = this->GetOutput();

    // Support progress methods/callbacks
    
    ProgressReporter progress(this, threadId, inputRegionForThread.GetNumberOfPixels());

    OutputImageSpacingType res2D = outImage2D->GetSpacing();
    
    double r2D = sqrt( res2D[0]*res2D[0] + res2D[1]*res2D[1] );

    Matrix<double, 4, 4> perspMatrix = this->m_PerspectiveTransform->GetMatrix();

    this->m_PerspectiveTransform->GetOriginIn2D( normalPosn[0],  normalPosn[1]);
    normalPosn[2] = this->m_PerspectiveTransform->GetFocalDistance();
    normalPosn[3] = 1.;

    Matrix<double, 4, 4> affineMatrix = this->m_AffineTransform->GetFullAffineMatrix();

#if 0
    std::cout << "Affine Matrix: " << std::endl << affineMatrix << std::endl;
    std::cout << "Perspective Matrix: " << std::endl << perspMatrix << std::endl;
#endif

    // Iterate over pixels in the 3D volume (i.e. input) image

    inputIterator = ImageRegionConstIteratorWithIndex<InputImageType>(inImage3D, inputRegionForThread);

    unsigned int nx, ny, nz;

    InputImageSpacingType inSpacing3D = inImage3D->GetSpacing();

    double r3D = sqrt(    inSpacing3D[0]*inSpacing3D[0] 
		       +  inSpacing3D[1]*inSpacing3D[1]  
		       +  inSpacing3D[2]*inSpacing3D[2] );

    for ( inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator) {

	// Determine the coordinate of the input voxel
	inIndex3D = inputIterator.GetIndex();
	inImage3D->TransformIndexToPhysicalPoint(inIndex3D, inPoint3D);

	// Calculate the voxel subsampling required
	transPoint3D[0] = inPoint3D[0];
	transPoint3D[1] = inPoint3D[1];
	transPoint3D[2] = inPoint3D[2];
	transPoint3D[3] = 1.;

	transPoint3D = affineMatrix*transPoint3D;

	projPoint2D = perspMatrix*transPoint3D;
	projPoint2D /= projPoint2D[3];

	projPoint2D[0] -= normalPosn[0];
	projPoint2D[1] -= normalPosn[1];

#if 1
	nSubVoxels = ceil( voxelSubSamplingFactor * r3D * projPoint2D.GetNorm() 
			   / ( r2D * transPoint3D.GetNorm() ) );
#else
	nSubVoxels = 10.;
#endif
	subVoxelProportion = 1./(nSubVoxels*nSubVoxels*nSubVoxels);

	dx = inSpacing3D[0]/nSubVoxels;
	dy = inSpacing3D[1]/nSubVoxels;
	dz = inSpacing3D[2]/nSubVoxels;

	ox = (inSpacing3D[0] - dx)/2.;
	oy = (inSpacing3D[1] - dy)/2.;
	oz = (inSpacing3D[2] - dz)/2.;

#if 0
	std::cout << "inIndex3D: " << inIndex3D
		  << " inPoint3D: " << inPoint3D 
		  << " trpt3D.norm: " << transPoint3D.GetNorm()
		  << " prjpt2DD.norm: " << projPoint2D.GetNorm()
		  << " nSubVoxels: " << nSubVoxels 
		  << std::endl;
#endif

	// Divide the voxel up into n^3 sub-voxels
	for         ( nz = 0, subPoint3D[2] = inPoint3D[2] - oz; nz < nSubVoxels; nz++, subPoint3D[2] += dz) {
	    for     ( ny = 0, subPoint3D[1] = inPoint3D[1] - oy; ny < nSubVoxels; ny++, subPoint3D[1] += dy) {
		for ( nx = 0, subPoint3D[0] = inPoint3D[0] - ox; nx < nSubVoxels; nx++, subPoint3D[0] += dx) {

		    transPoint3D = affineMatrix*subPoint3D;

		    projPoint2D = perspMatrix*transPoint3D;
		    projPoint2D /= projPoint2D[3];

		    outPoint2D[0] = projPoint2D[0];
		    outPoint2D[1] = projPoint2D[1];

		    if ( inputIterator.Get() && outImage2D->TransformPhysicalPointToIndex( outPoint2D, outIndex2D ) ) {
#if 0
			std::cout << nx << ", " << ny << ", " << nz << ": "
				  << " sub3D: " << subPoint3D
				  << " tr3D: " << transPoint3D
				  << " val3D: " << inputIterator.Get()
				  << " proj2D: " << projPoint2D
				  << " out2D: " << outPoint2D 
				  << " outInd2D: " << outIndex2D;
#endif

			projPoint2D[0] -= normalPosn[0];
			projPoint2D[1] -= normalPosn[1];

#if 1
			outValue2D = outImage2D->GetPixel( outIndex2D ) + static_cast<IntensityType>( inputIterator.Get()*subVoxelProportion );
#else
			outValue2D = outImage2D->GetPixel( outIndex2D ) + 1;
#endif
			outImage2D->SetPixel( outIndex2D, outValue2D );
#if 0
			std::cout << " outVal2D: " << outValue2D
				  << std::endl;
#endif

		    }
		}
	    }
	}
    }

}


/* -----------------------------------------------------------------------
   AfterThreadedGenerateData()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageProjector2D3D<IntensityType>
::AfterThreadedGenerateData(void)
{
#if 0
  static int n = 0;
  char filename[256];

  // First cast the image from double to float

  typedef itk::Image< float, 3 > FloatOutputImageType;
  typedef itk::CastImageFilter< OutputImageType, FloatOutputImageType > CastFilterType;

  typename CastFilterType::Pointer  caster =  CastFilterType::New();

  caster->SetInput( this->GetOutput() );

  typedef itk::ImageFileWriter< FloatOutputImageType > OutputImageWriterType;

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  sprintf(filename, "/tmp/ImageProjector2D3D_OUTPUT_%03d.gipl", ++n );
  writer->SetFileName( filename );
  writer->SetInput( caster->GetOutput() );

  try {
    std::cout << "AfterThreadedGenerateData: " << filename << std::endl;
    writer->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to write output to file: " << filename << "; " << err << endl;
  }

  if (n >= 21)
    exit(1);
#endif


#if 0
  OutputImagePointer     outImage2D = this->GetOutput();
  ImageRegionIterator<OutputImageType> outputIterator;

  cout << endl << "DEBUG - Output of forward projection: " << endl;
 
  outputIterator = ImageRegionIterator<OutputImageType>(outImage2D, outImage2D->GetLargestPossibleRegion());

  for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) 
      cout << outputIterator.Get() << " ";

  cout << endl;
#endif
}

} // end namespace itk


#endif
