/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-13 10:54:10 +0000 (Tue, 13 Dec 2011) $
 Revision          : $Revision: 8003 $
 Last modified by  : $Author: kkl $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkFluidPDEFilter_txx
#define __itkFluidPDEFilter_txx

#include "itkFluidPDEFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkDerivativeOperator.h"
#include "itkVectorNeighborhoodInnerProduct.h"
#include "itkPeriodicBoundaryCondition.h"

#include "itkLogHelper.h"

#ifdef CUDA_FFT
#include <cuda.h>
#include <cutil.h>
#include <cuda_runtime_api.h>
#include <cutil_inline_bankchecker.h>
#include <cutil_inline_runtime.h>
#include <cutil_inline_drvapi.h>
#include <cufft.h>
#endif

namespace itk {

template <class TScalarType, unsigned int NDimensions>
FluidPDEFilter<TScalarType, NDimensions>
::FluidPDEFilter()
{
  this->m_AdjointNavierLameOperatorInitialised = false; 
  this->m_NavierLameOperatorInitialised = false; 
  this->m_IsFFTWInitialised = false; 
  //niftkitkDebugMacro(<<"FluidPDEFilter(): no multi-threading for now (not useful in the cluster anyway.....).");
  //MultiThreader::SetGlobalMaximumNumberOfThreads(1); 
  m_fftwPlanSliceSize = -1;
  m_fftwPlanColSize = -1; 
  m_fftwPlanRowSize = -1; 
  
  // Init multi-threshold fftw. 
  // fftwf_init_threads(); 
  
  m_IsComputeVelcoity = true; 
}

template <class TScalarType, unsigned int NDimensions>
FluidPDEFilter<TScalarType, NDimensions>
::~FluidPDEFilter()
{
  fftwf_destroy_plan(this->m_fftwPlan); 
}


template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Lambda:" << m_Lambda << std::endl;
  os << indent << "Mu:" << m_Mu << std::endl;
}


template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // We need all the input.
  InputImagePointer input = const_cast<InputImageType *>(this->GetInput());
  if( !input )
    {
      return;
    }
  input->SetRequestedRegion( input->GetLargestPossibleRegion() );
}

template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  // call the superclass' implementation of this method
  Superclass::EnlargeOutputRequestedRegion(output);
  
  // generate everything in the region of interest
  output->SetRequestedRegionToLargestPossibleRegion();
}


template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::GenerateData()
{
  this->AllocateOutputs();
	
  if (this->m_IsComputeVelcoity)
  {
    if (NDimensions == 2)
    {
      CalculationVelocity2D(this->m_Lambda, this->m_Mu);
    }
    else if (NDimensions == 3)
    {
      std::cerr << "GenerateData(): solving forward...";
#ifdef CUDA_FFT
      std::cerr << "using CUDA..."; 
#else
      std::cerr << "using FFTW..."; 
#endif            
      time_t start = clock(); 
      CalculationVelocity3D(this->m_Lambda, this->m_Mu, false);
      std::cerr << "done. Time elapsed=" << (clock()-start)/CLOCKS_PER_SEC << std::endl;
      
      // No actually needed. 
      //if (this->m_IsSymmetric)
      //{
      //  niftkitkInfoMacro(<<"GenerateData(): solving backward");
      //  CalculationVelocity3D(this->m_Lambda, this->m_Mu, true);
      //}
    }
  }
  else
  {
    CalculationMomentum(this->m_Lambda, this->m_Mu); 
  }
}

template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::CalculateUnnormalised2DSineTransform(int rowSize, int colSize, float* input, float* output)
{
  //time_t time = clock(); 
  
  if (m_IsFFTWInitialised && (m_fftwPlanRowSize != rowSize || m_fftwPlanColSize != colSize))
  {
    niftkitkDebugMacro(<<"CalculateUnnormalised2DSineTransform(): FFTW plan size change - destroy plan and re-initialise");
    fftwf_destroy_plan(m_fftwPlan);
    m_IsFFTWInitialised = false; 
  }
  
  // FFTW assumes that the arrays are like input[sizeA][sizeB]. 
  // The index to an element is indexA*sizeB+indexB. 
  // The first two arguments to fftw_plan_r2r_2d is in the order in the array declaration, i.e. sizeA, sizeB.   
  if (!m_IsFFTWInitialised)
  {
    // fftwf_plan_with_nthreads(4); 
    m_fftwPlan = fftwf_plan_r2r_2d(colSize, rowSize, output, output, FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    m_IsFFTWInitialised = true; 
    m_fftwPlanColSize = colSize; 
    m_fftwPlanRowSize = rowSize; 
  }
  
  fftwf_execute_r2r(m_fftwPlan, input, output); 
  
  // fftw_destroy_plan(p2d);
  //niftkitkDebugMacro(<<"CalculateUnnormalised2DSineTransform(): time used=" << (clock()-time)/CLOCKS_PER_SEC);

#if 0
  std::cout << "sine transform" << std::endl;  
  for (int row = 0; row < rowSize; row++)
  {
    for (int col = 0; col < colSize; col++)  
    {
      std::cout << output[col*rowSize+row] << " ";
    }
    std::cout << std::endl;
  }
#endif
}

template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::CalculateUnnormalised3DSineTransform(int rowSize, int colSize, int sliceSize, float* input, float* output)
{
  // time_t time = clock(); 
  
  //niftkitkDebugMacro(<<"CalculateUnnormalised3DSineTransform(): dims=" << sliceSize <<"," << colSize << "," << rowSize);
  if (m_IsFFTWInitialised && (m_fftwPlanRowSize != rowSize || m_fftwPlanColSize != colSize || m_fftwPlanSliceSize != sliceSize))
  {
    niftkitkDebugMacro(<<"CalculateUnnormalised3DSineTransform(): FFTW plan size change - destroy plan and re-initialise");
    fftwf_destroy_plan(m_fftwPlan);
    m_IsFFTWInitialised = false; 
  }
  
  // FFTW assumes that the arrays are like input[sizeA][sizeB]. 
  // The index to an element is indexA*sizeB+indexB. 
  // The first two arguments to fftw_plan_r2r_2d is in the order in the array declaration, i.e. sizeA, sizeB.   
  if (!m_IsFFTWInitialised)
  {
    niftkitkDebugMacro(<<"CalculateUnnormalised3DSineTransform(): creating plan");
    
    // fftwf_plan_with_nthreads(4); 
    m_fftwPlan = fftwf_plan_r2r_3d(sliceSize, colSize, rowSize, output, output, FFTW_RODFT00, FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    m_IsFFTWInitialised = true; 
    m_fftwPlanSliceSize = sliceSize; 
    m_fftwPlanColSize = colSize; 
    m_fftwPlanRowSize = rowSize; 
  }
  
  if (m_fftwPlan == NULL)
  {
    niftkitkDebugMacro(<<"CalculateUnnormalised3DSineTransform(): failed to create FFTW plan.");
    itkExceptionMacro("CalculateUnnormalised3DSineTransform(): failed to create FFTW plan.")
  }
  
  fftwf_execute_r2r(m_fftwPlan, input, output); 
  
  //niftkitkDebugMacro(<<"CalculateUnnormalised3DSineTransform(): time used=" << (clock()-time)/CLOCKS_PER_SEC);
  
}

#ifdef CUDA_FFT
template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::CalculateUnnormalised3DSineTransformCUDA(int rowSize, int colSize, int sliceSize, float* input, float* output)
{
  typedef float2 Complex; 
  
  cudaSetDevice(cutGetMaxGflopsDeviceId());
  // 1D example: Input rowSize = 3 from (a,b,c). 
  //             Expand to 8 - (0,a,b,c,0,-c,-b,-a). 
  unsigned int fftSize = 2*2*2*(rowSize+1)*(colSize+1)*(sliceSize+1); 
  unsigned int memorySize = sizeof(Complex)*fftSize; 

  Complex* complexInput = (Complex*)malloc(memorySize);
  
  for (int z = 0; z < 2*(sliceSize+1); z++)
  {
    for (int y = 0; y < 2*(colSize+1); y++)
    {
      for (int x = 0; x < 2*(rowSize+1); x++)
      {
        int currentIndex = z*2*(rowSize+1)*2*(colSize+1)+y*2*(rowSize+1)+x; 
        
        if (x == 0 || y == 0 || z == 0 || x == rowSize+1 || y == colSize+1 || z == sliceSize+1)
        {
          complexInput[currentIndex].x = 0.f; 
          complexInput[currentIndex].y = 0.f; 
          continue; 
        }
        
        int inputXIndex = x-1; 
        float rowSign = 1.f; 
        if (x > rowSize+1)
        {
          rowSign = -1.f; 
          inputXIndex = 2*(rowSize+1)-1-x; 
        }
        int inputYIndex = y-1; 
        float colSign = 1.f; 
        if (y > colSize+1)
        {
          colSign = -1.f; 
          inputYIndex = 2*(colSize+1)-1-y; 
        }
        int inputZIndex = z-1; 
        float sliceSign = 1.f; 
        if (z > sliceSize+1)
        {
          sliceSign = -1.f; 
          inputZIndex = 2*(sliceSize+1)-1-z; 
        }
        
        // int inputIndex = inputZIndex*rowSize*colSize+inputYIndex*rowSize+inputXIndex; 
        int inputIndex = inputXIndex*sliceSize*colSize+inputYIndex*sliceSize+inputZIndex; 
        complexInput[currentIndex].x = rowSign*colSign*sliceSign*input[inputIndex];
        complexInput[currentIndex].y = 0.f; 
      }
    }
  }
  
  Complex* cudaComplexInput = NULL; 
  // Allocate device memory. 
  cutilSafeCall(cudaMalloc((void**)&cudaComplexInput, memorySize));
  // Copy host memory to device. 
  cutilSafeCall(cudaMemcpy(cudaComplexInput, complexInput, memorySize, cudaMemcpyHostToDevice));
  // CUFFT plan
  cufftHandle plan;
  cufftSafeCall(cufftPlan3d(&plan, 2*(rowSize+1), 2*(colSize+1), 2*(sliceSize+1), CUFFT_C2C));
  // Transform signal and kernel
  std::cout << "Transforming signal cufftExecC2C" << std::endl; 
  cufftSafeCall(cufftExecC2C(plan, (cufftComplex*)cudaComplexInput, (cufftComplex*)cudaComplexInput, CUFFT_FORWARD));
  // Copy device memory to host
  cutilSafeCall(cudaMemcpy(complexInput, cudaComplexInput, memorySize, cudaMemcpyDeviceToHost));
  
  for (int z = 0; z < sliceSize; z++)
  {
    for (int y = 0; y < colSize; y++)
    {
      for (int x = 0; x < rowSize; x++)
      {
        int currentIndex = (z+1)*2*(rowSize+1)*2*(colSize+1)+(y+1)*2*(rowSize+1)+(x+1); 
        // int outputIndex = z*rowSize*colSize+y*rowSize+x; 
        int outputIndex = x*sliceSize*colSize+y*sliceSize+z; 
        output[outputIndex] = complexInput[currentIndex].y;
      }
    }
  }
  
  // Destroy CUFFT context
  cufftSafeCall(cufftDestroy(plan));
  cutilSafeCall(cudaFree(cudaComplexInput));
  cutilDeviceReset();
  free(complexInput); 
}
#endif


template <class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::CalculateNormalised2DSineTransformUsing1DSineTransform(int rowSize, int colSize, float* input, float* output)
{
  float* rowInput = new float[rowSize];  
  float* rowOutput = new float[rowSize];  
  float* colInput = new float[colSize];  
  float* colOutput = new float[colSize];  
  // TODO: must reuse the plan later.
  fftwf_plan rowPlan = fftwf_plan_r2r_1d(rowSize, rowInput, rowOutput, FFTW_RODFT00, FFTW_ESTIMATE);
  fftwf_plan colPlan = fftwf_plan_r2r_1d(colSize, colInput, colOutput, FFTW_RODFT00, FFTW_ESTIMATE);

  for (int row = 0; row < rowSize; row++)
  {
    for (int col = 0; col < colSize; col++)  
    {
      colInput[col] = input[col*rowSize+row];
      //std::cout << colInput[col] << " ";
    }
    //std::cout << std::endl;
    fftwf_execute(colPlan); 
    for (int col = 0; col < colSize; col++)  
    {
      output[col*rowSize+row] = colOutput[col]/2.0;
    }
  }
  for (int col = 0; col < colSize; col++)  
  {
    for (int row = 0; row < rowSize; row++)
    {
      rowInput[row] = output[col*rowSize+row];
    }
    fftwf_execute(rowPlan); 
    for (int row = 0; row < rowSize; row++)
    {
      output[col*rowSize+row] = rowOutput[row]/2.0;
    }
  }
  fftwf_destroy_plan(rowPlan);
  fftwf_destroy_plan(colPlan);
  delete[] rowInput;
  delete[] rowOutput;
  delete[] colInput;
  delete[] colOutput;
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::CalculationVelocity2D(double lamda, double mu)
{
  niftkitkDebugMacro(<<"CalculationVelocity2D():Started");
  
  if (NDimensions != 2)
    itkExceptionMacro("The function CalculationVelocity2D only works in 2D.")
  
  // Input image is the "force image" 
  typename InputImageType::Pointer forceImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  
  // Output image is the "velocity image"
  typename OutputImageType::Pointer velocityImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  
  typedef itk::ConstNeighborhoodIterator< OutputImageType > NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  const typename OutputImageType::SizeType regionSize = forceImage->GetLargestPossibleRegion().GetSize();

  radius.Fill(1);

  NeighborhoodIteratorType registrationForceIterator(radius, forceImage, forceImage->GetLargestPossibleRegion());
  const typename NeighborhoodIteratorType::OffsetType offset1 = {{-1,-1}};
  const typename NeighborhoodIteratorType::OffsetType offset2 = {{ 0,-1}};
  const typename NeighborhoodIteratorType::OffsetType offset3 = {{+1,-1}};
  const typename NeighborhoodIteratorType::OffsetType offset4 = {{-1, 0}};
  const typename NeighborhoodIteratorType::OffsetType offset5 = {{ 0, 0}};
  const typename NeighborhoodIteratorType::OffsetType offset6 = {{+1, 0}};
  const typename NeighborhoodIteratorType::OffsetType offset7 = {{-1,+1}};
  const typename NeighborhoodIteratorType::OffsetType offset8 = {{ 0,+1}};
  const typename NeighborhoodIteratorType::OffsetType offset9 = {{+1,+1}};
  float* convolvedRegistrationForceX = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)));
  float* sineConvolvedRegistrationForceX = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)));
  float* convolvedRegistrationForceY = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)));
  float* sineConvolvedRegistrationForceY = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)));
  
  // Step 1. RHS Convolution in equation (14).
  registrationForceIterator.SetNeedToUseBoundaryCondition(false);
  for (registrationForceIterator.GoToBegin(); !registrationForceIterator.IsAtEnd(); ++registrationForceIterator)
  {
    const typename NeighborhoodIteratorType::IndexType currentImageIndex = registrationForceIterator.GetIndex();

    if (currentImageIndex[0] >= 1 && currentImageIndex[0] <= static_cast<int>(regionSize[0]-2) &&
        currentImageIndex[1] >= 1 && currentImageIndex[1] <= static_cast<int>(regionSize[1]-2))
    {
      const typename OutputImageType::PixelType tempPixel1 = registrationForceIterator.GetPixel(offset1);
      const typename OutputImageType::PixelType tempPixel2 = registrationForceIterator.GetPixel(offset2);
      const typename OutputImageType::PixelType tempPixel3 = registrationForceIterator.GetPixel(offset3);
      const typename OutputImageType::PixelType tempPixel4 = registrationForceIterator.GetPixel(offset4);
      const typename OutputImageType::PixelType tempPixel5 = registrationForceIterator.GetPixel(offset5);
      const typename OutputImageType::PixelType tempPixel6 = registrationForceIterator.GetPixel(offset6);
      const typename OutputImageType::PixelType tempPixel7 = registrationForceIterator.GetPixel(offset7);
      const typename OutputImageType::PixelType tempPixel8 = registrationForceIterator.GetPixel(offset8);
      const typename OutputImageType::PixelType tempPixel9 = registrationForceIterator.GetPixel(offset9);
      const int currentArrayIndex = (currentImageIndex[1]-1)*(regionSize[0]-2) + (currentImageIndex[0]-1);
      
      // Be careful that eq.(16) in Cahill ISBI paper is written the wrong way round.
      // L ajoint is multipled to b, and therefore the transpose should be on S^(0,1) instead of S(1,2).
       
      // S^(0,1) * b^(k,1)
      convolvedRegistrationForceX[currentArrayIndex] = tempPixel2[0]*(lamda+2*mu) + tempPixel4[0]*mu + tempPixel5[0]*(-2)*(lamda+3*mu) + tempPixel6[0]*mu + tempPixel8[0]*(lamda+2*mu);
      // S^(0,2) * b^(k,2)
      convolvedRegistrationForceX[currentArrayIndex] += ((lamda+mu)/4.0)*(-tempPixel1[1] + tempPixel3[1] + tempPixel7[1] - tempPixel9[1]);
	
      // S^(1,1) * b^(k,1)
      convolvedRegistrationForceY[currentArrayIndex] = ((lamda+mu)/4.0)*(-tempPixel1[0] + tempPixel3[0] + tempPixel7[0] - tempPixel9[0]);
      // S^(1,2) * b^(k,1) 
      convolvedRegistrationForceY[currentArrayIndex] += tempPixel2[1]*mu + tempPixel4[1]*(lamda+2*mu) + tempPixel5[1]*(-2)*(lamda+3*mu) + tempPixel6[1]*(lamda+2*mu) + tempPixel8[1]*mu;
    }
  }

  // Step 2. Sine transform of the result of step 1. 
  CalculateUnnormalised2DSineTransform(regionSize[0]-2, regionSize[1]-2, convolvedRegistrationForceX, sineConvolvedRegistrationForceX);
  CalculateUnnormalised2DSineTransform(regionSize[0]-2, regionSize[1]-2, convolvedRegistrationForceY, sineConvolvedRegistrationForceY);

  fftwf_free(convolvedRegistrationForceX);
  fftwf_free(convolvedRegistrationForceY);

  const double forwardSineTransformFactor = 4.0;
  const double factor = 4.0*mu*(lamda+2*mu);
  const double pi = 3.14159265359;
  float* velocityX = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)));
  float* velocityY = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)));
  
  // Step 3. Divide each component by beta.  
  for (int indexX = 1; indexX <= static_cast<int>(regionSize[0]-2); indexX++)
  {
    for (int indexY = 1; indexY <= static_cast<int>(regionSize[1]-2); indexY++)
    {
      double beta = 0.0;
      const int currentArrayIndex = (indexY-1)*(regionSize[0]-2) + (indexX-1);
       
      beta = cos(pi*indexX/(double)(regionSize[0]-1)) + cos(pi*indexY/(double)(regionSize[1]-1)) - 2.0;
      beta = factor*beta*beta;
      sineConvolvedRegistrationForceX[currentArrayIndex] = sineConvolvedRegistrationForceX[currentArrayIndex]/(beta*forwardSineTransformFactor);
      sineConvolvedRegistrationForceY[currentArrayIndex] = sineConvolvedRegistrationForceY[currentArrayIndex]/(beta*forwardSineTransformFactor);
    }
  }
  
  // Step 4. Compute the sine transform again. 
  CalculateUnnormalised2DSineTransform(regionSize[0]-2, regionSize[1]-2, sineConvolvedRegistrationForceX, velocityX);
  CalculateUnnormalised2DSineTransform(regionSize[0]-2, regionSize[1]-2, sineConvolvedRegistrationForceY, velocityY);

  fftwf_free(sineConvolvedRegistrationForceX);
  fftwf_free(sineConvolvedRegistrationForceY);
  
  typedef ImageRegionIteratorWithIndex < OutputImageType > VelocityIteratorType;
  typename OutputImageType::PixelType velocityValue;   
  
  VelocityIteratorType velocityIterator(velocityImage, velocityImage->GetLargestPossibleRegion());
  const double inverseSineTransformFactor = ((double)regionSize[0]-1.0)*((double)regionSize[1]-1.0); // 4.0*((double)regionSize[0]-1.0)*((double)regionSize[1]-1.0)/4.0;
  const double velocityFactor = 1.0;
  double maxVelocity = 0.0;  
  
  for (velocityIterator.GoToBegin(); !velocityIterator.IsAtEnd(); ++velocityIterator)
  {
    const typename OutputImageType::IndexType index = velocityIterator.GetIndex();
    
    velocityValue[0] = 0.0;
    velocityValue[1] = 0.0;
    if (index[0] >= 1 && index[0] <= static_cast<int>(regionSize[0]-2) &&
        index[1] >= 1 && index[1] <= static_cast<int>(regionSize[1]-2))
    {
      velocityValue[0] = velocityFactor*velocityX[(index[1]-1)*(regionSize[0]-2)+index[0]-1]/inverseSineTransformFactor;
      velocityValue[1] = velocityFactor*velocityY[(index[1]-1)*(regionSize[0]-2)+index[0]-1]/inverseSineTransformFactor;
      
      if (fabs(velocityValue[0]) > maxVelocity)
        maxVelocity = fabs(velocityValue[0]);
      if (fabs(velocityValue[1]) > maxVelocity)
        maxVelocity = fabs(velocityValue[1]);
    }
    velocityIterator.Set(velocityValue);
  }
  
  //std::cout << "maxVelocity=" << maxVelocity << std::endl;
  
  fftwf_free(velocityX);
  fftwf_free(velocityY);

  niftkitkDebugMacro(<<"CalculationVelocity2D():Finished");
} 


template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator11(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // mu*x^2
  order[0] = 2;
  order[1] = 0;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(mu);
  this->m_AdjointNavierLameOperator[0][0].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*y^2
  order[0] = 0;
  order[1] = 2;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2.0*mu);
  this->m_AdjointNavierLameOperator[0][0].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*(z^2)
  order[0] = 0;
  order[1] = 0;
  order[2] = 2;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2.0*mu);
  this->m_AdjointNavierLameOperator[0][0].AddSingleDerivativeTerm(term);
  
  this->m_AdjointNavierLameOperator[0][0].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator12(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // -(lambda+mu)*x*y
  order[0] = 1;
  order[1] = 1;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(-(lambda+mu));
  this->m_AdjointNavierLameOperator[0][1].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[0][1].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator13(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // -(lambda+mu)*x*z  
  order[0] = 1;
  order[1] = 0;
  order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(-(lambda+mu));
  this->m_AdjointNavierLameOperator[0][2].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[0][2].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator21(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
        
  // -(lambda+mu)*x*y
  order[0] = 1;
  order[1] = 1;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(-(lambda+mu));
  this->m_AdjointNavierLameOperator[1][0].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[1][0].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator22(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
                
  // mu*y^2
  order[0] = 0;
  order[1] = 2;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(mu);
  this->m_AdjointNavierLameOperator[1][1].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*x^2
  order[0] = 2;
  order[1] = 0;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2*mu);
  this->m_AdjointNavierLameOperator[1][1].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*(z^2)
  order[0] = 0;
  order[1] = 0;
  order[2] = 2;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2*mu);
  this->m_AdjointNavierLameOperator[1][1].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[1][1].CreateToRadius(1);
}

        
template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator23(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
        
  // -(lambda+mu)*y*z
  order[0] = 0;
  order[1] = 1;
  order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(-(lambda+mu));
  this->m_AdjointNavierLameOperator[1][2].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[1][2].CreateToRadius(1);
}
        
template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator31(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
        
  // -(lambda+mu)*x*z
  order[0] = 1;
  order[1] = 0;
  order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(-(lambda+mu));
  this->m_AdjointNavierLameOperator[2][0].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[2][0].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator32(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
                
  // -(lambda+mu)*y*z
  order[0] = 0;
  order[1] = 1;
  order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(-(lambda+mu));
  this->m_AdjointNavierLameOperator[2][1].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[2][1].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator33(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // mu*z^2
  order[0] = 0;
  order[1] = 0;
  order[2] = 2;
  term.SetDervativeOrder(order);
  term.SetConstant(mu);
  this->m_AdjointNavierLameOperator[2][2].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*x^2
  order[0] = 2;
  order[1] = 0;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2*mu);
  this->m_AdjointNavierLameOperator[2][2].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*(y^2)
  order[0] = 0;
  order[1] = 2;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2*mu);
  this->m_AdjointNavierLameOperator[2][2].AddSingleDerivativeTerm(term);
          
  this->m_AdjointNavierLameOperator[2][2].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeAdjointNavierLameOperator(double lambda, double mu)
{
  // Only do once. 
  if (!this->m_AdjointNavierLameOperatorInitialised)
  {
    ComputeAdjointNavierLameOperator11(lambda, mu);
    ComputeAdjointNavierLameOperator12(lambda, mu);
    ComputeAdjointNavierLameOperator13(lambda, mu);
    ComputeAdjointNavierLameOperator21(lambda, mu);
    ComputeAdjointNavierLameOperator22(lambda, mu);
    ComputeAdjointNavierLameOperator23(lambda, mu);
    ComputeAdjointNavierLameOperator31(lambda, mu);
    ComputeAdjointNavierLameOperator32(lambda, mu);
    ComputeAdjointNavierLameOperator33(lambda, mu);
    this->m_AdjointNavierLameOperatorInitialised = true; 
  }
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator(double lambda, double mu)
{
  // Only do once. 
  if (!this->m_NavierLameOperatorInitialised)
  {
    ComputeNavierLameOperator11(lambda, mu);
    ComputeNavierLameOperator12(lambda, mu);
    ComputeNavierLameOperator21(lambda, mu);
    ComputeNavierLameOperator22(lambda, mu);
    if (NDimensions > 2)
    {
      ComputeNavierLameOperator13(lambda, mu);
      ComputeNavierLameOperator23(lambda, mu);
      ComputeNavierLameOperator31(lambda, mu);
      ComputeNavierLameOperator32(lambda, mu);
      ComputeNavierLameOperator33(lambda, mu);
    }
    this->m_NavierLameOperatorInitialised = true; 
  }
}


template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::CalculationVelocity3D(double lambda, double mu, bool isDoingBackward)
{
  niftkitkDebugMacro(<<"CalculationVelocity3D():Started");
  
  if (NDimensions != 3)
    itkExceptionMacro("The function CalculationVelocity3D only works in 3D.")
  
  // Input image is the "force image" 
  typename InputImageType::Pointer forceImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  
  // Reverse the force image to solve it the other way, because the fluid solve is not entirely symmetric. 
  if (isDoingBackward)
  {
    typedef ImageRegionIterator<InputImageType> ForceImageRegionIteratorType; 
    ForceImageRegionIteratorType forceImageRegionIterator(forceImage, forceImage->GetLargestPossibleRegion()); 
    for (forceImageRegionIterator.GoToBegin(); !forceImageRegionIterator.IsAtEnd(); ++forceImageRegionIterator)
    {
      forceImageRegionIterator.Set(-forceImageRegionIterator.Get()); 
    }
  }
  
  // Output image is the "velocity image"
  typename OutputImageType::Pointer velocityImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  
  typedef itk::ConstNeighborhoodIterator< OutputImageType > NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  const typename OutputImageType::SizeType regionSize = forceImage->GetLargestPossibleRegion().GetSize();

  radius.Fill(1);
  
  float* convolvedRegistrationForceX = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  float* sineConvolvedRegistrationForceX = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  float* convolvedRegistrationForceY = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  float* sineConvolvedRegistrationForceY = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  float* convolvedRegistrationForceZ = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  float* sineConvolvedRegistrationForceZ = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  
  if (convolvedRegistrationForceX == NULL || sineConvolvedRegistrationForceX == NULL ||
      convolvedRegistrationForceY == NULL || sineConvolvedRegistrationForceY == NULL ||  
      convolvedRegistrationForceZ == NULL || sineConvolvedRegistrationForceZ == NULL)
  {
    itkExceptionMacro("CalculationVelocity3D: failed to allocate memory.")
  }

  // Stop 0. Prepare the finite difference operators. 
  ComputeAdjointNavierLameOperator(lambda, mu);
  niftkitkDebugMacro(<<"CalculationVelocity3D():regionSize=" << regionSize);

  // Step 1. RHS Convolution in equation (14).
  NeighborhoodIteratorType registrationForceIterator(radius, forceImage, forceImage->GetLargestPossibleRegion());
  typename NeighborhoodIteratorType::ConstIterator neighbourhoodIterator; 
  typename NondirectionalDerivativeOperatorType::ConstIterator adjointNavierLameOperatorIterator[3][3]; 
  for (registrationForceIterator.GoToBegin(); !registrationForceIterator.IsAtEnd(); ++registrationForceIterator)
  {
    const typename NeighborhoodIteratorType::IndexType currentImageIndex = registrationForceIterator.GetIndex();
    const int currentArrayIndex =  (currentImageIndex[2]-1)*(regionSize[1]-2)*(regionSize[0]-2) + (currentImageIndex[1]-1)*(regionSize[0]-2) + (currentImageIndex[0]-1);
      
    if (currentImageIndex[0] >= 1 && currentImageIndex[0] <= static_cast<int>(regionSize[0]-2) &&
        currentImageIndex[1] >= 1 && currentImageIndex[1] <= static_cast<int>(regionSize[1]-2) && 
        currentImageIndex[2] >= 1 && currentImageIndex[2] <= static_cast<int>(regionSize[2]-2))    
    {
      //niftkitkDebugMacro(<<"CalculationVelocity3D():currentArrayIndex=" << currentArrayIndex);
      convolvedRegistrationForceX[currentArrayIndex] = 0.0;
      convolvedRegistrationForceY[currentArrayIndex] = 0.0;
      convolvedRegistrationForceZ[currentArrayIndex] = 0.0;
      
      adjointNavierLameOperatorIterator[0][0] = m_AdjointNavierLameOperator[0][0].Begin(); 
      adjointNavierLameOperatorIterator[0][1] = m_AdjointNavierLameOperator[0][1].Begin(); 
      adjointNavierLameOperatorIterator[0][2] = m_AdjointNavierLameOperator[0][2].Begin(); 
      adjointNavierLameOperatorIterator[1][0] = m_AdjointNavierLameOperator[1][0].Begin(); 
      adjointNavierLameOperatorIterator[1][1] = m_AdjointNavierLameOperator[1][1].Begin(); 
      adjointNavierLameOperatorIterator[1][2] = m_AdjointNavierLameOperator[1][2].Begin(); 
      adjointNavierLameOperatorIterator[2][0] = m_AdjointNavierLameOperator[2][0].Begin(); 
      adjointNavierLameOperatorIterator[2][1] = m_AdjointNavierLameOperator[2][1].Begin(); 
      adjointNavierLameOperatorIterator[2][2] = m_AdjointNavierLameOperator[2][2].Begin(); 
      
      for (neighbourhoodIterator = registrationForceIterator.Begin();
           neighbourhoodIterator != registrationForceIterator.End(); 
           ++neighbourhoodIterator)
      {
        const typename OutputImageType::PixelType forcePixel = *(*neighbourhoodIterator); 
        
        convolvedRegistrationForceX[currentArrayIndex] += forcePixel[0]*(*adjointNavierLameOperatorIterator[0][0]);
        convolvedRegistrationForceX[currentArrayIndex] += forcePixel[1]*(*adjointNavierLameOperatorIterator[0][1]);
        convolvedRegistrationForceX[currentArrayIndex] += forcePixel[2]*(*adjointNavierLameOperatorIterator[0][2]);
      
        convolvedRegistrationForceY[currentArrayIndex] += forcePixel[0]*(*adjointNavierLameOperatorIterator[1][0]);
        convolvedRegistrationForceY[currentArrayIndex] += forcePixel[1]*(*adjointNavierLameOperatorIterator[1][1]);
        convolvedRegistrationForceY[currentArrayIndex] += forcePixel[2]*(*adjointNavierLameOperatorIterator[1][2]);
        
        convolvedRegistrationForceZ[currentArrayIndex] += forcePixel[0]*(*adjointNavierLameOperatorIterator[2][0]);
        convolvedRegistrationForceZ[currentArrayIndex] += forcePixel[1]*(*adjointNavierLameOperatorIterator[2][1]);
        convolvedRegistrationForceZ[currentArrayIndex] += forcePixel[2]*(*adjointNavierLameOperatorIterator[2][2]);
        
        ++adjointNavierLameOperatorIterator[0][0]; 
        ++adjointNavierLameOperatorIterator[0][1]; 
        ++adjointNavierLameOperatorIterator[0][2]; 
        ++adjointNavierLameOperatorIterator[1][0]; 
        ++adjointNavierLameOperatorIterator[1][1]; 
        ++adjointNavierLameOperatorIterator[1][2]; 
        ++adjointNavierLameOperatorIterator[2][0]; 
        ++adjointNavierLameOperatorIterator[2][1]; 
        ++adjointNavierLameOperatorIterator[2][2]; 
      }
    }
  }
  niftkitkDebugMacro(<<"CalculationVelocity3D(): convolution done");
  
  // Step 2. Sine transform of the result of step 1. 
#ifdef CUDA_FFT  
  CalculateUnnormalised3DSineTransformCUDA(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, convolvedRegistrationForceX, sineConvolvedRegistrationForceX);
  CalculateUnnormalised3DSineTransformCUDA(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, convolvedRegistrationForceY, sineConvolvedRegistrationForceY);
  CalculateUnnormalised3DSineTransformCUDA(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, convolvedRegistrationForceZ, sineConvolvedRegistrationForceZ);
#else  
  CalculateUnnormalised3DSineTransform(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, convolvedRegistrationForceX, sineConvolvedRegistrationForceX);
  CalculateUnnormalised3DSineTransform(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, convolvedRegistrationForceY, sineConvolvedRegistrationForceY);
  CalculateUnnormalised3DSineTransform(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, convolvedRegistrationForceZ, sineConvolvedRegistrationForceZ);
#endif   
  
  fftwf_free(convolvedRegistrationForceX);
  fftwf_free(convolvedRegistrationForceY);
  fftwf_free(convolvedRegistrationForceZ);
	
  const double forwardSineTransformFactor = 8.0;
  const double factor = 4.0*mu*(lambda+2.0*mu); 
  const double pi = 3.14159265359;
  float* velocityX = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  float* velocityY = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  float* velocityZ = static_cast<float*>(fftwf_malloc(sizeof(float)*(regionSize[0]-2)*(regionSize[1]-2)*(regionSize[2]-2)));
  
  // Step 3. Divide each component by beta.  
  for (int indexX = 1; indexX <= static_cast<int>(regionSize[0]-2); indexX++)
  {
    for (int indexY = 1; indexY <= static_cast<int>(regionSize[1]-2); indexY++)
    {
			for (int indexZ = 1; indexZ <= static_cast<int>(regionSize[2]-2); indexZ++)
			{
				double beta = 0.0;
				const int currentArrayIndex = (indexZ-1)*(regionSize[1]-2)*(regionSize[0]-2) + (indexY-1)*(regionSize[0]-2) + (indexX-1);
			
				beta = cos(pi*indexX/(double)(regionSize[0]-1)) + cos(pi*indexY/(double)(regionSize[1]-1)) + cos(pi*indexZ/(double)(regionSize[2]-1)) - 3.0;
				beta = factor*beta*beta;
				sineConvolvedRegistrationForceX[currentArrayIndex] = sineConvolvedRegistrationForceX[currentArrayIndex]/(beta*forwardSineTransformFactor);
				sineConvolvedRegistrationForceY[currentArrayIndex] = sineConvolvedRegistrationForceY[currentArrayIndex]/(beta*forwardSineTransformFactor);
				sineConvolvedRegistrationForceZ[currentArrayIndex] = sineConvolvedRegistrationForceZ[currentArrayIndex]/(beta*forwardSineTransformFactor);
			}
    }
  }
  
  // Step 4. Compute the sine transform again. 
#ifdef CUDA_FFT  
  CalculateUnnormalised3DSineTransformCUDA(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, sineConvolvedRegistrationForceX, velocityX);
  CalculateUnnormalised3DSineTransformCUDA(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, sineConvolvedRegistrationForceY, velocityY);
  CalculateUnnormalised3DSineTransformCUDA(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, sineConvolvedRegistrationForceZ, velocityZ);
#else  
  CalculateUnnormalised3DSineTransform(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, sineConvolvedRegistrationForceX, velocityX);
  CalculateUnnormalised3DSineTransform(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, sineConvolvedRegistrationForceY, velocityY);
  CalculateUnnormalised3DSineTransform(regionSize[0]-2, regionSize[1]-2, regionSize[2]-2, sineConvolvedRegistrationForceZ, velocityZ);
#endif 
	
  fftwf_free(sineConvolvedRegistrationForceX);
  fftwf_free(sineConvolvedRegistrationForceY);
  fftwf_free(sineConvolvedRegistrationForceZ);
  
  typedef ImageRegionIteratorWithIndex < OutputImageType > VelocityIteratorType;
  typename OutputImageType::PixelType velocityValue;   
  
  VelocityIteratorType velocityIterator(velocityImage, velocityImage->GetLargestPossibleRegion());
  const double inverseSineTransformFactor = ((double)regionSize[0]-1.0)*((double)regionSize[1]-1.0)*((double)regionSize[2]-1.0);
  const double velocityFactor = 1.0;
  
  for (velocityIterator.GoToBegin(); !velocityIterator.IsAtEnd(); ++velocityIterator)
  {
    const typename OutputImageType::IndexType index = velocityIterator.GetIndex();
    
    velocityValue[0] = 0.0;
    velocityValue[1] = 0.0;
    velocityValue[2] = 0.0;
    
    if (index[0] >= 1 && index[0] <= static_cast<int>(regionSize[0]-2) &&
        index[1] >= 1 && index[1] <= static_cast<int>(regionSize[1]-2) &&
  			index[2] >= 1 && index[2] <= static_cast<int>(regionSize[2]-2))
    {
      velocityValue[0] = velocityFactor*velocityX[(index[2]-1)*(regionSize[1]-2)*(regionSize[0]-2)+(index[1]-1)*(regionSize[0]-2)+index[0]-1]/inverseSineTransformFactor;
      velocityValue[1] = velocityFactor*velocityY[(index[2]-1)*(regionSize[1]-2)*(regionSize[0]-2)+(index[1]-1)*(regionSize[0]-2)+index[0]-1]/inverseSineTransformFactor;
      velocityValue[2] = velocityFactor*velocityZ[(index[2]-1)*(regionSize[1]-2)*(regionSize[0]-2)+(index[1]-1)*(regionSize[0]-2)+index[0]-1]/inverseSineTransformFactor;
      
    }
    if (!isDoingBackward)
    {
      velocityIterator.Set(velocityValue);
    }
    else
    {
      // Average of the forward and backward. 
      velocityIterator.Set((velocityIterator.Get()-velocityValue)/2.); 
    }
  }
  
  fftwf_free(velocityX);
  fftwf_free(velocityY);
  fftwf_free(velocityZ);
	
  niftkitkDebugMacro(<<"CalculationVelocity3D():Finished");
}



template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator11(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (lambda+2*mu)*x^2
  order[0] = 2;
  order[1] = 0;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2.0*mu);
  this->m_NavierLameOperator[0][0].AddSingleDerivativeTerm(term);
  // (mu)*y^2
  order[0] = 0;
  order[1] = 2;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(mu);
  this->m_NavierLameOperator[0][0].AddSingleDerivativeTerm(term);
  // (mu)*(z^2)
  if (NDimensions > 2)
  {
    order[0] = 0;
    order[1] = 0;
    order[2] = 2;
    term.SetDervativeOrder(order);
    term.SetConstant(mu);
    this->m_NavierLameOperator[0][0].AddSingleDerivativeTerm(term);
  }
  
  this->m_NavierLameOperator[0][0].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator12(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (lambda+mu)*xy
  order[0] = 1;
  order[1] = 1;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+mu);
  this->m_NavierLameOperator[0][1].AddSingleDerivativeTerm(term);
  
  this->m_NavierLameOperator[0][1].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator13(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (lambda+mu)*xz
  order[0] = 1;
  order[1] = 0;
  if (NDimensions > 2)
    order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+mu);
  this->m_NavierLameOperator[0][2].AddSingleDerivativeTerm(term);
  
  this->m_NavierLameOperator[0][2].CreateToRadius(1);
}


template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator21(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (lambda+mu)*yx
  order[0] = 1;
  order[1] = 1;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+mu);
  this->m_NavierLameOperator[1][0].AddSingleDerivativeTerm(term);
  
  this->m_NavierLameOperator[1][0].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator22(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (mu)*x^2
  order[0] = 2;
  order[1] = 0;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(mu);
  this->m_NavierLameOperator[1][1].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*y^2
  order[0] = 0;
  order[1] = 2;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+2.0*mu);
  this->m_NavierLameOperator[1][1].AddSingleDerivativeTerm(term);
  // (mu)*(z^2)
  if (NDimensions > 2)
  {
    order[0] = 0;
    order[1] = 0;
    order[2] = 2;
    term.SetDervativeOrder(order);
    term.SetConstant(mu);
    this->m_NavierLameOperator[1][1].AddSingleDerivativeTerm(term);
  }
  
  this->m_NavierLameOperator[1][1].CreateToRadius(1);
}


template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator23(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (lambda+mu)*yz
  order[0] = 0;
  order[1] = 1;
  if (NDimensions > 2)
    order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+mu);
  this->m_NavierLameOperator[1][2].AddSingleDerivativeTerm(term);
  
  this->m_NavierLameOperator[1][2].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator31(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (lambda+mu)*zx
  order[0] = 1;
  order[1] = 0;
  if (NDimensions > 2)
    order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+mu);
  this->m_NavierLameOperator[2][0].AddSingleDerivativeTerm(term);
  
  this->m_NavierLameOperator[2][0].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator32(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (lambda+mu)*zy
  order[0] = 0;
  order[1] = 1;
  if (NDimensions > 2)
    order[2] = 1;
  term.SetDervativeOrder(order);
  term.SetConstant(lambda+mu);
  this->m_NavierLameOperator[2][1].AddSingleDerivativeTerm(term);
  
  this->m_NavierLameOperator[2][1].CreateToRadius(1);
}

template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::ComputeNavierLameOperator33(double lambda, double mu)
{
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  typename NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  // (mu)*x^2
  order[0] = 2;
  order[1] = 0;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(mu);
  this->m_NavierLameOperator[2][2].AddSingleDerivativeTerm(term);
  // (mu)*y^2
  order[0] = 0;
  order[1] = 2;
  if (NDimensions > 2)
    order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(mu);
  this->m_NavierLameOperator[2][2].AddSingleDerivativeTerm(term);
  // (lambda+2*mu)*(z^2)
  if (NDimensions > 2)
  {
    order[0] = 0;
    order[1] = 0;
    order[2] = 2;
    term.SetDervativeOrder(order);
    term.SetConstant(lambda+2.0*mu);
    this->m_NavierLameOperator[2][2].AddSingleDerivativeTerm(term);
  }
  
  this->m_NavierLameOperator[2][2].CreateToRadius(1);
}


template<class TScalarType, unsigned int NDimensions>
void
FluidPDEFilter<TScalarType, NDimensions>
::CalculationMomentum(double lambda, double mu)
{
  const int dimensions = static_cast<int>(NDimensions); 
  niftkitkDebugMacro(<<"CalculationMomentum():Started");
  
  // Input image is the "velocity image" 
  typename InputImageType::Pointer velocityImage = static_cast< InputImageType * >(this->ProcessObject::GetInput(0));
  
  // Output image is the "momentum image"
  typename OutputImageType::Pointer momentumImage = static_cast< OutputImageType * >(this->ProcessObject::GetOutput(0));
  
  typedef itk::ConstNeighborhoodIterator< OutputImageType > NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  const typename OutputImageType::SizeType regionSize = velocityImage->GetLargestPossibleRegion().GetSize();

  radius.Fill(1);
  
  // Prepare the finite difference operators. 
  ComputeNavierLameOperator(lambda, mu);
  typename NondirectionalDerivativeOperatorType::ConstIterator navierLameOperatorIterator[NDimensions][NDimensions]; 

  // Apply the operators. 
  NeighborhoodIteratorType velocityIterator(radius, velocityImage, velocityImage->GetLargestPossibleRegion());
  typename NeighborhoodIteratorType::ConstIterator neighbourhoodIterator; 
  typedef ImageRegionIteratorWithIndex<OutputImageType> MomentumIteratorType;
  MomentumIteratorType momentumIterator(momentumImage, momentumImage->GetLargestPossibleRegion()); 
  for (velocityIterator.GoToBegin(), momentumIterator.GoToBegin(); 
       !velocityIterator.IsAtEnd(); 
       ++velocityIterator, ++momentumIterator)
  {
    const typename NeighborhoodIteratorType::IndexType currentImageIndex = velocityIterator.GetIndex();
    typename OutputImageType::PixelType outputValue; 
    outputValue.Fill(0.); 
      
    bool isBoundary = false; 
    for (int i = 0; i < dimensions; i++)
    {
      if (currentImageIndex[i] < 1 || currentImageIndex[i] > static_cast<int>(regionSize[i]-2)) 
          isBoundary = true; 
    }
    
    if (!isBoundary)
    {
      for (int row = 0; row < dimensions; row++)
      {
        for (int col = 0; col < dimensions; col++)
        {
          navierLameOperatorIterator[row][col] = this->m_NavierLameOperator[row][col].Begin(); 
        }
      }
      
      for (neighbourhoodIterator = velocityIterator.Begin();
           neighbourhoodIterator != velocityIterator.End(); 
           ++neighbourhoodIterator)
      {
        const typename OutputImageType::PixelType velocityPixel = *(*neighbourhoodIterator); 
        
        for (int row = 0; row < dimensions; row++)
        {
          for (int col = 0; col < dimensions; col++)
          {
            outputValue[row] += velocityPixel[col]*(*navierLameOperatorIterator[row][col]); 
            // niftkitkDebugMacro(<<"CalculationMomentum():outputValue=" << outputValue << ",velocityPixel=" << velocityPixel << ",navierLameOperatorIterator[row][col]=" << *navierLameOperatorIterator[row][col]);
          }
        }
        for (int row = 0; row < dimensions; row++)
        {
          for (int col = 0; col < dimensions; col++)
          {
            ++navierLameOperatorIterator[row][col]; 
          }
        }
        
      }
    }
    
    momentumIterator.Set(outputValue); 
  }
  niftkitkDebugMacro(<<"CalculationMomentum(): convolution done");
}
  




} // end namespace itk

#endif


