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
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include "ConversionUtils.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTranslationTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkNMIImageToImageMetric.h"
#include "itkNMILocalHistogramDerivativeForceFilter.h"
#include "itkFluidPDEFilter.h"

#include "itkImageRegionIteratorWithIndex.h"
#include "fftw3.h"

#define pi (3.14159265359)
#define tolerance (0.001) 
#define Dimension (2)
#define Dimension3D (3)

/**
 * 
 */
class FluidPDEFilterUnitTest
{
public:
  /**
   * Typdefs.  
   */
  typedef vnl_vector<double> VnlVectorType;
  typedef vnl_matrix< double > VnlMatrixType; 
  typedef itk::FluidPDEFilter< double, Dimension > FluidPDEFilterType;
  typedef itk::FluidPDEFilter< double, Dimension3D > FluidPDEFilter3DType;
  /**
   * 1D sine transform tests.
   */
  int Test1DSineTransform()
  {
    VnlVectorType data1D(13);
    VnlVectorType results1D(13);
  
    data1D.fill(0);
    results1D.fill(0);
    for (unsigned int i = 1; i <= data1D.size()-2; i++)  
    {
      data1D[i] = rand()%100;
    }
    
    for (unsigned int m = 1; m <= data1D.size()-2; m++)
    {
      for (unsigned int x = 1; x <= data1D.size()-2; x++)
      {
        results1D[m] += data1D[x]*sin(pi*x*m/(double)(data1D.size()-1));
      }
    }
    std::cout << "1D expected results:" << std::endl;
    std::cout << results1D << std::endl;  
  
    fftwf_plan p;
    int N=data1D.size()-2;
    float* in = new float[N];
    float* out = new float[N];
  
    for (int i = 0; i < N ; i++)  
    {
      in[i] = data1D[i+1]; 
    }
    p = fftwf_plan_r2r_1d(N, in, out, FFTW_RODFT00, FFTW_ESTIMATE);
    fftwf_execute(p); 
    std::cout << "out:" << std::endl;
    for (int i = 0; i < N ; i++)  
    {
      std::cout << out[i]/2.0 << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < N ; i++)  
    {
      if (fabs(results1D[i+1]-out[i]/2.0) > tolerance)
        return EXIT_FAILURE;
    }
    fftwf_destroy_plan(p);
    delete[] in; 
    delete[] out;
    
    return EXIT_SUCCESS;
  }
  /**
   * 2D sine transform tests.
   */
  int Test2DSineTransform()
  {
    FluidPDEFilterType::Pointer fluidTransform = FluidPDEFilterType::New();
    int rowSize2D = 15;
    int colSize2D = 6;
    VnlMatrixType data2D(rowSize2D, colSize2D);
    VnlMatrixType results2D(rowSize2D, colSize2D);
    
    data2D.fill(0);
    for (int row = 1; row < rowSize2D-1; row++)
    {
      for (int col = 1; col < colSize2D-1; col++)
      {
        data2D(row,col) = rand()%30;
      }
    }
    std::cout << "input:" << std::endl << data2D << std::endl;
    results2D.fill(0);
    for (int m = 1; m <= rowSize2D-2; m++)
    {
      for (int n = 1; n <= colSize2D-2; n++)
      {
        for (int row = 1; row <= rowSize2D-2; row++)
        {
          for (int col = 1; col <= colSize2D-2; col++)
          {
             results2D(m,n) += data2D(row,col)*sin(pi*row*m/(double)(rowSize2D-1))*sin(pi*col*n/(double)(colSize2D-1));
          }
        }  
      }
    }
    std::cout << "2D expected resutls:" << std::endl;
    std::cout << results2D << std::endl; 
  
    int N2dX = colSize2D-2;
    int N2dY = rowSize2D-2;
    float* in2d = new float[N2dX*N2dY];
    float* in2dCopy = new float[N2dX*N2dY];
    float* out2d = new float[N2dX*N2dY];
  
    std::cout << "2D inputs:" << std::endl;
    for (int y = 0; y < N2dY; y++)
    {
      for (int x = 0; x < N2dX; x++)
      {
        // row->y, col->x.
        in2d[y*N2dX+x] = data2D(y+1,x+1);
        in2dCopy[y*N2dX+x] = in2d[y*N2dX+x];
        std::cout << in2d[y*N2dX+x] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    fluidTransform->CalculateUnnormalised2DSineTransform(N2dX, N2dY, in2d, out2d);
    std::cout << "out2d:" << std::endl;
    for (int y = 0; y < N2dY; y++)
    {
      for (int x = 0; x < N2dX; x++)
      {
        std::cout << out2d[y*N2dX+x]/4.0 << " ";
        in2d[y*N2dX+x] = out2d[y*N2dX+x]/4.0;
        if (fabs(results2D(y+1,x+1)-out2d[y*N2dX+x]/4.0) > tolerance)
          return EXIT_FAILURE;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    fluidTransform->CalculateUnnormalised2DSineTransform(N2dX, N2dY, in2d, out2d);
    std::cout << "inverse out2d:" << std::endl;
    for (int y = 0; y < N2dY; y++)
    {
      for (int x = 0; x < N2dX; x++)
      {
        out2d[y*N2dX+x] = out2d[y*N2dX+x]/4.0/((N2dX+1)*(N2dY+1)/4.0);
        std::cout << out2d[y*N2dX+x] << " ";
        if (fabs(data2D(y+1,x+1)-out2d[y*N2dX+x]) > tolerance)
          return EXIT_FAILURE;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  
    fluidTransform->CalculateNormalised2DSineTransformUsing1DSineTransform(N2dX, N2dY, in2dCopy, out2d);
    std::cout << "out2d from two 1d sine:" << std::endl;
    for (int y = 0; y < N2dY; y++)
    {
      for (int x = 0; x < N2dX; x++)
      {
        std::cout << out2d[y*N2dX+x] << " ";
        if (fabs(results2D(y+1,x+1)-out2d[y*N2dX+x]) > tolerance)
          return EXIT_FAILURE;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    fluidTransform->CalculateNormalised2DSineTransformUsing1DSineTransform(N2dX, N2dY, out2d, in2d);
    std::cout << "new invert:" << std::endl;
    for (int y = 0; y < N2dY; y++)
    {
      for (int x = 0; x < N2dX; x++)
      {
        std::cout << in2d[y*N2dX+x]/((N2dX+1)*(N2dY+1)/4.0) << " ";
        if (fabs(data2D(y+1,x+1)-in2d[y*N2dX+x]/((N2dX+1)*(N2dY+1)/4.0)) > tolerance)
          return EXIT_FAILURE;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  
    delete[] in2d; 
    delete[] out2d;
    delete[] in2dCopy;
  
    return EXIT_SUCCESS;
  }
	
  /**
	 * 3D sine transform tests.
   */
  int Test3DSineTransform()
  {
    FluidPDEFilterType::Pointer fluidTransform = FluidPDEFilterType::New();
    const int rowSize = 15;
    const int colSize = 6;
		const int sliceSize = 10;
    double data[sliceSize][rowSize][colSize];
    double results[sliceSize][rowSize][colSize];
    
    std::cout << "3D data:" << std::endl;
    for (int slice = 0; slice < sliceSize; slice++)
    {
      for (int row = 0; row < rowSize; row++)
      {
        for (int col = 0; col < colSize; col++)
        {
					if (row == 0 || row == rowSize-1 ||
							col == 0 || col == colSize-1 ||
							slice == 0 || slice == sliceSize-1)
					{
						data[slice][row][col] = 0.0;
					}
					else
					{
						data[slice][row][col] = rand()%30;
					}
          std::cout << data[slice][row][col] << " "; 
				}
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "3D results:" << std::endl;
    for (int p = 0; p < sliceSize; p++)
    {
      for (int m = 0; m < rowSize; m++)
      {
        for (int n = 0; n < colSize; n++)
        {
					results[p][m][n] = 0.0;
					if (m != 0 && m != rowSize-1 &&
							n != 0 && n != colSize-1 &&
							p != 0 && p != sliceSize-1)
					{
						for (int row = 1; row <= rowSize-2; row++)
						{
							for (int col = 1; col <= colSize-2; col++)
							{
								for (int slice = 1; slice <= sliceSize-2; slice++)
								{
									results[p][m][n] += data[slice][row][col]*sin(pi*row*m/(double)(rowSize-1))*sin(pi*col*n/(double)(colSize-1))*sin(pi*slice*p/(double)(sliceSize-1));
								}
							}
						}
					}
          std::cout << results[p][m][n] << " ";     
				}
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    int NX = colSize-2;
    int NY = rowSize-2;
    int NZ = sliceSize-2;
    float* in = new float[NX*NY*NZ];
    float* inCopy = new float[NX*NY*NZ];
    float* out = new float[NX*NY*NZ];
		
    std::cout << "3D inputs:" << std::endl;
    for (int z = 0; z < NZ; z++)
		{
			for (int y = 0; y < NY; y++)
			{
				for (int x = 0; x < NX; x++)
				{
					// row->y, col->x.
					in[z*NX*NY+y*NX+x] = data[z+1][y+1][x+1];
					inCopy[z*NX*NY+y*NX+x] = in[z*NX*NY+y*NX+x];
					std::cout << in[z*NX*NY+y*NX+x] << " ";
				}
				std::cout << std::endl;
			}
      std::cout << std::endl;
		}
    std::cout << std::endl;
    
    fluidTransform->CalculateUnnormalised3DSineTransform(NX, NY, NZ, in, out);
    std::cout << "out3d:" << std::endl;
    for (int z = 0; z < NZ; z++)
    {
			for (int y = 0; y < NY; y++)
			{
				for (int x = 0; x < NX; x++)
				{
					std::cout << out[z*NX*NY+y*NX+x]/8.0 << " "; 
					in[z*NX*NY+y*NX+x] = out[z*NX*NY+y*NX+x]/8.0;
					if (fabs(results[z+1][y+1][x+1]-out[z*NX*NY+y*NX+x]/8.0) > tolerance)
          {
            std::cout << std::endl << out[z*NX*NY+y*NX+x]/8.0 << "," << results[z+1][y+1][x+1] << std::endl; 
						return EXIT_FAILURE;
          }
				}
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    fluidTransform->CalculateUnnormalised3DSineTransform(NX, NY, NZ, in, out);
    std::cout << "inverse out:" << std::endl;
    for (int z = 0; z < NZ; z++)
    {
			for (int y = 0; y < NY; y++)
			{
				for (int x = 0; x < NX; x++)
				{
					out[z*NX*NY+y*NX+x] = out[z*NX*NY+y*NX+x]/8.0/((NX+1)*(NY+1)*(NZ+1)/8.0);
					std::cout << out[z*NX*NY+y*NX+x] << " ";
					if (fabs(data[z+1][y+1][x+1]-out[z*NX*NY+y*NX+x]) > tolerance)
						return EXIT_FAILURE;
				}
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
	
    delete[] in; 
    delete[] out;
    delete[] inCopy;
		return EXIT_SUCCESS;
	}		
  /**
   * Fluid PDE solver tests.
   * Set velocity v = sin(pi*x/M)*sin(pi*y/N) and calculated the force b.
   * Put the force b into the solver to compare the calculate v to the expected v.
   */
  int TestFluidPDESolver(double lambda, double mu, int sizeX, int sizeY)
  {
    FluidPDEFilterType::Pointer fluidTransform = FluidPDEFilterType::New();
    FluidPDEFilterType::InputImageType::Pointer registrationForce = FluidPDEFilterType::InputImageType::New();
    typedef itk::ImageRegion< Dimension > RegionType;
    typedef itk::Size< Dimension > SizeType;
    typedef itk::Index<Dimension>  IndexType;
    SizeType size;
    IndexType start;
    RegionType region;
    float origin[]  = {0.0f, 0.0f};
    float spacing[]  = {1.0f, 1.0f};
     
    size[0] = sizeX;
    size[1] = sizeY;
    start[0] = 0;
    start[1] = 0;
    region.SetIndex(start);
    region.SetSize(size);
    registrationForce->SetLargestPossibleRegion(region);
    registrationForce->SetBufferedRegion(region);
    registrationForce->SetRequestedRegion(region);
    registrationForce->SetOrigin(origin);
    registrationForce->SetSpacing(spacing);
    registrationForce->Allocate();
    
    typedef itk::ImageRegionIteratorWithIndex< FluidPDEFilterType::InputImageType >  IteratorType;
    FluidPDEFilterType::InputImageType::PixelType force;
    VnlMatrixType velocity2D(size[1], size[0]);
    
    //std::cout << "force" << std::endl;
    for (unsigned int y = 0; y < size[1]; y++)
    {
      for (unsigned int x = 0; x < size[0]; x++)
      {
        IndexType index;
  
        index[0] = x;
        index[1] = y;
        
        force[0] = -pi*pi*((lambda+2*mu)/(((double)size[0]-1)*((double)size[0]-1))+mu/(((double)size[1]-1)*((double)size[1]-1)))
                    *sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1));
        force[1] = pi*pi/(((double)size[0]-1)*((double)size[1]-1))*(mu+lambda)*cos(pi*x/((double)size[0]-1))*cos(pi*y/((double)size[1]-1));;
        velocity2D(y,x) = sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1));
        registrationForce->SetPixel(index, force);
        //std::cout << force << " ";
      }
      //std::cout << std::endl;
    }
  
    //std::cout << "Expected velocity: " << std::endl << velocity2D << std::endl;
    
    fluidTransform->SetInput(registrationForce);
    fluidTransform->SetLambda(lambda);
    fluidTransform->SetMu(mu);
    fluidTransform->Update();
    FluidPDEFilterType::OutputImageType::Pointer velocity = fluidTransform->GetOutput();
  
    //std::cout << "Calculated velocity X:" << std::endl;  
    for (unsigned int y = 1; y <= size[1]-2; y++)
    {
      for (unsigned int x = 1; x <= size[0]-2; x++)
      {
        IndexType index;
  
        index[0] = x;
        index[1] = y;
        force = velocity->GetPixel(index);
        //std::cout << force[0] << " ";
        if (fabs(velocity2D(y,x) - force[0]) > fabs(velocity2D(y,x)*0.05))
          return EXIT_FAILURE;
      }
      //std::cout << std::endl;
    }
    //std::cout << std::endl;
    //std::cout << "Calculated velocity Y:" << std::endl;  
    for (unsigned int y = 1; y <= size[1]-2; y++)
    {
      for (unsigned int x = 1; x <= size[0]-2; x++)
      {
        IndexType index;
  
        index[0] = x;
        index[1] = y;
        force = velocity->GetPixel(index);
        //std::cout << force[1] << " "; 
        if (fabs(force[1]) > 0.05)
          return EXIT_FAILURE;
      }
      //std::cout << std::endl;
    }
    return EXIT_SUCCESS;
  }
  
  /**
   * Fluid PDE solver tests.
   */
  int TestFluidPDESolverInverse(double lambda, double mu, int sizeX, int sizeY)
  {
    FluidPDEFilterType::Pointer fluidTransform = FluidPDEFilterType::New();
    FluidPDEFilterType::InputImageType::Pointer registrationForce = FluidPDEFilterType::InputImageType::New();
    FluidPDEFilterType::InputImageType::Pointer registrationVelocity = FluidPDEFilterType::InputImageType::New();
    typedef itk::ImageRegion< Dimension > RegionType;
    typedef itk::Size< Dimension > SizeType;
    typedef itk::Index<Dimension>  IndexType;
    SizeType size;
    IndexType start;
    RegionType region;
    float origin[]  = {0.0f, 0.0f};
    float spacing[]  = {1.0f, 1.0f};
     
    size[0] = sizeX;
    size[1] = sizeY;
    start[0] = 0;
    start[1] = 0;
    region.SetIndex(start);
    region.SetSize(size);
    registrationForce->SetLargestPossibleRegion(region);
    registrationForce->SetBufferedRegion(region);
    registrationForce->SetRequestedRegion(region);
    registrationForce->SetOrigin(origin);
    registrationForce->SetSpacing(spacing);
    registrationForce->Allocate();
    registrationVelocity->SetLargestPossibleRegion(region);
    registrationVelocity->SetBufferedRegion(region);
    registrationVelocity->SetRequestedRegion(region);
    registrationVelocity->SetOrigin(origin);
    registrationVelocity->SetSpacing(spacing);
    registrationVelocity->Allocate();
    
    typedef itk::ImageRegionIteratorWithIndex< FluidPDEFilterType::InputImageType >  IteratorType;
    FluidPDEFilterType::InputImageType::PixelType force;
    FluidPDEFilterType::InputImageType::PixelType velocity;
    VnlMatrixType velocity2D(size[1], size[0]);
    
    //std::cout << "force" << std::endl;
    for (unsigned int y = 0; y < size[1]; y++)
    {
      for (unsigned int x = 0; x < size[0]; x++)
      {
        IndexType index;
  
        index[0] = x;
        index[1] = y;
        
        force[0] = -pi*pi*((lambda+2*mu)/(((double)size[0]-1)*((double)size[0]-1))+mu/(((double)size[1]-1)*((double)size[1]-1)))
                    *sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1));
        force[1] = pi*pi/(((double)size[0]-1)*((double)size[1]-1))*(mu+lambda)*cos(pi*x/((double)size[0]-1))*cos(pi*y/((double)size[1]-1));;
        velocity2D(y,x) = sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1));
        registrationForce->SetPixel(index, force);
        velocity[0] = sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1)); 
        velocity[1] = 0.; 
        registrationVelocity->SetPixel(index, velocity); 
        //std::cout << force << " ";
      }
      //std::cout << std::endl;
    }
  
    //std::cout << "Expected velocity: " << std::endl << velocity2D << std::endl;
    
    fluidTransform->SetInput(registrationForce);
    fluidTransform->SetLambda(lambda);
    fluidTransform->SetMu(mu);
    fluidTransform->SetIsComputeVelcoity(true); 
    fluidTransform->Update();
    FluidPDEFilterType::OutputImageType::Pointer tempVelocity = fluidTransform->GetOutput();
    tempVelocity->DisconnectPipeline(); 
    
    // fluidTransform->SetInput(registrationVelocity);
    fluidTransform->SetInput(tempVelocity); 
    fluidTransform->SetLambda(lambda);
    fluidTransform->SetMu(mu);
    fluidTransform->SetIsComputeVelcoity(false); 
    fluidTransform->Update();
    FluidPDEFilterType::OutputImageType::Pointer momentum = fluidTransform->GetOutput();
  
    //std::cout << "Calculated velocity X:" << std::endl;  
    for (unsigned int y = 1; y <= size[1]-2; y++)
    {
      for (unsigned int x = 1; x <= size[0]-2; x++)
      {
        IndexType index;
  
        index[0] = x;
        index[1] = y;
        FluidPDEFilterType::InputImageType::PixelType calculatedMomentum = momentum->GetPixel(index);
        FluidPDEFilterType::InputImageType::PixelType trueMomentum = registrationForce->GetPixel(index); 
        // std::cout << calculatedMomentum[0] << " " << trueMomentum[0] << " ";
        if (fabs(calculatedMomentum[0] - trueMomentum[0]) > fabs(trueMomentum[0]*0.05) && fabs(calculatedMomentum[0] - trueMomentum[0]) > 1e-6)
          return EXIT_FAILURE;
        //std::cout << calculatedMomentum[1] << " " << trueMomentum[1] << " ";
        if (fabs(calculatedMomentum[1] - trueMomentum[1]) > fabs(trueMomentum[1]*0.05) && fabs(calculatedMomentum[1] - trueMomentum[1]) > 1e-6)
          return EXIT_FAILURE;
      }
      //std::cout << std::endl;
    }
    //std::cout << std::endl;
    return EXIT_SUCCESS;
  }
  
  
  /**
   * Fluid PDE solver tests.
   * Set velocity v = sin(pi*x/M)*sin(pi*y/N)(sin(pi*z/P) and calculated the force b.
   * Put the force b into the solver to compare the calculate v to the expected v.
   */
  int TestFluidPDESolver3D(double lambda, double mu, int sizeX, int sizeY, int sizeZ)
  {
    FluidPDEFilter3DType::Pointer fluidTransform = FluidPDEFilter3DType::New();
    FluidPDEFilter3DType::InputImageType::Pointer registrationForce = FluidPDEFilter3DType::InputImageType::New();
    typedef itk::ImageRegion< Dimension3D > RegionType;
    typedef itk::Size< Dimension3D > SizeType;
    typedef itk::Index< Dimension3D >  IndexType;
    SizeType size;
    IndexType start;
    RegionType region;
    float origin[]  = {0.0f, 0.0f, 0.0f};
    float spacing[]  = {1.0f, 1.0f, 1.0f};
     
    size[0] = sizeX;
    size[1] = sizeY;
    size[2] = sizeZ;
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;
    region.SetIndex(start);
    region.SetSize(size);
    registrationForce->SetRegions(region);
    registrationForce->SetOrigin(origin);
    registrationForce->SetSpacing(spacing);
    registrationForce->Allocate();
    
    typedef itk::ImageRegionIteratorWithIndex< FluidPDEFilter3DType::InputImageType >  IteratorType;
    FluidPDEFilter3DType::InputImageType::PixelType force;

    double *velocity = new double[sizeZ*sizeY*sizeX];

    //std::cout << "Expected velocity:" << std::endl;
    for (unsigned int z = 0; z < size[2]; z++)
    {
      for (unsigned int y = 0; y < size[1]; y++)
      {
        for (unsigned int x = 0; x < size[0]; x++)
        {
          IndexType index;
    
          index[0] = x;
          index[1] = y;
          index[2] = z;
          
          force[0] = -pi*pi*((lambda+2*mu)/(((double)size[0]-1)*((double)size[0]-1))
                             +mu/(((double)size[1]-1)*((double)size[1]-1))
                             +mu/(((double)size[2]-1)*((double)size[2]-1)))
                           *sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1))*sin(pi*z/((double)size[2]-1));
          force[1] = pi*pi/(((double)size[0]-1)*((double)size[1]-1))*(mu+lambda)*cos(pi*x/((double)size[0]-1))*cos(pi*y/((double)size[1]-1))*sin(pi*z/((double)size[2]-1));
          force[2] = pi*pi/(((double)size[0]-1)*((double)size[2]-1))*(mu+lambda)*cos(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1))*cos(pi*z/((double)size[2]-1));
          registrationForce->SetPixel(index, force);
          velocity[z*size[1]*size[0] + y*size[0] + x] = sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1))*sin(pi*z/((double)size[2]-1));
        }
        //std::cout << std::endl;
      }
      //std::cout << std::endl;
    }
    //std::cout << std::endl;
  
    //std::cout << "Expected velocity: " << std::endl << velocity2D << std::endl;
    
    fluidTransform->SetInput(registrationForce);
    fluidTransform->SetLambda(lambda);
    fluidTransform->SetMu(mu);
    fluidTransform->Update();
    FluidPDEFilter3DType::OutputImageType::Pointer outputVelocity = fluidTransform->GetOutput();
  
    //std::cout << "Calculated velocity X:" << std::endl;  
    for (unsigned int z = 2; z <= size[2]-3; z++)
    {
      for (unsigned int y = 2; y <= size[1]-3; y++)
      {
        for (unsigned int x = 2; x <= size[0]-3; x++)
        {
          IndexType index;
    
          index[0] = x;
          index[1] = y;
          index[2] = z;
          force = outputVelocity->GetPixel(index);
          //std::cout << force << " ";
          //std::cout << std::endl;
          if (fabs(velocity[z*size[1]*size[0] + y*size[0] + x] - force[0]) > fabs(velocity[z*size[1]*size[0] + y*size[0] + x]*0.05))
          {
            std::cout << "Expected velocity=" << velocity[z*size[1]*size[0] + y*size[0] + x] << " at [" << x << "," << y << "," << z << "], calculated velocity=" << force << std::endl;
            return EXIT_FAILURE;
          }
          if (fabs(force[1]) > 0.05)
            return EXIT_FAILURE;
          if (fabs(force[2]) > 0.05)
            return EXIT_FAILURE;
        }
        //std::cout << std::endl;
      }
      //std::cout << std::endl;
    }
    std::cout << "Exiting: TestFluidPDESolver3D" << std::endl;  
    
    return EXIT_SUCCESS;
  }
  
  
  /**
   * Fluid PDE inverse solver tests.
   */
  int TestFluidPDESolver3DInverse(double lambda, double mu, int sizeX, int sizeY, int sizeZ)
  {
    FluidPDEFilter3DType::Pointer fluidTransform = FluidPDEFilter3DType::New();
    FluidPDEFilter3DType::InputImageType::Pointer registrationForce = FluidPDEFilter3DType::InputImageType::New();
    FluidPDEFilter3DType::InputImageType::Pointer registrationVelocity = FluidPDEFilter3DType::InputImageType::New();
    typedef itk::ImageRegion< Dimension3D > RegionType;
    typedef itk::Size< Dimension3D > SizeType;
    typedef itk::Index< Dimension3D >  IndexType;
    SizeType size;
    IndexType start;
    RegionType region;
    float origin[]  = {0.0f, 0.0f, 0.0f};
    float spacing[]  = {1.0f, 1.0f, 1.0f};
     
    size[0] = sizeX;
    size[1] = sizeY;
    size[2] = sizeZ;
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;
    region.SetIndex(start);
    region.SetSize(size);
    registrationForce->SetRegions(region);
    registrationForce->SetOrigin(origin);
    registrationForce->SetSpacing(spacing);
    registrationForce->Allocate();
    registrationVelocity->SetRegions(region);
    registrationVelocity->SetOrigin(origin);
    registrationVelocity->SetSpacing(spacing);
    registrationVelocity->Allocate();
    
    typedef itk::ImageRegionIteratorWithIndex< FluidPDEFilter3DType::InputImageType >  IteratorType;
    FluidPDEFilter3DType::InputImageType::PixelType force;
    FluidPDEFilter3DType::InputImageType::PixelType velocity;

    // double* velocity = new double[sizeZ*sizeY*sizeX];

    //std::cout << "Expected velocity:" << std::endl;
    for (unsigned int z = 0; z < size[2]; z++)
    {
      for (unsigned int y = 0; y < size[1]; y++)
      {
        for (unsigned int x = 0; x < size[0]; x++)
        {
          IndexType index;
    
          index[0] = x;
          index[1] = y;
          index[2] = z;
          
          force[0] = -pi*pi*((lambda+2*mu)/(((double)size[0]-1)*((double)size[0]-1))
                             +mu/(((double)size[1]-1)*((double)size[1]-1))
                             +mu/(((double)size[2]-1)*((double)size[2]-1)))
                           *sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1))*sin(pi*z/((double)size[2]-1));
          force[1] = pi*pi/(((double)size[0]-1)*((double)size[1]-1))*(mu+lambda)*cos(pi*x/((double)size[0]-1))*cos(pi*y/((double)size[1]-1))*sin(pi*z/((double)size[2]-1));
          force[2] = pi*pi/(((double)size[0]-1)*((double)size[2]-1))*(mu+lambda)*cos(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1))*cos(pi*z/((double)size[2]-1));
          // velocity[z*size[1]*size[0] + y*size[0] + x] = sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1))*sin(pi*z/((double)size[2]-1));
          registrationForce->SetPixel(index, force);
          velocity[0] = sin(pi*x/((double)size[0]-1))*sin(pi*y/((double)size[1]-1))*sin(pi*z/((double)size[2]-1)); 
          velocity[1] = 0.; 
          velocity[2] = 0.; 
          registrationVelocity->SetPixel(index, velocity); 
        }
        //std::cout << std::endl;
      }
      //std::cout << std::endl;
    }
    //std::cout << std::endl;
  
    //std::cout << "Expected velocity: " << std::endl << velocity2D << std::endl;
    
    fluidTransform->SetInput(registrationForce);
    fluidTransform->SetLambda(lambda);
    fluidTransform->SetMu(mu);
    fluidTransform->SetIsComputeVelcoity(true); 
    fluidTransform->Update();
    FluidPDEFilter3DType::OutputImageType::Pointer tempVelocity = fluidTransform->GetOutput();
    tempVelocity->DisconnectPipeline(); 
    
    fluidTransform->SetInput(tempVelocity); 
    fluidTransform->SetLambda(lambda);
    fluidTransform->SetMu(mu);
    fluidTransform->SetIsComputeVelcoity(false); 
    fluidTransform->Update();
    FluidPDEFilter3DType::OutputImageType::Pointer momentum = fluidTransform->GetOutput();
    
    FluidPDEFilter3DType::OutputImageType::Pointer outputVelocity = fluidTransform->GetOutput();
  
    //std::cout << "Calculated velocity X:" << std::endl;  
    for (unsigned int z = 2; z <= size[2]-3; z++)
    {
      for (unsigned int y = 2; y <= size[1]-3; y++)
      {
        for (unsigned int x = 2; x <= size[0]-3; x++)
        {
          IndexType index;
    
          index[0] = x;
          index[1] = y;
          index[2] = z;
          FluidPDEFilter3DType::InputImageType::PixelType calculatedMomentum = momentum->GetPixel(index);
          FluidPDEFilter3DType::InputImageType::PixelType trueMomentum = registrationForce->GetPixel(index); 
          //std::cout << calculatedMomentum[0] << " " << trueMomentum[0] << " ";
          if (fabs(calculatedMomentum[0] - trueMomentum[0]) > fabs(trueMomentum[0]*0.05) && fabs(calculatedMomentum[0] - trueMomentum[0]) > 1e-6)
            return EXIT_FAILURE;
          //std::cout << calculatedMomentum[1] << " " << trueMomentum[1] << " ";
          if (fabs(calculatedMomentum[1] - trueMomentum[1]) > fabs(trueMomentum[1]*0.05) && fabs(calculatedMomentum[1] - trueMomentum[1]) > 1e-6)
            return EXIT_FAILURE;
          if (fabs(calculatedMomentum[2] - trueMomentum[2]) > fabs(trueMomentum[2]*0.05) && fabs(calculatedMomentum[2] - trueMomentum[2]) > 1e-6)
            return EXIT_FAILURE;
        }
        //std::cout << std::endl;
      }
      //std::cout << std::endl;
    }
    
    return EXIT_SUCCESS;
  }
  
};


int FluidPDEFilterTest(int argc, char * argv[])
{
  srand(time(NULL));
  
  FluidPDEFilterUnitTest test;
  
  if (test.Test1DSineTransform() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (test.Test2DSineTransform() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (test.Test3DSineTransform() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (double lambda = 0; lambda <= 3.1; lambda++)
  {
    for (double mu = 1.0; mu <= 4.1; mu++)
    {
      for (int sizeX = 10; sizeX <= 20; sizeX++)
      {
        for (int sizeY = 10; sizeY <= 20; sizeY++)
        {
          std::cout << "Testing 2D solver lambda=" << lambda << ",mu=" << mu << ",size=" << sizeX << "," << sizeY << std::endl; 
          if (test.TestFluidPDESolver(lambda, mu, sizeX, sizeY) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        }
      }
    }
  }
    
  for (double lambda = 0; lambda <= 3.1; lambda++)
  {
    for (double mu = 1.0; mu <= 4.1; mu++)
    {
      for (int sizeX = 20; sizeX <= 21; sizeX++)
      {
        for (int sizeY = 20; sizeY <= 21; sizeY++)
        {
          for (int sizeZ = 30; sizeZ <= 31; sizeZ++)
          {
            std::cout << "Testing 3D solver lambda=" << lambda << ",mu=" << mu << ",size=" << sizeX << "," << sizeY << "," << sizeZ << std::endl; 
            if (test.TestFluidPDESolver3D(lambda, mu, sizeX, sizeY, sizeZ) != EXIT_SUCCESS)
              return EXIT_FAILURE; 
          }
        }
      }
    }
  }
  
  for (double lambda = 0; lambda <= 3.1; lambda++)
  {
    for (double mu = 1.0; mu <= 4.1; mu++)
    {
      for (int sizeX = 10; sizeX <= 20; sizeX++)
      {
        for (int sizeY = 10; sizeY <= 20; sizeY++)
        {
          std::cout << "Testing 2D inverse solver lambda=" << lambda << ",mu=" << mu << ",size=" << sizeX << "," << sizeY << std::endl; 
          if (test.TestFluidPDESolverInverse(lambda, mu, sizeX, sizeY) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        }
      }
    }
  }
    
  for (double lambda = 0; lambda <= 3.1; lambda++)
  {
    for (double mu = 1.0; mu <= 4.1; mu++)
    {
      for (int sizeX = 20; sizeX <= 21; sizeX++)
      {
        for (int sizeY = 20; sizeY <= 21; sizeY++)
        {
          for (int sizeZ = 30; sizeZ <= 31; sizeZ++)
          {
            std::cout << "Testing 3D solver lambda=" << lambda << ",mu=" << mu << ",size=" << sizeX << "," << sizeY << "," << sizeZ << std::endl; 
            if (test.TestFluidPDESolver3DInverse(lambda, mu, sizeX, sizeY, sizeZ) != EXIT_SUCCESS)
              return EXIT_FAILURE; 
          }
        }
      }
    }
  }

  // All objects should be automatically destroyed at this point
  std::cout << "Test PASSED !" << std::endl;

  return EXIT_SUCCESS;
}
