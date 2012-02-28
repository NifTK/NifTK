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
#ifndef __itkFluidPDEFilter_h
#define __itkFluidPDEFilter_h

#include "itkVector.h"
#include "itkImage.h"
#include "itkImageToImageFilter.h"

#include "itkNondirectionalDerivativeOperator.h"
#include "fftw3.h"

/**
 * Friend class for unit testing. 
 */  
class FluidPDEFilterUnitTest;

namespace itk {
/** 
 * \class FluidPDEFilter
 * \brief This class takes a vector image representing "force per voxel"
 * and outputs the velocity field by solving the Fluid PDE.
 */
template <
    class TScalarType = double,          // Data type for scalars
    unsigned int NDimensions = 3>        // Number of Dimensions i.e. 2D or 3D
class ITK_EXPORT FluidPDEFilter :
  public ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>, // Input image
                             Image< Vector<TScalarType, NDimensions>,  NDimensions>  // Output image
                           >
{
public:

  /** Standard "Self" typedef. */
  typedef FluidPDEFilter                                                                Self;
  typedef ImageToImageFilter< Image< Vector<TScalarType, NDimensions>,  NDimensions>,
                              Image< Vector<TScalarType, NDimensions>,  NDimensions>
                            >                                                           Superclass;
  typedef SmartPointer<Self>                                                            Pointer;
  typedef SmartPointer<const Self>                                                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FluidPDEFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, NDimensions);

  /** Standard typedefs. */
  typedef Vector<TScalarType, NDimensions>                                            OutputPixelType;
  typedef Image< OutputPixelType, NDimensions >                                       OutputImageType;
  typedef typename Superclass::InputImageType                                         InputImageType;
  typedef typename Superclass::InputImagePointer                                      InputImagePointer;
  typedef typename Superclass::InputImageRegionType                                   InputImageRegionType;

  /** Set/Get the Lambda. */
  itkSetMacro(Lambda, double);
  itkGetMacro(Lambda, double);

  /** Set/Get the Mu. */
  itkSetMacro(Mu, double);
  itkGetMacro(Mu, double);
  
  /**
   * Set/Get. 
   */
  itkSetMacro(IsComputeVelcoity, bool); 
  itkGetMacro(IsComputeVelcoity, bool); 
  itkSetMacro(IsSymmetric, bool); 
  itkGetMacro(IsSymmetric, bool); 
  
protected:
  FluidPDEFilter();
  ~FluidPDEFilter(); 
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  // The main filter method. Note, single threaded.
  virtual void GenerateData();
  
  /** Force the filter to request LargestPossibleRegion on input. */
  virtual void GenerateInputRequestedRegion();

  /** Force filter to create the output buffer at LargestPossibleRegion */
  virtual void EnlargeOutputRequestedRegion(DataObject *itkNotUsed);

  /**
   * Calculate the instanteous velocity at each voxel given the registration force
   * and puts the output in the output image.
   */
   void CalculationVelocity2D(double lamda, double mu); 
   
  /**
   * Calculate the instanteous velocity at each voxel given the registration force
   * and puts the output in the output image - 3D version.
   */
   void CalculationVelocity3D(double lamda, double mu, bool isDoingBackward); 
  
  /**
   * Normalised sine transform using 2 1D sine transform. 
   */
  void CalculateNormalised2DSineTransformUsing1DSineTransform(int sizeX, int sizeY, float* input, float* output);

  /**
   * Unnormalised sine transform - need to divide by 4 to get the normalised results.  
   */
  void CalculateUnnormalised2DSineTransform(int sizeX, int sizeY, float* input, float* output);
	
  void CalculateUnnormalised3DSineTransform(int sizeX, int sizeY, int sizeZ, float* input, float* output);
  
#ifdef CUDA_FFT
  void CalculateUnnormalised3DSineTransformCUDA(int sizeX, int sizeY, int sizeZ, float* input, float* output);
#endif  
  
  /**
   * Compute the adjoint Navier Lame operator. 
   */
  void ComputeAdjointNavierLameOperator(double lamda, double mu);
  void ComputeAdjointNavierLameOperator11(double lamda, double mu);
  void ComputeAdjointNavierLameOperator12(double lamda, double mu);
  void ComputeAdjointNavierLameOperator13(double lamda, double mu);
  void ComputeAdjointNavierLameOperator21(double lamda, double mu);
  void ComputeAdjointNavierLameOperator22(double lamda, double mu);
  void ComputeAdjointNavierLameOperator23(double lamda, double mu);
  void ComputeAdjointNavierLameOperator31(double lamda, double mu);
  void ComputeAdjointNavierLameOperator32(double lamda, double mu);
  void ComputeAdjointNavierLameOperator33(double lamda, double mu);
  
  /**
   * Compute Navier Lame operator. 
   */
  void ComputeNavierLameOperator(double lamda, double mu);
  void ComputeNavierLameOperator11(double lamda, double mu);
  void ComputeNavierLameOperator12(double lamda, double mu);
  void ComputeNavierLameOperator13(double lamda, double mu);
  void ComputeNavierLameOperator21(double lamda, double mu);
  void ComputeNavierLameOperator22(double lamda, double mu);
  void ComputeNavierLameOperator23(double lamda, double mu);
  void ComputeNavierLameOperator31(double lamda, double mu);
  void ComputeNavierLameOperator32(double lamda, double mu);
  void ComputeNavierLameOperator33(double lamda, double mu);
  
  /**
   * Convert the velocity to momentum. 
   */
  void CalculationMomentum(double lambda, double mu); 

  /**
   * Lambda in the Navier-Lame fluid PDE. Normally 0.0. 
   */
  double m_Lambda; 
  /**
   * Mu in the Navier-Lame fluid PDE. Normally 0.01. 
   */
  double m_Mu;
  /**
   * Private typedef of the finite difference operator. 
   */  
  typedef NondirectionalDerivativeOperator < double, NDimensions > NondirectionalDerivativeOperatorType; 
  /**
   * Flag to indicate if the adjoint Navier Lame operator has been initialised. 
   */
  bool m_AdjointNavierLameOperatorInitialised; 
  /**
   * The 3D adjoint Navier-Lame matrix operator.
   */
  NondirectionalDerivativeOperatorType m_AdjointNavierLameOperator[3][3]; 
  /**
   * FFTW plan. 
   */
  fftwf_plan m_fftwPlan;
  /**
   * FFTW plan initialisation flag. 
   */
  bool m_IsFFTWInitialised; 
  /**
   * Current size of fftw plan. 
   */
  int m_fftwPlanSliceSize; 
  /**
   * Current size of fftw plan. 
   */
  int m_fftwPlanColSize; 
  /**
   * Current size of fftw plan. 
   */
  int m_fftwPlanRowSize; 
  
  /**
   * NavierLameOperators intiailised? 
   */
  bool m_NavierLameOperatorInitialised; 
  
  /**
   * Computing velcoity or momentum? 
   */
  bool m_IsComputeVelcoity; 
  
  /**
   * The 3D adjoint Navier-Lame matrix operator.
   */
  NondirectionalDerivativeOperatorType m_NavierLameOperator[3][3]; 
  
  /**
   * Solve it symmetrically?
   */
  bool m_IsSymmetric; 
  
private:
  
  /**
   * Prohibited copy and assignment. 
   */
  FluidPDEFilter(const Self&); 
  void operator=(const Self&); 

  /**
   * Friend class for unit testing. 
   */  
  friend class ::FluidPDEFilterUnitTest;
  
  
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFluidPDEFilter.txx"
#endif

#endif
