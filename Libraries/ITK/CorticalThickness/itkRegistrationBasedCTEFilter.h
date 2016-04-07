/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkRegistrationBasedCTEFilter_h
#define itkRegistrationBasedCTEFilter_h

#include <itkImageToImageFilter.h>
#include <itkVector.h>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkGaussianSmoothVectorFieldFilter.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkSetOutputVectorToCurrentPositionFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkVectorMagnitudeImageFilter.h>
#include <itkVectorVPlusLambdaUImageFilter.h>
#include "itkVectorPhiPlusDeltaTTimesVFilter.h"
#include "itkDasGradientFilter.h"
#include <itkDiscreteGaussianImageFilter.h>

namespace itk {

/*
 * \class RegistrationBasedCTEFilter
 * \brief This class implements Das et al Neuroimage 45 (2009) 867-879.
 * 
 * The inputs to this filter should be exactly as follows.
 * <pre>
 * input1 = white matter pv image, set using SetWhiteMatterPVMap(image)
 * input2 = white + grey pv image, set using SetWhitePlusGreyMatterPVMap(image)
 * input3 = thickness prior image, set using SetThicknessPriorMap(image)
 * input4 = the grey white border, as a binary image where 1 = boundary and 0 is background, set using SetGWI(image)
 * </pre>
 * 
 * The output is the DiReCT thickness map.
 * 
 * After registration, we can additionally output the displacement field, 
 * using the method WriteDisplacementField(filename). 
 */
template< class TInputImage, typename TScalarType>
class ITK_EXPORT RegistrationBasedCTEFilter :
  public ImageToImageFilter<TInputImage, TInputImage>
{
public:

  /** Standard ITK typedefs. */
  typedef TScalarType                                                   VectorDataType;
  typedef RegistrationBasedCTEFilter                                    Self;
  typedef ImageToImageFilter<TInputImage, TInputImage>                  Superclass;
  typedef SmartPointer<Self>                                            Pointer;
  typedef SmartPointer<const Self>                                      ConstPointer;

  /** Method for creation through the object factory.  */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro(RegistrationBasedCTEFilter, ImageToImageFilter);

  /** Get the number of dimensions we are working in. */
  itkStaticConstMacro(Dimension, unsigned int, TInputImage::ImageDimension);

  /** Any further typedefs. */          
  typedef TInputImage                                                   ImageType;
  typedef typename ImageType::PixelType                                 PixelType;
  typedef typename ImageType::Pointer                                   ImagePointer;
  typedef typename ImageType::SizeType                                  SizeType;
  typedef typename ImageType::IndexType                                 IndexType;
  typedef typename ImageType::RegionType                                RegionType;
  typedef typename ImageType::SpacingType                               SpacingType;
  typedef typename ImageType::PointType                                 OriginType;
  typedef typename ImageType::DirectionType                             DirectionType;
  typedef Vector< VectorDataType, itkGetStaticConstMacro(Dimension) >   VectorPixelType;
  typedef Image< VectorPixelType, itkGetStaticConstMacro(Dimension) >   VectorImageType;
  typedef typename VectorImageType::Pointer                             VectorImagePointer;
  typedef typename VectorImageType::SizeType                            VectorImageSizeType;
  typedef typename VectorImageType::IndexType                           VectorImageIndexType;
  typedef typename VectorImageType::RegionType                          VectorImageRegionType;
  typedef typename VectorImageType::SpacingType                         VectorImageSpacingType;
  typedef typename VectorImageType::PointType                           VectorImagePointType;
  typedef typename VectorImageType::PointType                           VectorImageOriginType;
  typedef typename VectorImageType::DirectionType                       VectorImageDirectionType;
  typedef ImageFileReader< VectorImageType  >                           VectorImageReaderType;
  typedef ImageFileWriter< VectorImageType  >                           VectorImageWriterType;
  typedef ImageFileWriter< ImageType >                                  ScalarImageWriterType;
  
  typedef LinearInterpolateImageFunction< TInputImage, TScalarType >    LinearInterpolatorType;
  typedef GaussianSmoothVectorFieldFilter<TScalarType, 
                     itkGetStaticConstMacro(Dimension),
                     itkGetStaticConstMacro(Dimension)
                     >                                                  ConvolveFilterType;
  typedef DisplacementFieldJacobianDeterminantFilter<VectorImageType, 
                                                     TScalarType,
                                                     TInputImage>       JacobianFilterType;
  typedef MinimumMaximumImageCalculator<TInputImage>                    MinMaxJacobianType;
  typedef SetOutputVectorToCurrentPositionFilter<TScalarType, 
                                     itkGetStaticConstMacro(Dimension)> InitializePhiFilterType;
  typedef SubtractImageFilter<VectorImageType, VectorImageType>         SubtractImageFilterType;
  typedef VectorMagnitudeImageFilter<VectorImageType, TInputImage>      VectorMagnitudeFilterType; 
  typedef VectorVPlusLambdaUImageFilter<TScalarType,
                                     itkGetStaticConstMacro(Dimension)> VectorVPlusLambdaUFilterType;

  typedef VectorPhiPlusDeltaTTimesVFilter<TScalarType,
                                     itkGetStaticConstMacro(Dimension)> VectorPhiPlusDeltaTTimesVFilterType;
  typedef DasGradientFilter<TScalarType, 
                            itkGetStaticConstMacro(Dimension)>          DasGradientFilterType;
  typedef DasTransformImageFilter<TScalarType, 
                                  itkGetStaticConstMacro(Dimension)>    DasTransformImageFilterType;                               
  typedef DiscreteGaussianImageFilter<ImageType,ImageType>              GaussianSmoothImageFilterType;
  
  /** Set the white matter PV image. */
  void SetWhiteMatterPVMap(ImagePointer image) { this->SetInput(0, image); }
  
  /** Set the white+grey matter PV image. */
  void SetWhitePlusGreyMatterPVMap(ImagePointer image) { this->SetInput(1, image); }
  
  /** Set the thickness prior image */
  void SetThicknessPriorMap(ImagePointer image) { this->SetInput(2, image); }

  /** Set the GWI image. */
  void SetGWI(ImagePointer image) { this->SetInput(3, image); }
  
  /** Set/Get the maximum number of iterations. Default 100. */ 
  itkSetMacro(MaxIterations, unsigned int);
  itkGetMacro(MaxIterations, unsigned int);

  /** Set/Get the number of steps in integration of the ODE. Default 10. */
  itkSetMacro(M, unsigned int);
  itkGetMacro(M, unsigned int);

  /** Set/Get the number of steps in the temporal discretization of the velocity field. Default 10. */
  itkSetMacro(N, unsigned int);
  itkGetMacro(N, unsigned int);
  
  /** Set/Get the lambda, the gradient descent parameter. Default 1.0 */
  itkSetMacro(Lambda, double);
  itkGetMacro(Lambda, double);
  
  /** Set/Get the sigma, isotropic standard deviation of the gaussian kernel used for smoothing. Default 1.5 */
  itkSetMacro(Sigma, double);
  itkGetMacro(Sigma, double);

  /** Set/Get the epsilon, the fractional tolerance between successive evaluations of equation 2 in paper. Default 0.0001. */
  itkSetMacro(Epsilon, double);
  itkGetMacro(Epsilon, double);

  /** 
   * Set/Get the Alpha, the weighting in the cost function between image similarity and velocity field energy. Default 1.0.
   * The cost function (equation 2 in paper) is weighted:
   * 
   * <pre>
   * (1-alpha)*(velocity field energy) + alpha*(image similarity)
   * </pre>
   * so an alpha of 0.9 gives 90% image similarity, and 10% velocity field energy. 
   * */
  itkSetMacro(Alpha, double);
  itkGetMacro(Alpha, double);

  /** So we can write the displacement field after registration*/
  void WriteDisplacementField(std::string filename);
  
  /** 
   * Set/Get a flag to determine if (when we write the displacement field), 
   * we output the absolute location, or a vector offset. Default is false, 
   * so the output will be a vector offset at each position, as this is
   * more easily visualized. 
   */
  itkSetMacro(OutputAbsoluteLocation, bool);
  itkGetMacro(OutputAbsoluteLocation, bool);

  /** 
   * Set/Get a flag to decide if we bother tracking jacobian. Default true. 
   * When we say tracking, it just means that the min and max Jacobian are printed in the debug output.
   */
  itkSetMacro(TrackJacobian, bool);
  itkGetMacro(TrackJacobian, bool);

  /** 
   * Set/Get a flag to write an image showing the magnitude of the displacement field. Default false.
   * So for each iteration you get one magnitude image in file tmp.mag.nii.
   * This gives a relatively simple way of checking that the registration is progressing.
   */
  itkSetMacro(WriteMagnitudeOfDisplacementImage, bool);
  itkGetMacro(WriteMagnitudeOfDisplacementImage, bool);

  /** 
   * Set/Get a flag to decide if we write out TSurf. Default false.
   * This is done after registration to file tmp.tsurf.nii, just before we propogate the thickness value through the grey matter. 
   */
  itkSetMacro(WriteTSurfImage, bool);
  itkGetMacro(WriteTSurfImage, bool);

  /** 
   * Set/Get a flag to decide if we write out the gradient image. Default false.
   * For N discretisations of the velocity field, we write to file tmp.gradient.<i>.nii. 
   * So after each iteration you have N gradient images.
   */
  itkSetMacro(WriteGradientImage, bool);
  itkGetMacro(WriteGradientImage, bool);

  /** 
   * Set/Get a flag to decide if we write out the velocity image. Default false.
   * For N discretisations of the velocity field, we write to file tmp.velocity.<i>.nii.
   * So after each iteration you have N velocity images. 
   */
  itkSetMacro(WriteVelocityImage, bool);
  itkGetMacro(WriteVelocityImage, bool);

  /**
   * Set/Get a flag to decide if we write out the transformed moving image. Default false.
   */
  itkSetMacro(WriteTransformedMovingImage, bool);
  itkGetMacro(WriteTransformedMovingImage, bool);

  /**
   * Set/Get a flag to decide if we smooth the GM and GM+WM Pv maps. Default false. 
   */
  itkSetMacro(SmoothPVMaps, bool);
  itkGetMacro(SmoothPVMaps, bool);

  /**
   * Set/Get the Gaussian standard deviation for when we are smoothing the GM and GM+WM PV maps. Default 2mm. 
   */
  itkSetMacro(SmoothPVMapSigma, double);
  itkGetMacro(SmoothPVMapSigma, double);

  /**
   * Set/Get flag to use the gradient of the moving image, as opposed to
   * the gradient of the transformed moving image. Default false.
   */
  itkSetMacro(UseGradientMovingImage, bool);
  itkGetMacro(UseGradientMovingImage, bool);

protected:
  
  RegistrationBasedCTEFilter();
  ~RegistrationBasedCTEFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /* The main filter method. Note, single threaded. */
  virtual void GenerateData();

private:
  
  /**
   * Prohibited copy and assignment. 
   */
  RegistrationBasedCTEFilter(const Self&); 
  void operator=(const Self&); 
  
  unsigned int       m_MaxIterations;
  unsigned int       m_M;
  unsigned int       m_N;
  double             m_Epsilon;
  double             m_Sigma;
  double             m_Lambda;
  double             m_Alpha;
  double             m_SmoothPVMapSigma;
  bool               m_OutputAbsoluteLocation;
  bool               m_TrackJacobian;
  bool               m_WriteMagnitudeOfDisplacementImage;
  bool               m_WriteTSurfImage;
  bool               m_WriteGradientImage;
  bool               m_WriteVelocityImage;
  bool               m_WriteTransformedMovingImage;
  bool               m_SmoothPVMaps;
  bool               m_UseGradientMovingImage;
  
  VectorImagePointer m_PhiZeroImageUninitialized;
  VectorImagePointer m_PhiZeroImage;
  VectorImagePointer m_FinalPhiImage;
  
  /** Filters, (the rest are method local variables!) */
  typename InitializePhiFilterType::Pointer m_InitializePhiZeroFilter;
  typename GaussianSmoothImageFilterType::Pointer m_SmoothWMPVMapFilter;
  typename GaussianSmoothImageFilterType::Pointer m_SmoothGMWMPVMapFilter;
  
  /** Common function to allocate a 3D scalar image, and set it to zero. */
  void InitializeScalarImage(ImageType *image,
                             RegionType& region,
                             SpacingType& spacing,
                             OriginType& origin,
                             DirectionType& direction);
  
  /** Common function to allocate a 3D vector image, and set it to zero vectors. */
  void InitializeVectorImage(VectorImageType* image, 
                             const VectorImageRegionType& region, 
                             const VectorImageSpacingType& spacing, 
                             const VectorImageOriginType& origin, 
                             const VectorImageDirectionType& direction);
  
  /** Evaluates the energy of the velocity field for one timepoint (first part of equation 2). */
  double EvaluateVelocityField(VectorImageType* velocityField, double dt);
  
  /** Evaluates the similarity given transformation phi (second part of equation 2). */
  double EvaluateRegistrationSimilarity(VectorImageType* phi, ImageType* target, ImageType* source);
  
  /** Calculates the combined cost function (equation 2 in paper). */
  double EvaluateCostFunction(double velocityFieldEnergy, double imageSimilarity);
  
  /** Calculates the min and max jacobian, returning them by reference. */
  void CalculateMinAndMaxJacobian(VectorImageType* phi, double &min, double& max);
  
  /** Calculates the max displacement. */
  double CalculateMaxDisplacement(VectorImageType* phi);

  /** Calculates the maximum magnitude of a vector image. */
  double CalculateMaxMagnitude(VectorImageType* vec);
  
  /** Simply copies a into b. */
  void CopyVectorField(VectorImageType* a, VectorImageType* b);
  
  /** Removes a temporary file. */
  void RemoveFile(std::string filename);
  
  /** Just to generate the filename to load temporary images. */
  std::string GetFileName(int i, std::string, std::string);
  
}; // end class
  
} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRegistrationBasedCTEFilter.txx"
#endif

#endif
