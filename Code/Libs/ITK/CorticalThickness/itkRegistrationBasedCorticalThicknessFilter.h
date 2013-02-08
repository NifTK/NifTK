/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRegistrationBasedCorticalThicknessFilter_h
#define __itkRegistrationBasedCorticalThicknessFilter_h

#include "itkImageToImageFilter.h"

namespace itk {

/*
 * \class RegistrationBasedCorticalThicknessFilter
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
 * The image types should all be the same size and float. Note that this class
 * uses a stationary velocity field, so you can't set n as mentioned in the paper.
 * n is always 1.
 *
 */
template< class TInputImage, typename TScalarType>
class ITK_EXPORT RegistrationBasedCorticalThicknessFilter :
  public ImageToImageFilter<TInputImage, TInputImage>
{
public:

  /** Standard ITK typedefs. */
  typedef TScalarType                                                   VectorDataType;
  typedef RegistrationBasedCorticalThicknessFilter                      Self;
  typedef ImageToImageFilter<TInputImage, TInputImage>                  Superclass;
  typedef SmartPointer<Self>                                            Pointer;
  typedef SmartPointer<const Self>                                      ConstPointer;

  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RegistrationBasedCorticalThicknessFilter, ImageToImageFilter);

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
  typedef TScalarType                                                   MaskPixelType;
  typedef Image< MaskPixelType, itkGetStaticConstMacro(Dimension)>      MaskImageType;
  typedef typename MaskImageType::Pointer                               MaskImagePointer;
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
  typedef Image< VectorPixelType, itkGetStaticConstMacro(Dimension)+1>  TimeVaryingVectorImageType;
  typedef typename TimeVaryingVectorImageType::Pointer                  TimeVaryingVectorImagePointer;
  typedef typename TimeVaryingVectorImageType::PixelType                TimeVaryingVectorImagePixelType;
  typedef typename TimeVaryingVectorImageType::SizeType                 TimeVaryingVectorImageSizeType;
  typedef typename TimeVaryingVectorImageType::IndexType                TimeVaryingVectorImageIndexType;
  typedef typename TimeVaryingVectorImageType::SpacingType              TimeVaryingVectorImageSpacingType;
  typedef typename TimeVaryingVectorImageType::PointType                TimeVaryingVectorImagePointType;
  typedef typename TimeVaryingVectorImageType::DirectionType            TimeVaryingVectorImageDirectionType;
  typedef typename TimeVaryingVectorImageType::RegionType               TimeVaryingVectorImageRegionType;

  /** Set the white matter PV image. */
  void SetWhiteMatterPVMap(ImagePointer image) { this->SetInput(0, image); }

  /** Set the white+grey matter PV image. */
  void SetWhitePlusGreyMatterPVMap(ImagePointer image) { this->SetInput(1, image); }

  /** Set the thickness prior image */
  void SetThicknessPriorMap(ImagePointer image) { this->SetInput(2, image); }

  /** Set the GWI image. */
  void SetGWI(MaskImagePointer image) { this->SetInput(3, image); }

  /** Set the Grey Mask image. */
  void SetGreyMask(MaskImagePointer image) { this->SetInput(4, image); }
  
  /** Set/Get the maximum number of iterations. Default 100. */
  itkSetMacro(MaxIterations, unsigned int);
  itkGetMacro(MaxIterations, unsigned int);

  /** Set/Get the number of steps in integration of the ODE. Default 10. Also note that deltat t = 1/M. */
  itkSetMacro(M, unsigned int);
  itkGetMacro(M, unsigned int);

  /** Set/Get the lambda, the gradient descent parameter. Default 1.0 */
  itkSetMacro(Lambda, double);
  itkGetMacro(Lambda, double);

  /** Set/Get the isotropic standard deviation of the gaussian kernel used for smoothing the update field Default 1.5. */
  itkSetMacro(UpdateSigma, double);
  itkGetMacro(UpdateSigma, double);

  /** Set/Get the isotropic standard deviation of the gaussian kernel used for smoothing the deformation field Default 0. */
  itkSetMacro(DeformationSigma, double);
  itkGetMacro(DeformationSigma, double);

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

  /** Get the field energy of the deformation. Thats the sum of the Euclidean norm of each velocity vector. */
  itkGetMacro(FieldEnergy, double);

  /** Get the cost function = (1-alpha)(field energy) + alpha*(image similarity) */
  itkGetMacro(CostFunction, double);

  /** Get the min jacobian. */
  itkGetMacro(MinJacobian, double);

  /** Get the max jacobian. */
  itkGetMacro(MaxJacobian, double);

  /** Get the RMS change. */
  itkGetMacro(RMSChange, double);

  /** Get the SSD. */
  itkGetMacro(SSD, double);

  /** Get the cost. */
  itkGetMacro(Cost, double);

  /** Get the interface displacement image. */
  VectorImageType* GetInterfaceDisplacementImage() const { return m_InterfaceDisplacementImage; }
  
protected:

  RegistrationBasedCorticalThicknessFilter();
  ~RegistrationBasedCorticalThicknessFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /* The main filter method. Note, single threaded, as we are using a composite filter pattern, so its the contained filters that are threaded. */
  virtual void GenerateData();

private:

  /**
   * Prohibited copy and assignment.
   */
  RegistrationBasedCorticalThicknessFilter(const Self&);
  void operator=(const Self&);

  unsigned int       m_MaxIterations;
  unsigned int       m_M;
  double             m_Epsilon;
  double             m_UpdateSigma;
  double             m_DeformationSigma;
  double             m_Lambda;
  double             m_Alpha;
  double             m_FieldEnergy;
  double             m_RMSChange;
  double             m_MinJacobian;
  double             m_MaxJacobian;
  double             m_CostFunction;
  double             m_MaxThickness;
  double             m_MaxDisplacement;
  double             m_SSD;
  double             m_Cost;

  VectorImagePointer m_InterfaceDisplacementImage;
  
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRegistrationBasedCorticalThicknessFilter.txx"
#endif

#endif
