/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKFLUIDDEFORMABLETRANSFORM_H_
#define ITKFLUIDDEFORMABLETRANSFORM_H_

#include "itkDeformableTransform.h"
#include <itkImage.h>
#include <itkVector.h>


namespace itk
{
  
/** 
 * \class FluidDeformableTransform
 * \brief Deformable transform using a fluid representation.
 */
template <
    class TFixedImage,                   // Templated over the image type.
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions,            // Number of Dimensions i.e. 2D or 3D
    class TDeformationScalar>            // Data type in the deformation field.       
class ITK_EXPORT FluidDeformableTransform : 
public DeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar >
{
public:
  
  /** Standard class typedefs. */
  typedef FluidDeformableTransform                                      Self;
  typedef DeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar >  Superclass;
  typedef SmartPointer<Self>                                            Pointer;
  typedef SmartPointer<const Self>                                      ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( FluidDeformableTransform, DeformableTransform );
  
  /** Get the number of dimensions. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  
  /** Standard scalar type for this class. */
  typedef typename Superclass::ScalarType                      ScalarType;
  
  /** Standard parameters container. */
  typedef typename Superclass::ParametersType                  ParametersType;
  
  /** Standard Jacobian container. */
  typedef typename Superclass::JacobianType                    JacobianType;

  /** Standard coordinate point type for this class. */
  typedef typename Superclass::OutputPointType                 OutputPointType;
  typedef typename Superclass::InputPointType                  InputPointType;

  /** Typedefs for the deformation field. */
  typedef typename Superclass::DeformationFieldPixelType       DeformationFieldPixelType;
  typedef typename Superclass::DeformationFieldType            DeformationFieldType;

  /** The deformation field is defined over the fixed image. */
  typedef TFixedImage                                          FixedImageType;
  typedef typename TFixedImage::ConstPointer                   FixedImagePointer;
  
  /**
   * Use our own deformable parameter type to save memory. 
   */
  typedef typename Superclass::VectorFieldImageType            DeformableParameterType; 
  typedef typename DeformableParameterType::Pointer            DeformableParameterPointerType; 

  /** 
   * Convenience method to set up internal images.
   * Sets the internal deformation field to the same size as fixed image.
   * Sets the parameters array to the right size, and then calls SetIdentity().
   */
  virtual void Initialize(FixedImagePointer fixedImage);

  /** 
   * Set the deformation field to Identity.
   * Doesn't affect the Global transform.
   * Doesn't resize anything either.
   */
  virtual void SetIdentity();

  /** 
   * This method sets the parameters of the transform. For a fluid transformation,
   * the parameters are displacement vectors for each voxel.
   */
  virtual void SetParameters(const ParametersType & parameters)
  {
    niftkitkDebugMacro(<< "FluidDeformableTransform: set parameter does nothing");
  }
  
  /** 
   * This method sets the parameters of the transform. For a fluid transformation,
   * the parameters are displacement vectors for each voxel.
   */
  virtual void SetDeformableParameters(DeformableParameterPointerType parameters); 
  
  /**
   * Return the current deformable parameters, which is just the deformation field. Nice and easy!
   */
  virtual DeformableParameterPointerType GetDeformableParameters() { return this->m_DeformationField; }
  
  /**
   * Interpolate the deformation fluid when moving to a bigger image. 
   */
  virtual void InterpolateNextGrid(FixedImagePointer image); 
  
  /**
   * Return true if the deformable is regriddable. 
   * This then requires the implementation the Regrid function. 
   */
  virtual bool IsRegridable() const { return true; }
  
  /** 
   * Regrid and compose the new regridded deformation 
   * Note that this function destroys the content of currentPosition. 
   */
  virtual void UpdateRegriddedDeformationParameters(DeformableParameterPointerType regriddedParameters, DeformableParameterPointerType currentPosition, double factor); 
  
  /**
   * Return a copy of the deformable parameters. 
   */
  static DeformableParameterPointerType DuplicateDeformableParameters(const DeformableParameterType* deformableParameters); 
  
  /** 
   * Returns true if we are currently equal to Identity transform. 
   */
  virtual bool IsIdentity();
  
  /**
   * Invert using gradient descent. 
   */
  void InvertUsingGradientDescent(typename Self::Pointer invertedTransform, unsigned int maxIteration, double tol); 
  
  /**
   * Compute the square root of the deformation. 
   */
  void ComputeSquareRoot(typename Self::Pointer sqrtTransform, unsigned int maxInverseIteration, unsigned int maxIteration, double tol); 

protected:

  FluidDeformableTransform();
  virtual ~FluidDeformableTransform();

  /** Print contents of an FluidDeformableTransform. */
  void PrintSelf(std::ostream &os, Indent indent) const;
  
  /** To get the valid Jacobian region - because for fluid Diriac boundary condition - the deformation around the edge is 0. */
  virtual typename Superclass::JacobianDeterminantFilterType::OutputImageRegionType GetValidJacobianRegion() const 
  { 
    typename Superclass::JacobianDeterminantFilterType::OutputImageRegionType region = this->m_JacobianFilter->GetOutput()->GetLargestPossibleRegion(); 
  
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      region.SetIndex(i, 2); 
      region.SetSize(i, region.GetSize(i)-4); 
    }
    return region; 
  } 
  
  
  

private:

  FluidDeformableTransform(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};  
  
} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFluidDeformableTransform.txx"
#endif


#endif /*ITKFLUIDDEFORMABLETRANSFORM_H_*/
