/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-26 10:49:56 +0100 (Thu, 26 May 2011) $
 Revision          : $Revision: 6271 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKVelocityFieldDeformableTransform_H_
#define ITKVelocityFieldDeformableTransform_H_

#include "itkFluidDeformableTransform.h"
#include "itkImage.h"
#include "itkVector.h"
#include <boost/filesystem.hpp>
#include "itkVelocityFieldDeformableTransformFilename.h"

namespace itk
{
  
/** 
 * \class VelocityFieldDeformableTransform
 * \brief Deformable transform using a fluid representation.
 */
template <
    class TFixedImage,                   // Templated over the image type.
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions,            // Number of Dimensions i.e. 2D or 3D
    class TDeformationScalar>            // Data type in the deformation field.       
class ITK_EXPORT VelocityFieldDeformableTransform : 
public FluidDeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar >
{
public:
  
  /** Standard class typedefs. */
  typedef VelocityFieldDeformableTransform Self;
  typedef FluidDeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar >  Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( VelocityFieldDeformableTransform, FluidDeformableTransform );
  
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
   * Momentum (i.e. the "body force") that shoots in the geodesics. 
   */
  typedef Image<Vector<TDeformationScalar, NDimensions>,  NDimensions>  MomentumImageType; 
 
  /**
   * PDE solver. 
   */
  typedef FluidPDEFilter<TDeformationScalar, NDimensions>        FluidPDEType;
  
  /**
   * Set/Get. 
   */
  itkSetMacro(NumberOfVelocityField, int); 
  itkGetMacro(NumberOfVelocityField, int); 

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
   * Interpolate the deformation fluid when moving to a bigger image. 
   */
  virtual void InterpolateNextGrid(FixedImagePointer image); 
  
  /**
   * Accumulate the deformation from the velocity fields. 
   */
  void AccumulateDeformationFromVelocityField(int timePoint); 
  
  /**
   * Set the velocity field at a given time point. 
   */
  void SetVelocityField(typename DeformableParameterType::Pointer field, int timePoint)
  {
    this->m_VelocityField[timePoint] = field; 
  }
  
  /**
   * Set the velocity field at a given time point. 
   */
  typename DeformableParameterType::Pointer GetVelocityField(int timePoint)
  {
    return this->m_VelocityField[timePoint]; 
  }
  
  /**
   * Set to use the fixe image deformation. 
   */
  void UseFixedImageDeformationField()
  {
    if (this->m_FixedImageDeformationField.IsNotNull())
      this->m_DeformationField = this->m_FixedImageDeformationField; 
  }
  /**
   * Set to use the fixe image deformation. 
   */
  void UseMovingImageDeformationField()
  {
    if (this->m_MovingImageDeformationField.IsNotNull())
      this->m_DeformationField = this->m_MovingImageDeformationField; 
  }
  
  /**
   * Re-parameterise the time step to have constant velocity field. 
   */
  void ReparameteriseTime(); 
  
  /**
   * Get. 
   */
  typename Superclass::JacobianDeterminantFilterType::OutputImageType::Pointer GetForwardJacobianImage()
  {
    return this->m_ForwardJacobianImage; 
  }
  
  /**
   * Get.
   */
  typename Superclass::JacobianDeterminantFilterType::OutputImageType::Pointer GetBackwardJacobianImage()
  {
    return this->m_BackwardJacobianImage; 
  }
  
  /**
   * Shoot from the intiail velocity 
   */
  void Shoot(); 
  
  /**
   * Set. 
   */
  void SetFluidPDESolver(typename FluidPDEType::Pointer solver) 
  {
    this->m_FluidPDESolver = solver; 
  }
  
  /**
   * Accumulate backward deformation field. 
   */
  void AccumulateBackwardDeformationFromVelocityField(int timePoint); 
  
  /**
   * Accumulate forward deformation field. 
   */
  void AccumulateForwardDeformationFromVelocityField(int timePoint); 
  
  /**
   * Reset all the deformation fields to 0. 
   */
  void ResetDeformationFields(); 
  
  /**
   * Load the fixed image deformation field. 
   */
  void LoadFixedImageDeformationField(int timePoint)
  {
    this->LoadField(&this->m_FixedImageDeformationField, VelocityFieldDeformableTransformFilename::GetFixedImageDeformationFilename(timePoint)); 
    this->UseFixedImageDeformationField(); 
  }
  
  /**
   * Load the moving image deformation field. 
   */
  void LoadMovingImageDeformationField(int timePoint)
  {
    this->LoadField(&this->m_MovingImageDeformationField, VelocityFieldDeformableTransformFilename::GetMovingImageDeformationFieldFilename(timePoint)); 
    this->UseMovingImageDeformationField(); 
  }
  
  /**
   * Load the velocity field. 
   */
  void LoadVelocityField(int timePoint)
  {
    this->LoadField(&(this->m_VelocityField[0]), VelocityFieldDeformableTransformFilename::GetVelocityFieldFilename(timePoint)); 
  }  
  
  /**
   * Load the velocity field. 
   */
  void SaveVelocityField(typename DeformationFieldType::Pointer field, int timePoint)
  {
    this->SaveField(field, VelocityFieldDeformableTransformFilename::GetVelocityFieldFilename(timePoint)); 
  }  
  
  /**
   * Save the best velocity field. 
   */
  void SaveBestVelocityField()
  {
    for (int i = 0; i < this->m_NumberOfVelocityField; i++)
    {
      const char* oldFilename = VelocityFieldDeformableTransformFilename::GetVelocityFieldFilename(i); 
      const char* newFilename = VelocityFieldDeformableTransformFilename::GetBestVelocityFieldFilename(i); 
      
      typename boost::filesystem::file_status status = boost::filesystem::status(newFilename); 
      if (boost::filesystem::exists(status))
      {
        boost::filesystem::remove(newFilename); 
      }
      boost::filesystem::copy_file(oldFilename, newFilename); 
    }
  }
  
  /**
   * Load the best velocity field. 
   */
  void LoadBestVelocityField()
  {
    for (int i = 0; i < this->m_NumberOfVelocityField; i++)
    {
      const char* oldFilename = VelocityFieldDeformableTransformFilename::GetBestVelocityFieldFilename(i);  
      const char* newFilename = VelocityFieldDeformableTransformFilename::GetVelocityFieldFilename(i);  
      
      typename boost::filesystem::file_status status = boost::filesystem::status(newFilename); 
      if (boost::filesystem::exists(status))
      {
        boost::filesystem::remove(newFilename); 
      }
      boost::filesystem::copy_file(oldFilename, newFilename); 
    }
  }
    
  /**
   * Save the deformation/velocity field.
   */
  static void SaveField(typename DeformationFieldType::Pointer field, std::string filename); 
  
  /**
   * Load the deformation/velocity field. 
   */
  static void LoadField(typename DeformationFieldType::Pointer* field, std::string filename); 
  
  /**
   * Initialise the velocity fields and deformation fields to 0. 
   */
  void InitializeIdentityVelocityFields(); 
  
protected:

  VelocityFieldDeformableTransform();
  virtual ~VelocityFieldDeformableTransform();

  /** Print contents of an VelocityFieldDeformableTransform. */
  void PrintSelf(std::ostream &os, Indent indent) const;
  
  /**
   * Number of velocity field. 
   */
  int m_NumberOfVelocityField; 
  
  /**
   * Array of veclocity fields. 
   */
  typename std::vector<typename DeformableParameterType::Pointer> m_VelocityField; 
  
  /**
   * Moving image deformation field. 
   */
  typename DeformableParameterType::Pointer m_FixedImageDeformationField; 
  
  /**
   * Moving image deformation field. 
   */
  typename DeformableParameterType::Pointer m_MovingImageDeformationField; 
  
  /**
   * Time step. 
   */
  typename std::vector<double> m_TimeStep; 
  
  /**
   * Jacobian of the forward transformation. 
   */
  typename Superclass::JacobianDeterminantFilterType::OutputImageType::Pointer m_ForwardJacobianImage; 
      
  /**
   * Jacobian of the forward transformation. 
   */
  typename Superclass::JacobianDeterminantFilterType::OutputImageType::Pointer m_BackwardJacobianImage; 
  
  /**
   * PDE solver. 
   */
  typename FluidPDEType::Pointer m_FluidPDESolver; 
  
private:

  VelocityFieldDeformableTransform(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};  
  
} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVelocityFieldDeformableTransform.txx"
#endif


#endif /*ITKVelocityFieldDeformableTransform_H_*/
