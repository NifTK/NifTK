/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-07 13:03:03 +0100 (Fri, 07 Oct 2011) $
 Revision          : $Revision: 7460 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKDEFORMABLETRANSFORM_H_
#define ITKDEFORMABLETRANSFORM_H_

#include "itkTransform.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkImageRegionIterator.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"
#include "itkForwardDifferenceDisplacementFieldJacobianDeterminantFilter.h"

namespace itk
{
  
/** 
 * \class DeformableTransform
 * \brief Base class for deformable transforms
 * 
 * For both BSplineTransform and FluidDeformableTransform, the transformation
 * is represented as an image of vectors, one vector per voxel. Its parameterisation
 * depends on derived classes. So, for Fluid, you are actually working at
 * the voxel level, and for BSpline/FFD based transformation, the transformation
 * is parameterized by the control point grid, and interpolated to each voxel.
 * 
 * The optimizer API requires SetParameters() and GetParameters().
 * So, this class provides protected utility methods GetNumberOfParametersImpliedByImage,
 * ResizeParametersArray and MarshallParametersToImage to marshall parameters
 * from the array representation to an image. In both cases, the transformation
 * owns its internal parameter array, and should be responsible for resizing it, 
 * and deciding if its the right size.
 * 
 * Don't forget that even though the transformation owns its internal parameter
 * array, such that GetParameters returns a reference to the parameters contained herein,
 * the optimizer class provides new parameters arrays throughout the optimisation.
 * This means that when the optimizer calls transform->SetParameters(&parameters),
 * then the argument is passed by reference, and the parameters are copied to
 * the parameters array within this class.
 * 
 * This class also provides a Global transform. This can be any parametric transform
 * using the standard ITK Transform class hierarchy. If this has been set then
 * calling TransformPoint will multiply the point by the Global transform
 * (for example an affine one), and then add the deformation vector onto the point.
 * 
 * TODO: We haven't tested this Global transform. I think you would actually need 
 * to interpolate the vector field. So, in most cases, you are better off sorting
 * out your affine transform, resampling, and then just doing deformable stuff on 
 * pre-registered images.       
 * 
 * \sa BSplineTransform, FluidDeformableTransform.  
 */
template <
    class TFixedImage,                   // Templated over the image type.
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions,            // Number of Dimensions i.e. 2D or 3D
    class TDeformationScalar>            // Data type in the deformation field.       
class ITK_EXPORT DeformableTransform : public Transform< TScalarType, NDimensions, NDimensions >
{
public:
  
  /** Standard class typedefs. */
  typedef DeformableTransform                                Self;
  typedef Transform< TScalarType, NDimensions, NDimensions > Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( DeformableTransform, Transform );

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  
  /** Standard scalar type for this class. */
  typedef typename Superclass::ScalarType                     ScalarType;
  
  /** Standard parameters container. */
  typedef Array<double>                                       ParametersType;
  typedef Array<double>                                       DerivativeType;
  
  /** Standard coordinate point type for this class. */
  typedef typename Superclass::OutputPointType                OutputPointType;
  typedef typename Superclass::InputPointType                 InputPointType;

  /** Standard vector type for this class. */
  typedef typename Superclass::InputVectorType                InputVectorType;
  typedef typename Superclass::OutputVectorType               OutputVectorType;
  
  /** Standard covariant vector type for this class */
  typedef typename Superclass::InputCovariantVectorType       InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType      OutputCovariantVectorType;
  
  /** Standard vnl_vector type for this class. */
  typedef typename Superclass::InputVnlVectorType             InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType            OutputVnlVectorType;

  /** These are so we can marshall parameters into an image of either control points, or voxels. */
  typedef Vector< TDeformationScalar, NDimensions >           VectorFieldPixelType;
  typedef Image< VectorFieldPixelType, NDimensions >          VectorFieldImageType;
  typedef typename VectorFieldImageType::Pointer              VectorFieldImagePointer;
  typedef typename VectorFieldImageType::SizeType             VectorFieldSizeType;
  typedef ImageRegionIterator<VectorFieldImageType>           VectorFieldIteratorType;
  typedef ImageRegionConstIterator<VectorFieldImageType>      VectorFieldConstIteratorType;
  /**
   * Both Fluid and FFD will represent the deformation 
   * field as a vector at each point in an image.
   */
  typedef VectorFieldPixelType                                DeformationFieldPixelType;
  typedef VectorFieldImageType                                DeformationFieldType;
  typedef typename DeformationFieldType::Pointer              DeformationFieldPointer;
  typedef ImageRegion<NDimensions>                            DeformationFieldRegionType;
  typedef typename DeformationFieldRegionType::IndexType      DeformationFieldIndexType;
  typedef typename DeformationFieldRegionType::SizeType       DeformationFieldSizeType;
  typedef typename DeformationFieldType::SpacingType          DeformationFieldSpacingType;
  typedef typename DeformationFieldType::DirectionType        DeformationFieldDirectionType;
  typedef typename DeformationFieldType::PointType            DeformationFieldOriginType;
  typedef Image<TDeformationScalar, NDimensions>              DeformationFieldComponentImageType; 
  
  /** The deformation field is defined over the fixed image. */
  typedef TFixedImage                                         FixedImageType;
  typedef typename FixedImageType::ConstPointer               FixedImagePointer;

  /** Typedef of the bulk transform, i.e. we set up an affine transform. */
  typedef Transform<TScalarType,
                    itkGetStaticConstMacro(SpaceDimension),
                    itkGetStaticConstMacro(SpaceDimension)>   GlobalTransformType;
  typedef typename GlobalTransformType::ConstPointer          GlobalTransformPointer;

  /** Typedef for Jacobian Calculator. */
  typedef DisplacementFieldJacobianDeterminantFilter< 
                    DeformationFieldType, 
                    TScalarType >                             JacobianDeterminantFilterType;
  typedef typename JacobianDeterminantFilterType::Pointer     JacobianDeterminantFilterPointer;
  
  /** 
   * This method specifies the bulk transform to be applied. 
   */
  itkSetConstObjectMacro( GlobalTransform, GlobalTransformType );
  itkGetConstObjectMacro( GlobalTransform, GlobalTransformType );
  
  /**
   * Set/Get the search radius (voxel unit) for computing the inverse. 
   */
  itkSetMacro(InverseSearchRadius, double); 
  itkGetMacro(InverseSearchRadius, double); 

  /**
   * Return the transformation parameters.
   */
  virtual const ParametersType& GetParameters(void) const { return this->m_Parameters; }

  /**
   * Set the transformation parameters. 
   */
  virtual void SetParameters(const ParametersType& parameters) { this->m_Parameters = parameters; }

  /** 
   * Return the number of parameters that completely define the Transfom
   */
  virtual unsigned int GetNumberOfParameters(void) const { return this->m_Parameters.Size(); }

  /**
   * Sets the transformation parameters from an image.
   * The parameter 'force' is to force parameter array to the right size.
   * If force is false, and the parameters array is the wrong size, an exception is thrown.
   * If force is true, the parameters array is resized.
   */
  virtual void SetParametersFromField(const VectorFieldImagePointer& image, bool force=false);
  
  /**
   * Get the deformation field pointer. 
   */
  virtual DeformationFieldType* GetDeformationField() const { return this->m_DeformationField.GetPointer(); }

  /**
   * Actually run through the deformation field, and calculate the max deformation.
   * This is the maximum magnitude of a deformation vector (not of the individual component in x,y,z).
   */
  TScalarType ComputeMaxDeformation();
  
  /**
   * Actually run through the deformation field, and calculate the min deformation.
   * This is the minimum magnitude of a deformation vector (not of the individual component in x,y,z).
   */
  TScalarType ComputeMinDeformation();
  
  /** Actually run through the deformation field, and calculate the max jacobian. */
  TScalarType ComputeMaxJacobian();
  
  /** Actually run through the deformation field, and calculate the min jacobian. */
  TScalarType ComputeMinJacobian();
  
  /** Calculate the log of the jacobian determinant at each voxel. */
  TScalarType GetSumLogJacobianDeterminant();
  
  /** Write transform, so subclass can override it if necessary. Default implementation is suitable for Fluid. */
  virtual void WriteTransform(std::string filename) { this->WriteVectorImage(filename); }
  
  /** Write parameters, so subclass can override if necessary. Default implementation is suitable for Fluid. */
  virtual void WriteParameters(std::string filename) { this->WriteVectorImage(filename); }
  
  /** Write the jacobian image of the deformation field out to the given filename. */
  void WriteJacobianImage(std::string filename);
  
  /** Write the Midas jacobian image of the deformation field out to the given filename. */
  void WriteMidasStrImage(std::string filename,  int origin[NDimensions], typename TFixedImage::RegionType paddedDesiredRegion, const typename JacobianDeterminantFilterType::OutputImageType* jacobianImage);
  
  /** Write the Midas jacobian image of the deformation field out to the given filename. */
  void ReadMidasStrImage(std::string filename,  int origin[NDimensions], typename TFixedImage::RegionType paddedDesiredRegion, typename JacobianDeterminantFilterType::OutputImageType* jacobianImage);
  
  /** Write the Midas vector image of the deformation field out to the given filename. */
  void WriteMidasVecImage(std::string filename,  int origin[NDimensions], typename TFixedImage::RegionType paddedDesiredRegion);

  /** Write the vector image of the deformation field out to the given filename. */
  void WriteVectorImage(std::string filename);

  /** 
   * Set the parameters array to Identity.
   * Doesn't affect the Global transform.
   * Doesn't resize anything either.
   */
  virtual void SetIdentity();

  /** 
   * Convenience method to set up internal deformation field to the same size
   * as the supplied image. In this method, we can't set up a corresponding
   * m_Parameters array, as different subclasses will have different requirements.
   */
  void Initialize(FixedImagePointer image);
  
  /** Declared virtual in base class, transform points*/
  virtual OutputPointType  TransformPoint(const InputPointType  &point ) const;

  /** Method to transform a vector - not applicable for this type of transform. */
  virtual OutputVectorType TransformVector(const InputVectorType &) const
    { 
      itkExceptionMacro( << "Method not applicable for deformable transform." );
      return OutputVectorType(); 
    }

  /** Method to transform a vnl_vector - not applicable for this type of transform */
  virtual OutputVnlVectorType TransformVector(const InputVnlVectorType &) const
    { 
      itkExceptionMacro(<< "Method not applicable for deformable transform. ");
      return OutputVnlVectorType(); 
    }

  /** Method to transform a CovariantVector - not applicable for this type of transform */
  virtual OutputCovariantVectorType TransformCovariantVector(const InputCovariantVectorType &) const
    { 
      itkExceptionMacro(<< "Method not applicable for deformable transfrom. ");
      return OutputCovariantVectorType(); 
    } 

  /** 
   * Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  virtual bool IsLinear() const { return false; }

  /** Returns true if we are currently equal to Identity transform. */
  virtual bool IsIdentity();
  
  /** Get the fixed parameters for saving. */
  virtual const ParametersType& GetFixedParameters(void) const;
  
  /** Set the fixed paramters for loading */
  virtual void SetFixedParameters(const ParametersType& parameters);
  
  /** 
   * Create inverse of a deformable transformation according to 
   * Methods for inverting dense displacement fields: Evaluation in brain image registration,
   * Crum et al, MICCAI, 2007. 
   */ 
  bool GetInverse(Self* inverse) const; 
  /** 
   * Concatenate the current deformation field after the given deformation field 
   */
  void ConcatenateAfterGivenTransform(Self* givenTransform); 
  /**
   * Return true if the deformable is regriddable. 
   * This then requires the implementation the Regrid function. 
   */
  virtual bool IsRegridable() const { return false; }
  /** 
   * Regrid and compose the new regridded deformation 
   */
  virtual void UpdateRegriddedDeformationParameters(ParametersType& regriddedParameters, const ParametersType& currentPosition)
  { 
    itkExceptionMacro( << "Method Regrid not implemented for deformable transform." );
  }
  /**
   * Call to set the Jacobian calculation to be using the forward difference. 
   * This function should be called immediately after constructing the deformable transform. 
   */
  void SetUseForwardDifferenceJacobianCalculation()
  {
    typedef ForwardDifferenceDisplacementFieldJacobianDeterminantFilter<DeformationFieldType, TScalarType, typename JacobianDeterminantFilterType::OutputImageType> FilterType; 
    this->m_JacobianFilter = FilterType::New(); 
    this->m_JacobianFilter->SetInput(this->m_DeformationField);
    this->m_JacobianFilter->SetUseImageSpacingOff();
  }
  /**
   * Iniitilise the deformation field with the global transformation.
   */
  void InitialiseGlobalTransformation();
  
  /**
   * Get the Jacobian image. 
   */
  typename JacobianDeterminantFilterType::OutputImageType* GetJacobianImage() const { return m_JacobianFilter->GetOutput(); }
  
  /**
   * Extract the x, y, z components of the vector deformation field into 3 images. 
   */
  void ExtractComponents(); 
  
  /**
   * Invert using fixed point iteration. 
   * 
   * Recommended values: 
   * maxIterations=30
   * maxOuterIterations=5
   * tol=0.001
   */
  void InvertUsingIterativeFixedPoint(typename Self::Pointer invertedTransform, int maxIterations, int maxOuterIterations, double tol); 
  
protected:
  
  DeformableTransform();
  virtual ~DeformableTransform();

  /** Print contents of an FluidDeformableTransform. */
  void PrintSelf(std::ostream &os, Indent indent) const;

  /** Works out the number of parameters implied by the image. */
  unsigned long int GetNumberOfParametersImpliedByImage(const VectorFieldImagePointer image);

  /** Makes the parameters array match the size of the image, and resets everything. */
  void ResizeParametersArray(const VectorFieldImagePointer image);
  
  /** To Convert parameters array to an image. */
  void MarshallParametersToImage(VectorFieldImagePointer image);
  
  /** To Convert image to parameters array. */
  void MarshallImageToParameters(const VectorFieldImagePointer image, ParametersType& parameters);
  
  /** To get the valid Jacobian region - because for fluid Diriac boundary condition - the deformation around the edge is 0. */
  virtual typename JacobianDeterminantFilterType::OutputImageRegionType GetValidJacobianRegion() const { return this->m_JacobianFilter->GetOutput()->GetLargestPossibleRegion(); } 
  
  /** The deformation/displacement field, represented as an image. */
  typename DeformationFieldType::Pointer m_DeformationField;

  /** The global transform. */
  GlobalTransformPointer  m_GlobalTransform;

  /** Jacobian Filter. Currently no need to expose this publically, so no Setter/Getter. */
  JacobianDeterminantFilterPointer m_JacobianFilter;
  
  /** Search radius for calculating the inverse transformation */
  double m_InverseSearchRadius; 
  
  /** Tolerance for checking if the forward transformation hits the source image grid. */
  double m_InverseVoxelTolerance; 
  
  /** Tolerance for checking in the iteration if the forward transformation is close to the source image grid. */
  double m_InverseIterationTolerance; 
  
  /** Maximum number of iteration in the inverse computaion. */
  double m_MaxNumberOfInverseIterations; 
  
  /**
   * Hold the x, y, z components of the deformation field. 
   */
  typename DeformationFieldComponentImageType::Pointer m_DeformationFieldComponent[NDimensions]; 
  
private:
  
  DeformableTransform(const Self&); // purposefully not implemented
  void operator=(const Self&);      // purposefully not implemented
  
  /** Purely to avoid keeping creating news ones when we multiply points. */
  mutable DeformationFieldIndexType m_TemporaryIndex;

  /** To make sure Jacobian filter is up to date. */
  void ForceJacobianUpdate();
  
};  

} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDeformableTransform.txx"
#endif

#endif /*ITKDEFORMABLETRANSFORM_H_*/
