#ifndef __itkNiftySimGravityTransformation_h
#define __itkNiftySimGravityTransformation_h

#include "itkNiftySimTransformation.h"


namespace itk
{

/**
 * \class NiftySimGravityTransformation
 * \brief Class to apply gravitational force to a NiftySim  model and apply this transformation to an image.
  (The rigid parameters are set in NiftySimTransformation)

  1: Rotation in x
  2: Rotation in y
  3: Rotation in z
  4: Translation in x
  5: Translation in y
  6: Translation in z

  7: Gravitation magnitude
 *
 * \ingroup Transforms
 *
 */


template <
    class TFixedImage,                   // Templated over the image type.
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions,            // Number of Dimensions i.e. 2D or 3D
    class TDeformationScalar>            // Data type in the deformation field.       
class ITK_EXPORT NiftySimGravityTransformation : 
public NiftySimTransformation< TFixedImage, TScalarType, NDimensions, TDeformationScalar >
{
public:
  
  /** Standard class typedefs. */
  typedef NiftySimGravityTransformation                                                        Self;
  typedef NiftySimTransformation< TFixedImage, TScalarType, NDimensions, TDeformationScalar >  Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( NiftySimGravityTransformation, NiftySimTransformation );
  
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


  /** TODO: Changes to be applied here! Initial checks necessary?
   */

  virtual void Initialize(FixedImagePointer image);

  /// Set the plate displacement parameter
  //void SetPlateDisplacementParameter( double displacement );

  /// This method sets the parameters of the transform.
  virtual void SetParameters(const ParametersType &parameters);


protected:

  NiftySimGravityTransformation();
  virtual ~NiftySimGravityTransformation();

  /// The unloaded distance between the plates
  //float m_MaxPlateSeparation;

  /// The direction of contact plate 1's displacement
  //std::vector<float> m_PlateOneDisplacementDirn;
  /// The direction of contact plate 2's displacement
  //std::vector<float> m_PlateTwoDisplacementDirn;

  /** Print contents of an NiftySimGravityTransformation. */
  void PrintSelf(std::ostream &os, Indent indent) const;
  
  /// Check that a std::vector is 3D
  //void CheckVectorIs3D( std::vector<float> v );
  
  /// The magnitude of a vector
  //float Magnitude( std::vector<float> v );

  /// Normalise a vector
  //std::vector<float> Normalise( std::vector<float> v );
  
  /// Return the normal to the plane defined by points a, b and c
  //std::vector<float> CalculateNormalToPlane( std::vector<float> a, 
  //					     std::vector<float> b, 
  //					     std::vector<float> c );

  /// Calculate the angle between two planes or normals n1 and n2
  //float CalculateAngleBetweenNormals( std::vector<float> n1, 
  //				      std::vector<float> n2 );

  /** Calculate the determinant of
      | a b c |
      | d e f |
      | g h i | */
  //float Determinant( float a, float b, float c,
  //		     float d, float e, float f,
  //		     float g, float h, float i );

  /** Calculate the distance between a plane through points p1, p2 and p3
      and the point q */
  //float CalculateDistanceFromPointToLine( std::vector<float> p1, 
  //					  std::vector<float> p2, 
  // 					  std::vector<float> p3,
  //					  std::vector<float> q );

private:

  NiftySimGravityTransformation(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};  
  
} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNiftySimGravityTransformation.txx"
#endif


#endif /*  __itkNiftySimGravityTransformation_h */
