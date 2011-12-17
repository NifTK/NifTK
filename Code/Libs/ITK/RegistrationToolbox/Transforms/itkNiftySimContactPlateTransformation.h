/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-02-11 15:16:22 +0000 (Fri, 11 Feb 2011) $
 Revision          : $Revision: 4977 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkNiftySimContactPlateTransformation_h
#define __itkNiftySimContactPlateTransformation_h

#include "itkNiftySimTransformation.h"

#include <vector>

namespace itk
{

/**
 * \class NiftySimContactPlateTransformation
 * \brief Class to apply a contact plate NiftySim transformation to an image.
  (The rigid parameters are set in NiftySimTransformation)

  1: Rotation in x
  2: Rotation in y
  3: Rotation in z
  4: Translation in x
  5: Translation in y
  6: Translation in z

  7: Contact plate displacement
  8: Anisotropy
  9: Poisson's ratio
 *
 * \ingroup Transforms
 *
 */


template <
    class TFixedImage,                   // Templated over the image type.
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions,            // Number of Dimensions i.e. 2D or 3D
    class TDeformationScalar>            // Data type in the deformation field.       
class ITK_EXPORT NiftySimContactPlateTransformation : 
public NiftySimTransformation< TFixedImage, TScalarType, NDimensions, TDeformationScalar >
{
public:
  
  /** Standard class typedefs. */
  typedef NiftySimContactPlateTransformation                                                            Self;
  typedef NiftySimTransformation< TFixedImage, TScalarType, NDimensions, TDeformationScalar >  Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( NiftySimContactPlateTransformation, NiftySimTransformation );
  
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
   * Checks that the compression planes are parrallel which is
   * required by this class and sets the initial plate separation
   * parameter.
   */
  virtual void Initialize(FixedImagePointer image);

  /// Set the plate displacement parameter
  void SetPlateDisplacementAndMaterialParameters( double displacement, float anisotropy, float poissonRatio );

  /// This method sets the parameters of the transform.
  virtual void SetParameters(const ParametersType &parameters);

  void GetDispBoundaries( float dispBoundaries[2] )
  { 
    dispBoundaries[0] = m_DispBoundaries[0]; 
    dispBoundaries[1] = m_DispBoundaries[1];  
  }


protected:

  NiftySimContactPlateTransformation();
  virtual ~NiftySimContactPlateTransformation();
 
  // The indices of the transofrmation parameters
  enum p_params { p_rol=0, p_inPlane, p_tx, p_ty, p_disp, p_aniso, p_poi};

  /// The unloaded distance between the plates
  float m_MaxPlateSeparation;

  /// The maximum distance between the plates, in directions X and Z
  float m_MaxPlateSeparationXZ;

  /// The direction of contact plate 1's displacement
  std::vector<float> m_PlateOneDisplacementDirn;
  /// The direction of contact plate 2's displacement
  std::vector<float> m_PlateTwoDisplacementDirn;

  /** Print contents of an NiftySimContactPlateTransformation. */
  void PrintSelf(std::ostream &os, Indent indent) const;
  
  /// Check that a std::vector is 3D
  void CheckVectorIs3D( std::vector<float> v );
  
  /// The magnitude of a vector
  float Magnitude( std::vector<float> v );

  /// Normalise a vector
  std::vector<float> Normalise( std::vector<float> v );
  
  /// Return the normal to the plane defined by points a, b and c
  std::vector<float> CalculateNormalToPlane( std::vector<float> a, 
					     std::vector<float> b, 
					     std::vector<float> c );

  /// Calculate the angle between two planes or normals n1 and n2
  float CalculateAngleBetweenNormals( std::vector<float> n1, 
				      std::vector<float> n2 );

  /** Calculate the determinant of
      | a b c |
      | d e f |
      | g h i | */
  float Determinant( float a, float b, float c,
		     float d, float e, float f,
		     float g, float h, float i );

  /** Calculate the distance between a plane through points p1, p2 and p3
      and the point q */
  float CalculateDistanceFromPointToLine( std::vector<float> p1, 
					  std::vector<float> p2, 
					  std::vector<float> p3,
					  std::vector<float> q );
					  
  float m_DispBoundaries[2];

private:

  NiftySimContactPlateTransformation(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};  
  
} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNiftySimContactPlateTransformation.txx"
#endif


#endif /*  __itkNiftySimContactPlateTransformation_h */
