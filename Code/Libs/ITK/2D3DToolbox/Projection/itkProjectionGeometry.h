/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkProjectionGeometry_h
#define __itkProjectionGeometry_h

#include "itkImage.h"
#include "itkPerspectiveProjectionTransform.h"
#include "itkEulerAffineTransform.h"

namespace itk {

/** \class ProjectionGeometry
 *  \brief Abstract class to calculate the geometry of a CT or tomo
 *  machine.
 */
template <class IntensityType = float>
class ITK_EXPORT ProjectionGeometry : public Object
{
public:

  /** Standard class typedefs. */
  typedef ProjectionGeometry         Self;
  typedef Object                        Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ProjectionGeometry, Object);

  /** Some convenient typedefs. */
  typedef typename itk::Size<2>              ProjectionSizeType;
  typedef typename itk::Vector<double,2>     ProjectionSpacingType;

  typedef typename itk::Size<3>              VolumeSizeType;
  typedef typename itk::Vector<double,3>     VolumeSpacingType;

  typedef typename itk::EulerAffineTransform<double, 3, 3>        EulerAffineTransformType;
  typedef typename EulerAffineTransformType::Pointer              EulerAffineTransformPointerType;

  typedef typename itk::PerspectiveProjectionTransform<double>    PerspectiveProjectionTransformType;
  typedef typename PerspectiveProjectionTransformType::Pointer    PerspectiveProjectionTransformPointerType;

  /** Return a pointer to the perspective projection matrix for
      projection 'i'. */
  virtual PerspectiveProjectionTransformType::Pointer GetPerspectiveTransform(int i)
  { niftkitkErrorMacro( "Subclasses should override this method" ); return 0; }

  /** Return a pointer to the affine transformation matrix for
      projection 'i'. */
  virtual EulerAffineTransformType::Pointer GetAffineTransform(int i)
  { niftkitkErrorMacro( "Subclasses should override this method" ); return 0; }

  /// Set a rotation in 'x'
  itkSetMacro(RotationInX, double);
  /// Set a rotation in 'y'
  itkSetMacro(RotationInY, double);
  /// Set a rotation in 'z'
  itkSetMacro(RotationInZ, double);

  /// Get a rotation in 'x'
  itkGetMacro(RotationInX, double);
  /// Get a rotation in 'y'
  itkGetMacro(RotationInY, double);
  /// Get a rotation in 'z'
  itkGetMacro(RotationInZ, double);

  /// Set the projection size
  void SetProjectionSize(const ProjectionSizeType &r) {m_ProjectionSize = r; m_FlagInitialised = false;}
  /// Set the projection spacing
  void SetProjectionSpacing(const ProjectionSpacingType &s) {m_ProjectionSpacing = s; m_FlagInitialised = false;}

  /// Set the volume size
  void SetVolumeSize(const VolumeSizeType &r) {m_VolumeSize = r; m_FlagInitialised = false;}
  /// Set the volume spacing
  void SetVolumeSpacing(const VolumeSpacingType &s) {m_VolumeSpacing = s; m_FlagInitialised = false;}

  /// Return the number of projections for derived geometries
  virtual unsigned int GetNumberOfProjections(void)
  { niftkitkErrorMacro( "Subclasses should override this method" ); return 0;}
  

protected:
  ProjectionGeometry();
  virtual ~ProjectionGeometry() {};

  void PrintSelf(std::ostream& os, Indent indent) const{
    Superclass::PrintSelf(os, indent);
  }

  /// Initialise the object and check inputs are defined
  void Initialise(void);

  /// Flag indicating whether the object has been initialised
  bool m_FlagInitialised;

  /// Rotation in 'x' to allow reorientation of the volume
  double m_RotationInX;
  /// Rotation in 'y' to allow reorientation of the volume
  double m_RotationInY;
  /// Rotation in 'z' to allow reorientation of the volume
  double m_RotationInZ;

  /// A pointer to the 3D volume size
  VolumeSizeType m_VolumeSize;
  /// A pointer to the 3D volume spacing
  VolumeSpacingType m_VolumeSpacing;

  /// A pointer to the 3D projection size
  ProjectionSizeType m_ProjectionSize;
  /// A pointer to the 3D projection spacing
  ProjectionSpacingType m_ProjectionSpacing;

private:
  ProjectionGeometry(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkProjectionGeometry.txx"
#endif

#endif
