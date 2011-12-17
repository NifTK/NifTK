/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkIsocentricConeBeamRotationGeometry_h
#define __itkIsocentricConeBeamRotationGeometry_h

#include "itkProjectionGeometry.h"

namespace itk {

enum 
IsocentricConeBeamRotationTypeEnum
{
  ISOCENTRIC_CONE_BEAM_ROTATION_IN_X,
  ISOCENTRIC_CONE_BEAM_ROTATION_IN_Y,
  ISOCENTRIC_CONE_BEAM_ROTATION_IN_Z
};


/** \class IsocentricConeBeamRotationGeometry
 *  \brief Class to calculate the geometry of an isocentric
 *  cone beam projection CT or tomosynthesis machine.
 */
template <class IntensityType = float>
class ITK_EXPORT IsocentricConeBeamRotationGeometry :
    public ProjectionGeometry<IntensityType>
{
public:

  /** Standard class typedefs. */
  typedef IsocentricConeBeamRotationGeometry   Self;
  typedef ProjectionGeometry<IntensityType>    Superclass;
  typedef SmartPointer<Self>                   Pointer;
  typedef SmartPointer<const Self>             ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(IsocentricConeBeamRotationGeometry, ProjectionGeometry);

  /** Some convenient typedefs. */
  typedef typename Superclass::ProjectionSizeType            ProjectionSizeType;
  typedef typename Superclass::ProjectionSpacingType         ProjectionSpacingType;

  typedef typename Superclass::VolumeSizeType                VolumeSizeType;
  typedef typename Superclass::VolumeSpacingType             VolumeSpacingType;

  typedef typename Superclass::EulerAffineTransformType                  EulerAffineTransformType;
  typedef typename Superclass::EulerAffineTransformPointerType           EulerAffineTransformPointerType;

  typedef typename Superclass::PerspectiveProjectionTransformType        PerspectiveProjectionTransformType;
  typedef typename Superclass::PerspectiveProjectionTransformPointerType PerspectiveProjectionTransformPointerType;

  itkSetMacro( NumberOfProjections, unsigned int );
  itkSetMacro( FirstAngle, double );
  itkSetMacro( AngularRange, double );
  itkSetMacro( FocalLength, double );

  /** Set the object translation. */
  void SetTranslation(double tx, double ty, double tz) {
    m_Translation[0] = tx;
    m_Translation[1] = ty;
    m_Translation[2] = tz;
    this->Modified();
  }

  /// Initialise the object
  virtual void Initialise(void);

  /** Set the axis about which to rotate. */
  void SetRotationAxis(IsocentricConeBeamRotationTypeEnum axis) {m_RotationType = axis;}

  /** Return a pointer to the perspective projection matrix for
      projection 'i'. */
  virtual PerspectiveProjectionTransformPointerType GetPerspectiveTransform(int i);

  /** Return a pointer to the affine transformation matrix for
      projection 'i'. */
  virtual EulerAffineTransformPointerType GetAffineTransform(int i);

  /// Return the number of projections for this geometry
  virtual unsigned int GetNumberOfProjections(void) { return m_NumberOfProjections; }

protected:
  IsocentricConeBeamRotationGeometry();
  virtual ~IsocentricConeBeamRotationGeometry() {if (m_ProjectionAngles) delete[] m_ProjectionAngles;}
  void PrintSelf(std::ostream& os, Indent indent) const;


private:
  IsocentricConeBeamRotationGeometry(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  /// Whether to rotate in 'x', 'y' or 'z'
  IsocentricConeBeamRotationTypeEnum m_RotationType;

  /// The number of projections in the sequence
  unsigned int m_NumberOfProjections; 
  
  /// The angle of the first projection in the sequence
  double m_FirstAngle;      
  
  /// The full angular range of the sequence
  double m_AngularRange;   

  /// The focal length of the projection
  double m_FocalLength;

  /** Additional translation if the object is not half way between the
      source and detector */
  double m_Translation[3];

  /// Define the array to store the projection angles
  double *m_ProjectionAngles;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkIsocentricConeBeamRotationGeometry.txx"
#endif

#endif
