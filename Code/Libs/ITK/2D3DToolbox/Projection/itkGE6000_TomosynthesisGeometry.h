/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkGE6000_TomosynthesisGeometry_h
#define __itkGE6000_TomosynthesisGeometry_h

#include "itkProjectionGeometry.h"

namespace itk {

/** \class GE6000_TomosynthesisGeometry
 *  \brief Class to calculate the geometry of a GE tomosynthesis
 *  machine.
 */
template <class IntensityType = float>
class ITK_EXPORT GE6000_TomosynthesisGeometry :
    public ProjectionGeometry<IntensityType>
{
public:

  /** Standard class typedefs. */
  typedef GE6000_TomosynthesisGeometry      Self;
  typedef ProjectionGeometry<IntensityType> Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GE6000_TomosynthesisGeometry, ProjectionGeometry);

  /** Some convenient typedefs. */
  typedef typename Superclass::ProjectionSizeType            ProjectionSizeType;
  typedef typename Superclass::ProjectionSpacingType         ProjectionSpacingType;

  typedef typename Superclass::VolumeSizeType                VolumeSizeType;
  typedef typename Superclass::VolumeSpacingType             VolumeSpacingType;

  typedef typename Superclass::EulerAffineTransformType                  EulerAffineTransformType;
  typedef typename Superclass::EulerAffineTransformPointerType           EulerAffineTransformPointerType;

  typedef typename Superclass::PerspectiveProjectionTransformType        PerspectiveProjectionTransformType;
  typedef typename Superclass::PerspectiveProjectionTransformPointerType PerspectiveProjectionTransformPointerType;

  /** Return a pointer to the perspective projection matrix for
      projection 'i'. */
  virtual PerspectiveProjectionTransformPointerType GetPerspectiveTransform(int i);

  /** Return a pointer to the affine transformation matrix for
      projection 'i'. */
  virtual EulerAffineTransformPointerType GetAffineTransform(int i);

  /// Return the number of projections for this geometry
  virtual unsigned int GetNumberOfProjections(void) { return 15; }


protected:
  GE6000_TomosynthesisGeometry();
  virtual ~GE6000_TomosynthesisGeometry() {}
  void PrintSelf(std::ostream& os, Indent indent) const;

  /// Calculate the projection normal position
  double CalcNormalPosition(double alpha);


private:
  GE6000_TomosynthesisGeometry(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGE6000_TomosynthesisGeometry.txx"
#endif

#endif

