/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-06-01 09:38:00 +0100 (Wed, 01 Jun 2011) $
 Revision          : $Revision: 6322 $
 Last modified by  : $Author: mjc $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkRay_h
#define __itkRay_h

#include "itkMatrix.h"

namespace itk {

/** \class Ray
 *  \brief Class to project a ray through a 3D volume.
 */
template <class TInputImage, class TCoordRep = double>
class ITK_EXPORT Ray
{
public:

  /// Constructor
  Ray();

  /// Destructor
  virtual ~Ray() {};

  typedef itk::Point<TCoordRep, 3>                   InputPointType;
  typedef itk::Point<TCoordRep, 2>                   OutputPointType;

  typedef TInputImage                                InputImageType;
  typedef typename InputImageType::PixelType         PixelType;
  typedef typename InputImageType::IndexType         IndexType;
  typedef typename InputImageType::SizeType          SizeType;
  typedef typename InputImageType::OffsetValueType   OffsetValueType;

  /** The combined affine and perspective projection matrix type. NB:
  defined as a square matrix because ITK does not appear to support
  matrix multiplication of non square matrices. */
  typedef itk::Matrix<double, 4, 4> ProjectionMatrixType;

  /// Set the combined affine and projection matrix
  void SetProjectionMatrix(const ProjectionMatrixType input) { m_ProjectionMatrix = input; }


  /**
   * Set the image class
   */
  void SetImage(const InputImageType *input) { 
    m_Image = input; 
    Initialise();
  }

  /**
   *  Initialise the ray using the position of a point on the 2D
   *  projection image
   *
   *  \param RayPosn       The position of the ray in 2D (mm).
   *
   *  \return True if this is a valid ray.
   */
  bool SetRay(OutputPointType RayPosn);

  /** Set the resolution of the 2D projection image. This is used to
   * normalise the projected intensities to ensure the back-projected
   * 3D image will have similar sum of intensities to the original
   * volume. */
  void SetProjectionResolution2Dmm(double xRes, double yRes) {
    m_ProjectionResolution2Dmm[0] = xRes;
    m_ProjectionResolution2Dmm[1] = yRes;
  }

  /// Return the interpolated intensity of the current ray point.
  double GetCurrentIntensity(void) const;

  /// Return the ray point spacing in mm
  double GetRayPointSpacing(void) const {
    typename InputImageType::SpacingType spacing=this->m_Image->GetSpacing();

    if (m_ValidRay)
      return vcl_sqrt(m_VoxelIncrement[0]*spacing[0]*m_VoxelIncrement[0]*spacing[0]
                    + m_VoxelIncrement[1]*spacing[1]*m_VoxelIncrement[1]*spacing[1]
                    + m_VoxelIncrement[2]*spacing[2]*m_VoxelIncrement[2]*spacing[2] );
    else
      return 0.;
  };

  /// Return the number of points on the ray
  int GetNumberOfRayPoints(void) {return m_TotalRayVoxelPlanes;}

  /// Get the the bilinear coefficient for the current ray point.
  void GetBilinearCoefficients(double &y, double &z) const;

  PixelType **GetRayIntersectionVoxels(void) {return m_RayIntersectionVoxels;}

  /// Get the voxel index of the intersection
  const int *GetRayIntersectionVoxelIndex(void) const {return m_RayIntersectionVoxelIndex;}

  /// Get the traversal direction
  int GetTraversalDirection(void) {return m_TraversalDirection;}

  /** \brief
   *  Integrate the interpolated intensities along the ray and
   *  return the result.
   *
   *  This routine can be called after instantiating the ray and
   *  calling SetProjectionCoord2D() or Reset(). It may then be called
   *  as many times thereafter for different 2D projection
   *  coordinates.
   *
   *  \param integral      The integrated intensities along the ray.
   *
   * \return True if a valid ray was specified.
   */
  bool Integrate(double &integral)
    {
    return IntegrateAboveThreshold(integral, 0);
    };


  /** \brief
   * Integrate the interpolated intensities above a given threshold,
   * along the ray and return the result.
   *
   * This routine can be called after instantiating the ray and
   * calling SetProjectionCoord2D() or Reset(). It may then be called
   * as many times thereafter for different 2D projection
   * coordinates.
   *
   * \param integral      The integrated intensities along the ray.
   * \param threshold     The integration threshold [default value: 0]
   *
   * \return True if a valid ray was specified.
   */
  bool IntegrateAboveThreshold(double &integral, double threshold);

  /** \brief
   * Increment each of the intensities of the 4 planar voxels
   * surrounding every point along the ray using bilinear interpolation.
   *
   * \parameter increment      Intensity increment for each of the current 4 voxels
   */
  bool IncrementRayVoxelIntensities(double increment);

  /** \brief
   * Increment each of the intensities of the 4 planar voxels
   * surrounding the current ray point using bilinear interpolation.
   *
   * \parameter increment      Intensity increment for each of the current 4 voxels
   */
  void IncrementIntensities(double increment);

  /** \brief
   * Increment each of the intensities of the 4 planar voxels
   * surrounding the current ray point by one.
   */
  void IncrementIntensities(void);

  /// Reset the iterator to the start of the ray.
  void Reset(void);

  /** \brief
      Step along the ray.

      This routine can be called iteratively to step along a given ray.
      To specify a new ray call: 'SetRay()' first. To re-traverse
      the current ray call: 'Reset()'.

      @return False if there are no more points on the ray.
  */
  bool NextPoint(void);


protected:

  /// Set the initial zero state of the object
  void ZeroState();

  /// Initialise the object
  void Initialise(void);

  /// Calculate the endpoint coordinats of the ray in voxels.
  void EndPointsInVoxels(void);

  /**
   * Calculate the incremental direction vector in voxels, 'dVoxel',
   * required to traverse the ray.
   */
  void CalcDirnVector(void);

  /**
   * Reduce the length of the ray until both start and end
   * coordinates lie inside the volume.
   *
   * \return True if a valid ray has been, false otherwise.
   */
  bool AdjustRayLength(void);

  /**
   *   Obtain pointers to the four voxels surrounding the point where the ray
   *   enters the volume.
   */
  void InitialiseVoxelPointers(void);

  /// Increment the voxel pointers surrounding the current point on the ray.
  void IncrementVoxelPointers(void);

  /// Record volume dimensions and resolution
  void RecordVolumeDimensions(void);

  /// Define the corners of the volume
  void DefineCorners(void);

  /** \brief
   * Calculate the planes which define the volume.
   *
   * Member function to calculate the equations of the planes of 4 of
   * the sides of the volume, calculate the positions of the 8 corners
   * of the volume in mm in World, also calculate the values of the
   * slopes of the lines which go to make up the volume( defined as
   * lines in cube x,y,z dirn and then each of these lines has a slope
   * in the world x,y,z dirn [3]) and finally also to return the length
   * of the sides of the lines in mm.
   */
  void CalcPlanesAndCorners(void);

  /** \brief
   *  Calculate the ray intercepts with the volume.
   *
   *  See where the ray cuts the volume, check that truncation does not occur,
   *  if not, then start ray where it first intercepts the volume and set
   *  x_max to be where it leaves the volume.
   *
   *  \return True if a valid ray has been specified, false otherwise.
   */
  bool CalcRayIntercepts(void);

  /// Calculate the equation of the ray throught the volume.
  void Line3D(double x2D, double y2D, double r[3], double u[3]);

  

  /**
   *   The ray is traversed by stepping in the axial direction
   *   that enables the greatest number of planes in the volume to be
   *   intercepted.
   */
  typedef enum {
    UNDEFINED_DIRECTION=0,        //!< Undefined
    TRANSVERSE_IN_X,              //!< x
    TRANSVERSE_IN_Y,              //!< y
    TRANSVERSE_IN_Z,              //!< z
    LAST_DIRECTION
  } TraversalDirection;

  // Cache the image in the structure. Skip the smart pointer for
  // efficiency. This inner class will go in/out of scope with every
  // call to Evaluate()
  const InputImageType *m_Image;

  /// Flag indicating whether the current ray is valid
  bool m_ValidRay;

  /// The ray intercept 2D coordinate.
  double m_RayPosition2Dmm[2];

  /// The resolution of the 2D projection image
  double m_ProjectionResolution2Dmm[2];

  /** \brief
   * The start position of the ray in voxels.
   *
   * NB. Two of the components of this coordinate (i.e. those lying within
   * the planes of voxels being traversed) will be shifted by half a
   * voxel. This enables indices of the neighbouring voxels within the plane
   * to be determined by simply casting to 'int' and optionally adding 1.
   */
  double m_RayVoxelStartPosition[3];

  /** \brief
   * The end coordinate of the ray in voxels.
   *
   * NB. Two of the components of this coordinate (i.e. those lying within
   * the planes of voxels being traversed) will be shifted by half a
   * voxel. This enables indices of the neighbouring voxels within the plane
   * to be determined by simply casting to 'int' and optionally adding 1.
   */
  double m_RayVoxelEndPosition[3];


  /** \brief
   * The current coordinate on the ray in voxels.
   *
   * NB. Two of the components of this coordinate (i.e. those lying within
   * the planes of voxels being traversed) will be shifted by half a
   * voxel. This enables indices of the neighbouring voxels within the plane
   * to be determined by simply casting to 'int' and optionally adding 1.
   */
  double m_Position3Dvox[3];

  /** The incremental direction vector of the ray in voxels. */
  double m_VoxelIncrement[3];

  /// The direction in which the ray is incremented thorough the volume (x, y or z).
  TraversalDirection m_TraversalDirection;

  /// The total number of planes of voxels traversed by the ray.
  int m_TotalRayVoxelPlanes;

  /// The current number of planes of voxels traversed by the ray.
  int m_NumVoxelPlanesTraversed;

  /// Pointers to the current four voxels surrounding the ray's trajectory.
  PixelType *m_RayIntersectionVoxels[4];

  /**
   * The voxel coordinate of the bottom-left voxel of the current
   * four voxels surrounding the ray's trajectory.
   */
  int m_RayIntersectionVoxelIndex[3];

  /// The dimension in voxels of the 3D volume in along the x axis
  int m_NumberOfVoxelsInX;
  /// The dimension in voxels of the 3D volume in along the y axis
  int m_NumberOfVoxelsInY;
  /// The dimension in voxels of the 3D volume in along the z axis
  int m_NumberOfVoxelsInZ;

  /// Voxel dimension in x
  double m_VoxelDimensionInX;
  /// Voxel dimension in y
  double m_VoxelDimensionInY;
  /// Voxel dimension in z
  double m_VoxelDimensionInZ;

  /// The coordinate of the point at which the ray enters the volume in mm.
  double m_RayStartCoordInMM[3];
  /// The coordinate of the point at which the ray exits the volume in mm.
  double m_RayEndCoordInMM[3];


  /** \brief
      Planes which define the boundary of the volume in mm
      (six planes and four parameters: Ax+By+Cz+D). */
  double m_BoundingPlane[6][4];
  /// The eight corners of the volume (x,y,z coordinates for each).
  double m_BoundingCorner[8][3];

  /// The position of the ray in 3D
  double m_CurrentRayPositionInMM[3];

  /// The direction of the ray
  double m_RayDirectionInMM[3];

  /// The combined affine and perspective transformation
  ProjectionMatrixType m_ProjectionMatrix;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRay.txx"
#endif

#endif
