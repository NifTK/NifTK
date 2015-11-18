/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkBasicVec3D_h
#define __mitkBasicVec3D_h

#include <math.h>
#include <iostream>

#include "niftkCoreExports.h"

namespace mitk
{

/**
 * \class BasicVec3D
 * \brief Simple 3D Vector implementation that is used in the Surface Extraction 
 * and surface smoothing and decimation algorithms.
 */

class NIFTKCORE_EXPORT BasicVec3D
{ 
public:
  /// \brief Default constructor
  BasicVec3D();
  /// \brief Constructor with coordinates as parameters
  BasicVec3D(float x1, float y1, float z1);
  /// \brief Constructor with coordinates as parameters in a float array
  BasicVec3D(float av[3]);
  /// \brief Constructor with coordinates as parameters in a Vec3D
  BasicVec3D(const BasicVec3D& v);
  /// \brief Destructor that intentionally does nothing
  virtual ~BasicVec3D() {  };

  /// \brief Assignment operator
  BasicVec3D& operator=(const BasicVec3D& v);

  /// \brief Comparision operator
  bool operator==(const BasicVec3D& v);
  /// \brief Not equal operator
  bool operator!=(const BasicVec3D& v);

  //@{
  /** Scalar operations */
  BasicVec3D operator+(float f) const;
  BasicVec3D operator-(float f) const;
  BasicVec3D operator*(float f) const;
  BasicVec3D operator/(float f) const;

  BasicVec3D& operator+=(float f);
  BasicVec3D& operator-=(float f);
  BasicVec3D& operator*=(float f);
  BasicVec3D& operator/=(float f);
  //@}

  //@{
  /** Vector operations */
  BasicVec3D  operator+ (const BasicVec3D& v) const;
  BasicVec3D& operator+=(const BasicVec3D& v);
  BasicVec3D  operator- (const BasicVec3D& v) const;
  BasicVec3D& operator-=(const BasicVec3D& v);
  //@}

  /// \brief Unary operator that changes the sign of the coordinates
  BasicVec3D operator-() const;

  //@{
  /** Other operations */
  friend BasicVec3D operator*(float a, const BasicVec3D& v) { return BasicVec3D (a * v.GetX(), a * v.GetY(), a * v.GetZ()); }
  friend std::ostream& operator<<(std::ostream& os, const BasicVec3D& vo);
  //@}

  //@{
  /** Dot and Cross Products */
  float      Dot(const BasicVec3D& v) const;
  BasicVec3D Cross(const BasicVec3D& v) const;
  BasicVec3D NormalizedCross(const BasicVec3D& v) const;

  float      Dot(const BasicVec3D& v1, const BasicVec3D& v2);
  BasicVec3D Cross(const BasicVec3D& v1, const BasicVec3D& v2);
  BasicVec3D NormalizedCross(const BasicVec3D& v1, const BasicVec3D& v2);
  //@}

  /// \brief Normalize the vector
  void  Normalize();
  /// \brief Set all coordinates to zero
  void  SetZero();
  /// \brief Compute and return dot product with self
  float SelfDot();
  /// \brief Compute and return length of the vector
  float Length();

  /// \brief Compute and return distance from a point
  float Distance(const BasicVec3D& v) const;
  /// \brief Compute and return squared distance from a point
  float DistanceSquared(const BasicVec3D& v) const;

  /// \brief Compute and return distance of two points
  float Distance(const BasicVec3D& v1, const BasicVec3D& v2);
  /// \brief Compute and return squared distance of two points
  float DistanceSquared(const BasicVec3D& v1, const BasicVec3D& v2);

  /// \brief Sets the X coordinate of the vector
  void  SetX(float x) { m_X = x; }
  /// \brief Returns the X coordinate of the vector
  float GetX() const  { return m_X; }

  /// \brief Sets the Y coordinate of the vector
  void  SetY(float y) { m_Y = y; }
  /// \brief Returns the Y coordinate of the vector
  float GetY() const  { return m_Y; }

  /// \brief Sets the Z coordinate of the vector
  void  SetZ(float z) { m_Z = z; }
  /// \brief Returns the Z coordinate of the vector
  float GetZ() const  { return m_Z; }

private:
  //@{
  /** X-Y-Z coordinates */
  float m_X;
  float m_Y;
  float m_Z;
  //@}
};

} // end of namespace

#endif // #ifndef __mitkBasicVec3D_h