/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkBasicVec3D.h"

namespace mitk
{

// Constructors and Destructors
BasicVec3D::BasicVec3D() 
{
  m_X = 0.0f;
  m_Y = 0.0f;
  m_Z = 0.0f;
}

BasicVec3D::BasicVec3D(float x1, float y1, float z1)
{
  m_X = x1; 
  m_Y = y1; 
  m_Z = z1;
}

BasicVec3D::BasicVec3D(float av[3])
{
  m_X = av[0];
  m_Y = av[1];
  m_Z = av[2];
}

BasicVec3D::BasicVec3D(const BasicVec3D& other)
{
  m_X = other.m_X;
  m_Y = other.m_Y;
  m_Z = other.m_Z;
}

// Assignment operator
BasicVec3D& BasicVec3D::operator=(const BasicVec3D& v)
{ 
  m_X = v.m_X;
  m_Y = v.m_Y;
  m_Z = v.m_Z;
  
  return *this;
}

// Comparision operators
bool BasicVec3D::operator==(const BasicVec3D& v)
{ 
  return (m_X == v.m_X && m_Y == v.m_Y && m_Z == v.m_Z);
}

bool BasicVec3D::operator!=(const BasicVec3D& v)
{ 
  return (m_X != v.m_X || m_Y != v.m_Y || m_Z != v.m_Z);
}

// Scalar operations
BasicVec3D BasicVec3D::operator+(float f) const
{ 
  return BasicVec3D(m_X + f, m_Y + f, m_Z + f);
}

BasicVec3D BasicVec3D::operator-(float f) const
{ 
  return BasicVec3D(m_X - f, m_Y - f, m_Z - f);
}

BasicVec3D BasicVec3D::operator*(float f) const
{ 
  return BasicVec3D(m_X * f, m_Y * f, m_Z * f);
}

BasicVec3D BasicVec3D::operator/(float f) const
{ 
  BasicVec3D v1(m_X,m_Y,m_Z);

  if (f != 0.0f)
  { 
    v1.m_X /= f;
    v1.m_Y /= f;
    v1.m_Z /= f;
  } 

  return v1;
}

BasicVec3D& BasicVec3D::operator+=(float f)
{ 
  m_X += f;
  m_Y += f;
  m_Z += f;
  return *this;
}

BasicVec3D& BasicVec3D::operator-=(float f)
{ 
  m_X -= f;
  m_Y -= f;
  m_Z -= f;
  return *this;
}

BasicVec3D& BasicVec3D::operator*=(float f)
{ 
  m_X *= f;
  m_Y *= f;
  m_Z *= f;
  return *this;
}

BasicVec3D& BasicVec3D::operator/=(float f)
{ 
  if(f!=0.0f)
  {  
    m_X /= f;
    m_Y /= f;
    m_Z /= f;
  }

  return *this;
}

// Vector operations
BasicVec3D BasicVec3D::operator+(const BasicVec3D& v) const
{
  return BasicVec3D(m_X + v.m_X, m_Y + v.m_Y, m_Z + v.m_Z);
}

BasicVec3D& BasicVec3D::operator+=(const BasicVec3D& v)
{ 
  m_X += v.m_X;
  m_Y += v.m_Y;
  m_Z += v.m_Z;
  return *this;
}

BasicVec3D  BasicVec3D::operator-(const BasicVec3D& v) const
{
  return BasicVec3D(m_X - v.m_X, m_Y - v.m_Y, m_Z - v.m_Z);
}

BasicVec3D& BasicVec3D::operator-=(const BasicVec3D& v)
{
  m_X -= v.m_X;
  m_Y -= v.m_Y;
  m_Z -= v.m_Z;
  return *this;
}

// Unary operators - change sign
BasicVec3D BasicVec3D::operator-() const
{
  return BasicVec3D (-m_X, -m_Y, -m_Z);
}

std::ostream& operator<<(std::ostream& os, const BasicVec3D& vo)
{
  return os << "<" << vo.m_X << ", " << vo.m_Y << ", " << vo.m_Z << ">";
}

// Dot and Cross Products
float BasicVec3D::Dot(const BasicVec3D& v) const
{ 
  return (m_X * v.m_X + m_Y * v.m_Y + m_Z * v.m_Z);
}

BasicVec3D BasicVec3D::Cross(const BasicVec3D& v) const
{ 
  BasicVec3D vr(m_Y * v.m_Z - m_Z * v.m_Y, m_Z * v.m_X - m_X * v.m_Z, m_X * v.m_Y - m_Y * v.m_X);
  return vr;
}

BasicVec3D BasicVec3D::NormalizedCross(const BasicVec3D& v) const
{ 
  BasicVec3D vr(m_Y * v.m_Z - m_Z * v.m_Y, m_Z * v.m_X - m_X * v.m_Z, m_X * v.m_Y - m_Y * v.m_X);
  vr.Normalize();
  return vr;
}

// dot and cross products
float BasicVec3D::Dot(const BasicVec3D& v1, const BasicVec3D& v2) 
{ 
  return (v1.m_X * v2.m_X + v1.m_Y * v2.m_Y +v1. m_Z * v2.m_Z);
}

BasicVec3D BasicVec3D::Cross(const BasicVec3D& v1, const BasicVec3D& v2)
{ 
  BasicVec3D vr (v1.m_Y * v2.m_Z - v1.m_Z * v2.m_Y,
           v1.m_Z * v2.m_X - v1.m_X * v2.m_Z,
           v1.m_X * v2.m_Y - v1.m_Y * v2.m_X); 

  return vr;
}

BasicVec3D BasicVec3D::NormalizedCross(const BasicVec3D& v1, const BasicVec3D& v2)
{ 
  BasicVec3D vr(v1.m_Y * v2.m_Z - v1.m_Z * v2.m_Y,
          v1.m_Z * v2.m_X - v1.m_X * v2.m_Z,
          v1.m_X * v2.m_Y - v1.m_Y * v2.m_X); 

  vr.Normalize();
  return vr;
}

// Miscellaneous
void BasicVec3D::Normalize()
{ 
  float a = float(sqrt(m_X*m_X + m_Y*m_Y + m_Z*m_Z));

  if (a!=0.0f) 
  {
    m_X/=a;
    m_Y/=a;
    m_Z/=a;
  }
}

void BasicVec3D::SetZero()
{ 
  m_X = 0.0f;
  m_Y = 0.0f;
  m_Z = 0.0f;
}

float BasicVec3D::SelfDot()
{ 
  return m_X*m_X + m_Y*m_Y + m_Z*m_Z;
}


float BasicVec3D::Length()
{ 
  return float(sqrt(SelfDot()));
}

float BasicVec3D::Distance(const BasicVec3D& v) const
{
  return sqrt(DistanceSquared(v));
}

float BasicVec3D::DistanceSquared(const BasicVec3D& v) const
{
  return (v.m_X - m_X)*(v.m_X - m_X) + (v.m_Y - m_Y)*(v.m_Y - m_Y) + (v.m_Z - m_Z)*(v.m_Z - m_Z);
}

float BasicVec3D::Distance(const BasicVec3D& v1, const BasicVec3D& v2)
{
  return sqrt(DistanceSquared(v1, v2));
}

float BasicVec3D::DistanceSquared(const BasicVec3D& v1, const BasicVec3D& v2)
{
  return (v1.m_X - v2.m_X)*(v1.m_X - v2.m_X) + (v1.m_Y - v2.m_Y)*(v1.m_Y - v2.m_Y) + (v1.m_Z - v2.m_Z)*(v1.m_Z - v2.m_Z);
}

}