/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <assert.h>
#include "mitkBasicTriangle.h"

#include "mitkBasicMesh.h"

namespace mitk
{
// Constructors
BasicTriangle::BasicTriangle()
: m_Vert1(-1)
, m_Vert2(-1)
, m_Vert3(-1)
, m_Mesh(0)
, m_TriNormal(0.0, 0.0, 0.0)
, m_Active(true)
, m_Index(-1)
{
}

BasicTriangle::~BasicTriangle()
{
}

BasicTriangle::BasicTriangle(int v1, int v2, int v3)
: m_Vert1(v1)
, m_Vert2(v2)
, m_Vert3(v3)
, m_Mesh(0)
, m_TriNormal(0.0, 0.0, 0.0)
, m_Active(true)
, m_Index(-1)
{
}

BasicTriangle::BasicTriangle(mitk::BasicMesh* mp, int v1, int v2, int v3)
: m_Vert1(v1)
, m_Vert2(v2)
, m_Vert3(v3)
, m_Mesh(mp)
, m_TriNormal(0.0, 0.0, 0.0)
, m_Active(true)
, m_Index(-1)
{
  assert(mp);

  CalcNormal();
}

// copy ctor
BasicTriangle::BasicTriangle(const BasicTriangle& t)
: m_Vert1(t.m_Vert1)
, m_Vert2(t.m_Vert2)
, m_Vert3(t.m_Vert3)
, m_Mesh(t.m_Mesh)
, m_TriNormal(t.m_TriNormal)
, m_DParam(t.m_DParam)
, m_Active(t.m_Active)
, m_Index(t.m_Index)
{
  //CalcNormal();
};

// assignment operator
BasicTriangle& BasicTriangle::operator=(const BasicTriangle& t)
{
  //Check for assignment to self
  if (&t == this)
    return *this;

  m_Vert1     = t.m_Vert1;
  m_Vert2     = t.m_Vert2;
  m_Vert3     = t.m_Vert3;
  m_Mesh      = t.m_Mesh;
  m_TriNormal = t.m_TriNormal;
  m_DParam    = t.m_DParam;
  m_Active    = t.m_Active;
  m_Index     = t.m_Index;

  return *this;
}

// assumes pointing to same list of verts
bool BasicTriangle::operator==(const BasicTriangle& t)
{
  return (m_Vert1 == t.m_Vert1 && 
          m_Vert2 == t.m_Vert2 && 
          m_Vert3 == t.m_Vert3 &&
          m_Mesh == t.m_Mesh
    );
}

bool BasicTriangle::HasVertex(int vi)
{
  return (vi == m_Vert1 ||
          vi == m_Vert2 ||
          vi == m_Vert3);
}

// When we collapse an edge, we may change the vertex of a BasicTriangle.
void BasicTriangle::ChangeVertex(int vFrom, int vTo)
{
  assert(vFrom != vTo);
  assert(vFrom == m_Vert1 || vFrom == m_Vert2 || vFrom == m_Vert3);

  if (vFrom == m_Vert1)
  {
    m_Vert1 = vTo;
  } 
  else if (vFrom == m_Vert2)
  {
    m_Vert2 = vTo;
  }
  else if (vFrom == m_Vert3)
  {
    m_Vert3 = vTo;
  }
  else
  {
    //!FIX error
  }
}

void BasicTriangle::GetVertIndices(int& v1, int& v2, int& v3)
{
  v1=m_Vert1;
  v2=m_Vert2;
  v3=m_Vert3;
}


// retrieve vertices as an array of floats
const float* BasicTriangle::GetVert1CoordArray()
{
  return (m_Mesh->GetVertex(m_Vert1)).GetCoordArray();
}

const float* BasicTriangle::GetVert2CoordArray()
{
  return (m_Mesh->GetVertex(m_Vert2)).GetCoordArray();
}

const float* BasicTriangle::GetVert3CoordArray()
{
  return (m_Mesh->GetVertex(m_Vert3)).GetCoordArray();
}

// retrieve vertices as a Vec3D object
const BasicVec3D& BasicTriangle::GetVert1Coords() const
{
  return m_Mesh->GetVertex(m_Vert1).GetCoords();
}

const BasicVec3D& BasicTriangle::GetVert2Coords() const
{
  return m_Mesh->GetVertex(m_Vert2).GetCoords();
}

const BasicVec3D& BasicTriangle::GetVert3Coords() const 
{
  return m_Mesh->GetVertex(m_Vert3).GetCoords();
}

// retrieve vertices as a vertex object
const BasicVertex& BasicTriangle::GetVert1() const
{
  return m_Mesh->GetVertex(m_Vert1);
}

const BasicVertex& BasicTriangle::GetVert2() const
{
  return m_Mesh->GetVertex(m_Vert2);
}

const BasicVertex& BasicTriangle::GetVert3() const 
{
  return m_Mesh->GetVertex(m_Vert3);
}

const float* BasicTriangle::GetVert1NormalArray()
{
  return (m_Mesh->GetVertex(m_Vert1)).GetNormalArray();
}

const float* BasicTriangle::GetVert2NormalArray()
{
  return (m_Mesh->GetVertex(m_Vert2)).GetNormalArray();
}

const float* BasicTriangle::GetVert3NormalArray()
{
  return (m_Mesh->GetVertex(m_Vert3)).GetNormalArray();
}

const BasicVec3D& BasicTriangle::GetVert1Normal() const
{
  return m_Mesh->GetVertex(m_Vert1).GetNormal();
}

const BasicVec3D& BasicTriangle::GetVert2Normal() const
{
  return m_Mesh->GetVertex(m_Vert2).GetNormal();
}

const BasicVec3D& BasicTriangle::GetVert3Normal() const
{
  return m_Mesh->GetVertex(m_Vert3).GetNormal();
}

float* BasicTriangle::GetTriNormalArray()
{
  m_TriNormalArray[0]=m_TriNormal.GetX();
  m_TriNormalArray[1]=m_TriNormal.GetY();
  m_TriNormalArray[2]=m_TriNormal.GetZ();
  return m_TriNormalArray; 
}

// Calculate normal of triangle
void BasicTriangle::CalcNormal()
{
  if (m_Mesh == 0)
    return;
  //assert(m_Mesh);

  BasicVec3D vec1 = (m_Mesh->GetVertex(m_Vert1)).GetCoords();
  BasicVec3D vec2 = (m_Mesh->GetVertex(m_Vert2)).GetCoords();
  BasicVec3D vec3 = (m_Mesh->GetVertex(m_Vert3)).GetCoords();
  BasicVec3D veca = vec2 - vec1;
  BasicVec3D vecb = vec3 - vec1;

  m_TriNormal = veca.NormalizedCross(vecb);
  // Note that if the triangle is degenerate (all vertices lie in a line),
  // the normal will be <0,0,0>

  // This is the "d" from the plane equation ax + by + cz + d = 0;

  m_DParam = -m_TriNormal.Dot(vec1);
}

// Calculate area of triangle
float BasicTriangle::CalcArea()
{
  assert(m_Mesh);

  // If a triangle is defined by 3 points, say p, q and r, then
  // its area is 0.5 * length of ((p - r) cross (q - r))
  // See Real-Time Rendering book, Appendix A
  BasicVec3D vec1 = (m_Mesh->GetVertex(m_Vert1)).GetCoords();
  BasicVec3D vec2 = (m_Mesh->GetVertex(m_Vert2)).GetCoords();
  BasicVec3D vec3 = (m_Mesh->GetVertex(m_Vert3)).GetCoords();
  BasicVec3D vecA = vec1 - vec2;
  BasicVec3D vecB = vec3 - vec2;

  BasicVec3D crossProd = vecA.Cross(vecB);
  float area = float(0.5 * crossProd.Length());
  return area;
}

int BasicTriangle::GetVertIndex(int which)
{
  int retVal = -1;
  switch (which)
  {
  case 0:
    retVal = m_Vert1;
    break;
  case 1:
    retVal = m_Vert2;
    break;
  case 2:
    retVal = m_Vert3;
    break;
  }

  return retVal;
}

void BasicTriangle::SetVertIndex(int which, int indexVal)
{
  switch (which)
  {
  case 0:
    m_Vert1 = indexVal;
    break;
  case 1:
    m_Vert2 = indexVal;
    break;
  case 2:
    m_Vert3 = indexVal;
    break;
  }
}


// Used for output
std::ostream& operator<<(std::ostream& os, const BasicTriangle& to)
{
  os << "vert1: " << to.m_Vert1 << " vert2: " << to.m_Vert2 << " vert3: " << to.m_Vert3; // for some reason this isn't working as a friend function, not sure why
  os << " Normal: " << to.m_TriNormal << " Active? " << to.IsActive();
  os << " Index: " << to.m_Index;

  // it is pulling ostream from the STL typedef, not the regular ostream, though.
  return os;
}

}