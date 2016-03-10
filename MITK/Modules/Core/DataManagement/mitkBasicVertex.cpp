/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkBasicVertex.h"
#include "mitkBasicMesh.h"

namespace mitk
{

// Constructors and Destructors
BasicVertex::BasicVertex()
: m_Coords(0.0, 0.0, 0.0)
, m_Normal(0.0, 0.0, 0.0)
, m_Active(false)
, m_Cost(0)
, m_MinCostNeighbor(-1)
, m_Index(-1)
, m_QuadricTriArea(0)
{
  InitQuadric();
}

BasicVertex::BasicVertex(float x1, float y1, float z1)
: m_Coords(x1, y1, z1)
, m_Normal(0.0, 0.0, 0.0)
, m_Active(true)
, m_Cost(0)
, m_MinCostNeighbor(-1)
, m_Index(-1)
, m_QuadricTriArea(0)
{
  InitQuadric();
}

BasicVertex::BasicVertex(float av[3])
: m_Coords(av)
, m_Normal(0.0, 0.0, 0.0)
, m_Active(true)
, m_Cost(0)
, m_MinCostNeighbor(-1)
, m_Index(-1)
, m_QuadricTriArea(0)
{
  InitQuadric();
}

BasicVertex::BasicVertex(float av[3], float vn[3])
: m_Coords(av)
, m_Normal(vn)
, m_Active(true)
, m_Cost(0)
, m_MinCostNeighbor(-1)
, m_Index(-1)
, m_QuadricTriArea(0)
{
  InitQuadric();
}

// copy ctor
BasicVertex::BasicVertex(const BasicVertex& v)
: m_Coords(v.m_Coords)
, m_Normal(v.m_Normal)
, m_VertNeighbors(v.m_VertNeighbors)
, m_TriNeighbors(v.m_TriNeighbors)
, m_Active(v.m_Active)
, m_Cost(v.m_Cost)
, m_MinCostNeighbor(v.m_MinCostNeighbor)
, m_Index(v.m_Index)
, m_QuadricTriArea(v.m_QuadricTriArea)
{
  // copy quadric
  for (int i = 0; i < 4; ++i)
  { 
    for (int j = 0; j < 4; ++j)
    { 
      m_QuadricError[i][j] = v.m_QuadricError[i][j];
    }
  }
}

// destructor
BasicVertex::~BasicVertex() 
{ 
  m_VertNeighbors.clear();
  m_TriNeighbors.clear();
  //if (m_VertNeighbors.size() != 0)
  //  m_VertNeighbors.erase(m_VertNeighbors.begin(), m_VertNeighbors.end());
  //
  //if (m_TriNeighbors.size() != 0)
  //  m_TriNeighbors.erase(m_TriNeighbors.begin(), m_TriNeighbors.end());
}


// Assignment operator
BasicVertex& BasicVertex::operator=(const BasicVertex& v) 
{ 
  if (this == &v)
    return *this; // check for assignment to self

  m_Coords =v.m_Coords;
  m_Normal = v.m_Normal; 
  m_VertNeighbors = v.m_VertNeighbors;
  m_TriNeighbors = v.m_TriNeighbors;
  m_Active = v.m_Active;
  m_Cost = v.m_Cost;
  m_MinCostNeighbor = v.m_MinCostNeighbor;
  m_Index = v.m_Index;
  m_QuadricTriArea = v.m_QuadricTriArea;

  // copy quadric
  for (int i = 0; i < 4; ++i) 
  { 
    for (int j = 0; j < 4; ++j) 
    { 
      m_QuadricError[i][j] = v.m_QuadricError[i][j];
    }
  }
  return *this;
}

// Assignment operator
BasicVertex& BasicVertex::operator=(const float av[3])
{ 
  m_Coords.SetX(av[0]);
  m_Coords.SetY(av[1]);
  m_Coords.SetZ(av[2]);

  // erase the list of neighboring vertices, faces
  // since we're copying from an array of floats
  m_VertNeighbors.erase(m_VertNeighbors.begin(), m_VertNeighbors.end());
  m_TriNeighbors.erase(m_TriNeighbors.begin(), m_TriNeighbors.end());
  m_Cost = 0;
  m_MinCostNeighbor = -1;
  m_Index = -1;
  m_QuadricTriArea = 0;
  InitQuadric();
  return *this;
}

// Comparision operators
bool BasicVertex::operator==(const BasicVertex& v)
{ 
  bool isEqual = (m_Coords == v.m_Coords && m_Normal == v.m_Normal);
  return isEqual;
}

bool BasicVertex::operator!=(const BasicVertex& v) 
{ 
  return (m_Coords != v.m_Coords || m_Normal != v.m_Normal);
}


bool BasicVertex::operator<(const BasicVertex &right) const
{
  if(m_Coords.GetX() < right.GetCoordX())
    return true;
  else if(m_Coords.GetX() > right.GetCoordX())
    return false;

  if(m_Coords.GetY() < right.GetCoordY())
    return true;
  else if(m_Coords.GetY() > right.GetCoordY())
    return false;

  if(m_Coords.GetZ() < right.GetCoordZ())
    return true;
  else if(m_Coords.GetZ() >  right.GetCoordZ())
    return false;

  return false;
}

bool BasicVertex::operator>(const BasicVertex &right) const
{
  if(m_Coords.GetX() > right.GetCoordX())
    return true;
  else if(m_Coords.GetX() < right.GetCoordX())
    return false;

  if(m_Coords.GetY() > right.GetCoordY())
    return true;
  else if(m_Coords.GetY() < right.GetCoordY())
    return false;

  if(m_Coords.GetZ() > right.GetCoordZ())
    return true;
  else if(m_Coords.GetZ() <  right.GetCoordZ())
    return false;

  return false;
}



// Friend functions for input and output

std::ostream&
  operator<<(std::ostream& os, const BasicVertex& vo)

{
  //	return (os << "<" << vo.m_Coords.x << ", " << vo.m_Coords.y << ", " << vo.m_Coords.z << ">");
  os << " Index: " << vo.GetIndex() << " ";
  os << vo.GetCoordArray(); // for some reason this isn't working as a friend function, not sure why
  // it is pulling ostream from the STL typedef, not the regular ostream, though.
  set<int>::iterator pos;
  os << " Vert Neighbors:";
  for (pos = vo.GetVertNeighbors().begin(); pos != vo.GetVertNeighbors().end(); ++pos)
  {
    os << " " << *pos;
  }
  os << " Tri Neighbors:";
  for (pos = vo.GetTriNeighbors().begin(); pos != vo.GetTriNeighbors().end(); ++pos)
  {
    os << " " << *pos;
  }

  os << " Is Active: " << vo.IsActive();
  os << " Cost: " << vo.GetCost();
  os << " Min Vert: " << vo.GetMinCostEdgeVert();
  return os;
}

// NOTE: a better solution would be to return a reference
const float* BasicVertex::GetCoordArray() const 
{ 
  m_V[0]=m_Coords.GetX();
  m_V[1]=m_Coords.GetY();
  m_V[2]=m_Coords.GetZ();
  return m_V;
}

const float* BasicVertex::GetNormalArray() const 
{ 
  m_VN[0]=m_Normal.GetX();
  m_VN[1]=m_Normal.GetY();
  m_VN[2]=m_Normal.GetZ();
  return m_VN;
}

// Used for Garland & Heckbert's quadric edge collapse cost (used for mesh simplifications/progressive meshes)
void BasicVertex::InitQuadric()
{ 
  for (int i = 0; i < 4; ++i)
  { 
    for (int j = 0; j < 4; ++j)
    { 
      m_QuadricError[i][j] = -1;
    }
  }
}


// Calculate the Quadric 4x4 matrix
void BasicVertex::CalcQuadric(BasicMesh& m, bool bUseTriArea)
{
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      m_QuadricError[i][j] = 0;
    }
  }

  set<int>::iterator pos;
  for (pos = m_TriNeighbors.begin(); pos != m_TriNeighbors.end(); ++pos)
  {
    int triIndex = *pos;
    BasicTriangle& t = m.GetTri(triIndex);
    if (t.IsActive()) 
    {
      float triArea = 1;
      if (bUseTriArea)
      {
        triArea = t.CalcArea();
        m_QuadricTriArea += triArea;
      }

      const BasicVec3D normal = t.GetTriNormal();
      const float a = normal.GetX();
      const float b = normal.GetY();
      const float c = normal.GetZ();
      const float d = t.GetDParam();

      // NOTE: we could optimize this a bit by calculating values
      // like a * b and then using that twice (for _Q[0][1] and _Q[1][0]),
      // etc., since the matrix is symmetrical.  For now, I don't think
      // it's worth it.
      m_QuadricError[0][0] += triArea * a * a;
      m_QuadricError[0][1] += triArea * a * b;
      m_QuadricError[0][2] += triArea * a * c;
      m_QuadricError[0][3] += triArea * a * d;

      m_QuadricError[1][0] += triArea * b * a;
      m_QuadricError[1][1] += triArea * b * b;
      m_QuadricError[1][2] += triArea * b * c;
      m_QuadricError[1][3] += triArea * b * d;

      m_QuadricError[2][0] += triArea * c * a;
      m_QuadricError[2][1] += triArea * c * b;
      m_QuadricError[2][2] += triArea * c * c;
      m_QuadricError[2][3] += triArea * c * d;

      m_QuadricError[3][0] += triArea * d * a;
      m_QuadricError[3][1] += triArea * d * b;
      m_QuadricError[3][2] += triArea * d * c;
      m_QuadricError[3][3] += triArea * d * d;
    }
  }
}

void BasicVertex::GetQuadric(double Qret[4][4])
{ 
  for (int i = 0; i < 4; ++i)
  { 
    for (int j = 0; j < 4; ++j)
    {
      Qret[i][j] = m_QuadricError[i][j];
    }
  }
}

void BasicVertex::SetQuadric(double Qnew[4][4]) 
{ 
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      m_QuadricError[i][j] = Qnew[i][j];
    }
  }
}



// Go through the list of all neighboring vertices, and see how many
// triangles this vertex has in common w/ each neighboring vertex.  Normally
// there will be two triangles in common, but if there is only one, then this 
// vertex is on an edge.
bool BasicVertex::IsBorder(BasicMesh& m)
{
  set<int>::iterator pos, pos2;
  for (pos = GetVertNeighbors().begin(); pos != GetVertNeighbors().end(); ++pos)
  {
    int triCount = 0;

    BasicVertex& v = m.GetVertex(*pos);

    for (pos2 = v.GetTriNeighbors().begin(); pos2 != v.GetTriNeighbors().end(); ++pos2)
    {
      if (m.GetTri(*pos2).HasVertex(m_Index) )
      {
        ++triCount;
      }
    }

    if (1 == triCount)
    {
      return true;
    }
  }

  return false;
}

// Return all border info if the vertex is on an edge of the mesh.
void BasicVertex::GetAllBorderEdges(set<Border> &borderSet, BasicMesh& m)
{
  // Go through the list of all neighboring vertices, and see how many
  // triangles this vertex has in common w/ each neighboring vertex.  Normally
  // there will be two triangles in common, but if there is only one, then this 
  // vertex is on an edge.
  set<int>::iterator pos, pos2;

  for (pos = GetVertNeighbors().begin(); pos != GetVertNeighbors().end(); ++pos)
  {
    int triCount = 0;
    int triIndex = -1;
    BasicVertex& v = m.GetVertex(*pos);
    for (pos2 = v.GetTriNeighbors().begin(); pos2 != v.GetTriNeighbors().end(); ++pos2)
    {
      if (m.GetTri(*pos2).HasVertex(m_Index) )
      {
        ++triCount;
        triIndex = m.GetTri(*pos2).GetIndex();
      }
    }

    if (1 == triCount) // if only one triangle in common, it's an edge
    {
      // store the smaller index first
      Border b;
      b.triIndex = triIndex;
      if (m_Index < v.GetIndex())
      {
        b.vert1 = m_Index;
        b.vert2 = v.GetIndex();
      }
      else
      {
        b.vert1 = v.GetIndex();
        b.vert2 = m_Index;
      }
      borderSet.insert(b);
    }
  }
}

}
