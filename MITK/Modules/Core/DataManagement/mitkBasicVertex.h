/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkBasicVertex_h
#define __mitkBasicVertex_h

using namespace std;

#include <vector>
#include <set>

#include "mitkBasicVec3D.h"
#include "niftkCoreExports.h"

namespace mitk
{

class BasicMesh;

// Used to store an edge -- two vertices which have only one
// triangle in common form an edge of the mesh.
struct Border
{ 
  int vert1;
  int vert2;
  int triIndex;

  // We need operator< because it's used by the STL's set<> to determine equality
  // (if (not a<b) and (not b>a) then a is equal to b)
  bool operator <(const Border& b) const 
  { 
    int v1, v2, b1, b2;

    // make sure the smaller vert index is always first.
    if (vert1 < vert2)
    { 
      v1 = vert1; v2 = vert2;
    } 
    else 
    { 
      v1 = vert2; v2 = vert1;
    }

    if (b.vert1 < b.vert2)
    { 
      b1 = b.vert1; b2 = b.vert2;
    } 
    else 
    { 
      b1 = b.vert2; b2 = b.vert1;
    }

    if (v1 < b1) 
      return true;

    if (v1 > b1) 
      return false;

    return (v2 < b2); // v1 == b1
  }
};

/**
* \class BasicVertex
* \brief Simple vertex implementation that is used in the Surface Extraction 
* and surface smoothing and decimation algorithms. It has both a position and a normal.
*/

class NIFTKCORE_EXPORT BasicVertex
{ 

public:
  /// \brief Default constructor
  BasicVertex();
  /// \brief Constructor with coordinates as parameters
  BasicVertex(float x1, float y1, float z1);
  /// \brief Constructor with coordinates as parameters in an array
  BasicVertex(float av[3]);
  /// \brief Constructor with coordinates and normals as parameters in arrays
  BasicVertex(float av[3], float vn[3]);

  /// \brief Copy constructor
  BasicVertex(const BasicVertex& v);

  /// \brief destructor
  virtual ~BasicVertex();

  /// \brief Assignment operator
  BasicVertex& operator=(const BasicVertex& v);
  /// \brief Assignment operator
  BasicVertex& operator=(const float av[3]);

  /// \brief Comparison operator
  bool operator==(const BasicVertex& v);
  /// \brief Comparison operator - not equal
  bool operator!=(const BasicVertex& v);

  /// \brief operator< is used to order vertices by edge removal costs
  bool operator<(const BasicVertex &right) const;
  /// \brief operator> is used to order vertices by edge removal costs
  bool operator>(const BasicVertex &right) const;

  /// \brief Output to std::stream
  friend std::ostream& operator<<(std::ostream& , const BasicVertex& );

  /// \brief Set coordinates as 3D vector
  void                     SetCoords(const BasicVec3D& v) { m_Coords = v; };
  /// \brief Get coordinates as 3D vector
  inline       BasicVec3D& GetCoords()                    { return m_Coords; }
  /// \brief Set coordinates as 3D vector - const
  inline const BasicVec3D& GetCoords() const              { return m_Coords; }
  /// \brief Get the whole coordinate array as a float[3] array
  const float* GetCoordArray() const;

  /// \brief Set X coordinate of the vertex individually
  void SetCoordX(float x) { m_Coords.SetX(x); }
  /// \brief Set Y coordinate of the vertex individually
  void SetCoordY(float y) { m_Coords.SetY(y); }
  /// \brief Set Z coordinate of the vertex individually
  void SetCoordZ(float z) { m_Coords.SetZ(z); }

  /// \brief Get X coordinate of the vertex
  float GetCoordX() { return m_Coords.GetX(); }
  /// \brief Get Y coordinate of the vertex
  float GetCoordY() { return m_Coords.GetY(); }
  /// \brief Get Z coordinate of the vertex
  float GetCoordZ() { return m_Coords.GetZ(); }

  /// \brief Get X coordinate of the vertex - const
  const float GetCoordX() const { return m_Coords.GetX(); }
  /// \brief Get Y coordinate of the vertex - const
  const float GetCoordY() const { return m_Coords.GetY(); }
  /// \brief Get Z coordinate of the vertex - const
  const float GetCoordZ() const { return m_Coords.GetZ(); }

  /// \brief Set normal as 3D vector
  void SetNormal(const BasicVec3D& vn)       { m_Normal = vn; };
  /// \brief Get normal as 3D vector
  inline       BasicVec3D& GetNormal()       { return m_Normal; };
  /// \brief Get normal as 3D vector - const
  inline const BasicVec3D& GetNormal() const { return m_Normal; };
  /// \brief Get the whole normal array as a float[3] array
  const float* GetNormalArray() const;

  /// \brief Set X coordinate of the vertex normal individually
  void SetNormalX(float x) { m_Normal.SetX(x); }
  /// \brief Set Y coordinate of the vertex normal individually
  void SetNormalY(float y) { m_Normal.SetY(y); }
  /// \brief Set Z coordinate of the vertex normal individually
  void SetNormalZ(float z) { m_Normal.SetZ(z); }

  /// \brief Get X coordinate of the vertex normal
  float GetNormalX() { return m_Normal.GetX(); }
  /// \brief Get Y coordinate of the vertex normal
  float GetNormalY() { return m_Normal.GetY(); }
  /// \brief Get Z coordinate of the vertex normal
  float GetNormalZ() { return m_Normal.GetZ(); }

  /// \brief Get X coordinate of the vertex normal - const
  const float GetNormalX() const { return m_Normal.GetX(); }
  /// \brief Get Y coordinate of the vertex normal - const
  const float GetNormalY() const { return m_Normal.GetY(); }
  /// \brief Get Z coordinate of the vertex normal - const
  const float GetNormalZ() const { return m_Normal.GetZ(); }

  /// \brief Add neighbor - a vertex that is connected by an edge
  void AddVertNeighbor(int v) { m_VertNeighbors.insert(v); }

  /// \brief Remove a vertex which is no longer connected by an edge
  unsigned RemoveVertNeighbor(int v) { return m_VertNeighbors.erase(v); }

  /// \brief Add a triangle neighbor - a triangle which uses this vertex
  void AddTriNeighbor(int t) { m_TriNeighbors.insert(t); }

  /// \brief Remove triangle if it no longer uses this vertex
  unsigned RemoveTriNeighbor(int t) { return m_TriNeighbors.erase(t); }

  /// \brief Get the whole set of vertex neighbours - const
  const set<int>& GetVertNeighbors() const { return m_VertNeighbors; }
  // \brief Get the whole set of vertex neighbours
  set<int>&       GetVertNeighbors()       { return m_VertNeighbors; }
  
  /// \brief Get the whole set of triangle neighbours - const
  const set<int>& GetTriNeighbors() const  { return m_TriNeighbors; }
  /// \brief Get the whole set of triangle neighbours
  set<int>&       GetTriNeighbors()        { return m_TriNeighbors; }

  /// \brief Check is vertex has a certain vertex neighbour
  bool HasVertNeighbor(int v) const { return (m_VertNeighbors.find(v) != m_VertNeighbors.end()); }
  /// \brief Check is vertex has a certain triangle neighbour
  bool HasTriNeighbor(int t)  const { return (m_TriNeighbors.find(t) != m_TriNeighbors.end()); }

  /// \brief Returns the edge remove costs that is used in mesh simplification
  double GetEdgeRemoveCost()         { return m_Cost; };
  /// \brief Sets the edge remove costs that is used in mesh simplification
  void   SetEdgeRemoveCost(double f) { m_Cost = f; };

  /// \brief Returns the index of the vertex that is at other end of the min. cost edge (used in mesh simplification)
  int  GetMinCostEdgeVert() const { return m_MinCostNeighbor; };
  /// \brief Sets the index of the vertex that is at other end of the min. cost edge (used in mesh simplification)
  void SetMinCostEdgeVert(int i)  { m_MinCostNeighbor = i; }

  /// \brief Returns the cost of removing this vertex from the mesh
  double GetCost() const { return m_Cost; }

  /// \brief Get index of the current vertex
  int  GetIndex() const { return m_Index; }
  /// \brief Set index of the current vertex
  void SetIndex(int i)  { m_Index = i; }

  /// \brief Returns the flag value that indicates if the vertex is active (true) or it was removed (false) from the mesh
  inline bool IsActive()  const { return m_Active; }
  /// \brief Sets a flag the flag that indicates if the vertex is active (true) or it was removed (false) from the mesh
  inline void SetActive(bool b) { m_Active = b; }

  /// \brief Checks if the current vertex is on the border of the mesh
  /// (i.e. is there an edge which uses this vertex which 
  /// is only used for one triangle?)
  bool IsBorder(BasicMesh& m); 
  
  /// \brief Gets the set of border edges
  /// Is the current vertex on an edge?  If so, get edge information.
  /// This is used to put constraints on the border so that the mesh
  /// edges aren't "eaten away" by the mesh simplification.
  void GetAllBorderEdges(set<Border> &borderSet, BasicMesh& m);

  ///\brief Evaluate the quadric cost function for the current vertex
  /// Used for Garland & Heckbert's quadric edge collapse cost (used for mesh simplifications/progressive meshes)
  void CalcQuadric(BasicMesh& m, bool bUseTriArea); // calculate the 4x4 Quadric matrix
  ///\brief Returns the quadric cost function for the current vertex
  void GetQuadric(double Qret[4][4]);
  ///\brief Sets the quadric cost function for the current vertex
  void SetQuadric(double Qnew[4][4]);

  ///\brief Gets the quadric cost function summed over the triangle area
  double GetQuadricSummedTriArea() { return m_QuadricTriArea; };
  ///\brief Sets the quadric cost function summed over the triangle area
  void   SetQuadricSummedTriArea(double newArea) { m_QuadricTriArea = newArea; };

private:
  /// \brief Initializes the quadric cost evaluation
  /// Used for Garland & Heckbert's quadric edge collapse cost (used for mesh simplifications/progressive meshes)
  void InitQuadric();

private:
  BasicVec3D m_Coords;         // X, Y, Z position of this vertex
  BasicVec3D m_Normal;         // vertex normal, used for Gouraud shading

  set<int>   m_VertNeighbors;   // connected to this vertex via an edge
  set<int>   m_TriNeighbors;       // triangles of which this vertex is a part

  bool       m_Active;             // false if vertex has been removed

  double     m_Cost;               // cost of removing this vertex from Progressive Mesh
  int        m_MinCostNeighbor;    // index of vertex at other end of the min. cost edge

  int        m_Index;
  double     m_QuadricError[4][4]; // Used for Quadric error cost.
  double     m_QuadricTriArea;     // summed area of triangles used to compute quadrics
  
  mutable float m_V[3];            //mutable floats used for returning values
  mutable float m_VN[3];           //mutable floats used for returning values
};

} //end of namespace

#endif // #ifndef __BasicVertex_h