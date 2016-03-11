/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkBasicTriangle_h
#define __mitkBasicTriangle_h

#include <assert.h>
#include <iostream>
#include "niftkCoreExports.h"
#include "mitkBasicVec3D.h"

namespace mitk
{

class BasicVertex;
class BasicMesh;

/**
* \class BasicTriangle
* \brief Simple triangle implementation that is used in the Surface Extraction 
* and surface smoothing and decimation algorithms. It keeps hold of vertices,
* triangle index and has an active flag.
*/

class NIFTKCORE_EXPORT BasicTriangle
{
public:
  /// \brief Default constructor
  BasicTriangle();
  /// \brief Constructor with vertex indices as parameters
  BasicTriangle(int v1, int v2, int v3);
  /// \brief Constructor with mesh pointer and vertex indices as parameters
  BasicTriangle(mitk::BasicMesh* mp, int v1, int v2, int v3);

  /// \brief Destructors
  virtual ~BasicTriangle();

  /// \brief Copy constructor
  BasicTriangle(const BasicTriangle& t);

  /// \brief Assignment operator
  BasicTriangle& operator=(const BasicTriangle& t);
  
  /// \brief Comparison operator. It assumes pointing to same list of verts.
  bool operator==(const BasicTriangle& t);

  /// \brief Output to file or stream
  friend std::ostream& operator<<(std::ostream& os, const BasicTriangle& to);

  /// \brief Changes the mesh pointer to the specified one.
  void ChangeMesh(BasicMesh* mp) { m_Mesh = mp; }

  /// \brief  Returns true if the triangle active, false if it was removed from the mesh
  bool IsActive() const { return m_Active; }
  /// \brief  Sets the active status of the triangle. False = triangle was removed
  void SetActive(bool b) { m_Active = b; }
  /// \brief  Returns true if the triangle has a specific vertex (checked by index)
  bool HasVertex(int vi);

  /// \brief Changes a member vertex to another. When we collapse an edge,
  ///  we need to change the vertex of a BasicTriangle.
  void ChangeVertex(int vFrom, int vTo);

  /// \brief Gets the currently stored vertex indices
  void GetVertIndices(int& v1, int& v2, int& v3);

  /// \brief Gets the coordinates of the first member vertex as a float array
  const float* GetVert1CoordArray();
  /// \brief Gets the coordinates of the second member vertex as a float array
  const float* GetVert2CoordArray();
  /// \brief Gets the coordinates of the third member vertex as a float array
  const float* GetVert3CoordArray();

  /// \brief Gets the const reference to the coordinates (Vec3D) of the first member vertex
  const BasicVec3D& GetVert1Coords() const;
  /// \brief Gets the const reference to the coordinates (Vec3D) of the second member vertex
  const BasicVec3D& GetVert2Coords() const;
  /// \brief Gets the const reference to the coordinates (Vec3D) of the third member vertex
  const BasicVec3D& GetVert3Coords() const;
  
  /// \brief Gets the const reference to the first member vertex
  const BasicVertex& GetVert1() const;
  /// \brief Gets the const reference to the second member vertex
  const BasicVertex& GetVert2() const;
  /// \brief Gets the const reference to the third member vertex
  const BasicVertex& GetVert3() const;

  /// \brief Gets the coordinates of the first member vertex's normal as a float array
  const float* GetVert1NormalArray();
  /// \brief Gets the coordinates of the second member vertex's normal as a float array
  const float* GetVert2NormalArray();
  /// \brief Gets the coordinates of the third member vertex's normal as a float array
  const float* GetVert3NormalArray();

  /// \brief Gets the const reference to the coordinates (Vec3D) of the first member vertex's normal
  const BasicVec3D& GetVert1Normal() const;
  /// \brief Gets the const reference to the coordinates (Vec3D) of the second member vertex's normal
  const BasicVec3D& GetVert2Normal() const;
  /// \brief Gets the const reference to the coordinates (Vec3D) of the third member vertex's normal
  const BasicVec3D& GetVert3Normal() const;

  //****************************************
  /// \brief Sets the triangle normal (as Vec3D)
  void SetTriNormal(BasicVec3D triNorm) { m_TriNormal = triNorm; }
  /// \brief Gets const refernece to the triangle normal (as Vec3D)
  const BasicVec3D & GetTriNormal() const { return m_TriNormal; }
  /// \brief Gets the coordinates of the triangle normal as a float[3] array
  float* GetTriNormalArray();
  
  /// \brief Gets the X component of the triangle normal
  float GetTriNormalX() { return m_TriNormal.GetX(); }
  /// \brief Gets the Y component of the triangle normal
  float GetTriNormalY() { return m_TriNormal.GetY(); }
  /// \brief Gets the Z component of the triangle normal
  float GetTriNormalZ() { return m_TriNormal.GetZ(); }

  /// \brief Sets the X component of the triangle normal
  void SetTriNormalX(float nx) { m_TriNormal.SetX(nx); }
  /// \brief Sets the Y component of the triangle normal
  void SetTriNormalY(float ny) { m_TriNormal.SetY(ny); }
  /// \brief Sets the Z component of the triangle normal
  void SetTriNormalZ(float nz) { m_TriNormal.SetZ(nz); }
  
  /// \brief Re-computes the normal for the BasicTriangle
  void CalcNormal();
   
  /// \brief Returns area of BasicTriangle
  float CalcArea();

  /// \brief Returns the index of the member vertex with the specifed index (0-2)
  int GetVertIndex(int which);
  /// \brief Sets the index of the member vertex with the specifed index (0-2)
  void SetVertIndex(int which, int indexVal);

  /// \brief Returns the index of the first member vertex
  int GetVert1Index() const { return m_Vert1; }
  /// \brief Returns the index of the second member vertex
  int GetVert2Index() const { return m_Vert2; }
  /// \brief Returns the index of the third member vertex
  int GetVert3Index() const { return m_Vert3; }

  /// \brief Sets the index of the first member vertex
  void SetVert1Index(int v1) { m_Vert1 = v1; }
  /// \brief Sets the index of the second member vertex
  void SetVert2Index(int v2) { m_Vert2 = v2; }
  /// \brief Sets the index of the third member vertex
  void SetVert3Index(int v3) { m_Vert3 = v3; }

  /// \brief Returns the index of the current triangle
  int GetIndex() const { return m_Index; }
  /// \brief Sets the index of the current triangle
  void SetIndex(int i) { m_Index = i; }

  /// \brief Returns the'd' is from the plane equation ax + by + cz + d = 0
  float GetDParam() const { return m_DParam; }
  /// \brief Sets the'd' parameter of the plane equation ax + by + cz + d = 0
  void  SetDParam(float d) { m_DParam = d; }

protected:

  int m_Vert1; // index of the first member vertex
  int m_Vert2; // index of the second member vertex
  int m_Vert3; // index of the third member vertex

  BasicVec3D    m_TriNormal;         // normal to plane
  mutable float m_TriNormalArray[3]; //used for returning values 

  // This parameter is the "d" in the
  // plane equation ax + by + cz + d = 0
  // The plane equation of this BasicTriangle is used
  float       m_DParam;
  bool        m_Active; // active flag
  BasicMesh * m_Mesh;   // pointer to the mesh structure that holds the current triangle
  int         m_Index;  // index in list of BasicTriangles w/in mesh
 };

 } // end of namespace

#endif //__mitkBasicTriangle_h
