/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <assert.h>
#include <float.h>

#include <iostream>

#include "mitkBasicMesh.h"

namespace mitk
{

  BasicMesh::BasicMesh()
  {
    m_NumVerts = 0;
    m_NumTriangles = 0;
    m_VertList.clear();
    m_TriList.clear();
  }

  BasicMesh::BasicMesh(vector<BasicVertex> &vertices, vector<BasicTriangle> &triangles, int numOfVerts, int numOfTris)
  {
    m_NumVerts = 0;
    m_NumTriangles = 0;
    m_VertList.clear();
    m_TriList.clear();

    InitWithVertsAndTris(vertices, triangles, numOfVerts, numOfTris);
  }

  BasicMesh::BasicMesh(const BasicMesh& other)
  {
    m_NumVerts     = other.m_NumVerts;
    m_NumTriangles = other.m_NumTriangles;
    m_VertList     = other.m_VertList;
    m_TriList      = other.m_TriList;
    // NOTE: should reset tris in m_VertList, m_TriList
  }

  BasicMesh& BasicMesh::operator=(const BasicMesh& other)
  {
    if (this == &other) 
      return *this; // don't assign to self
    
    m_NumVerts     = other.m_NumVerts;
    m_NumTriangles = other.m_NumTriangles;
    m_VertList     = other.m_VertList;
    m_TriList      = other.m_TriList;
    
    // NOTE: should reset tris in m_VertList, m_TriList
    return *this;
  }

  BasicMesh::~BasicMesh()
  {
    m_NumVerts = 0;
    m_NumTriangles = 0;
    m_VertList.erase(m_VertList.begin(), m_VertList.end());
    m_TriList.erase(m_TriList.begin(), m_TriList.end());
  }

  void BasicMesh::InitWithVertsAndTris(
    vector<BasicVertex> &vertices
  , vector<BasicTriangle> &triangles
  , int numOfVerts
  , int numOfTris)
  {
    m_VertList.clear();
    m_VertList.resize(numOfVerts);
    m_NumVerts = numOfVerts;
      
    // read vertices
    for (int i = 0; i < numOfVerts; i++)
    {
      BasicVertex bv;
      bv = vertices.at(i);
      bv.SetIndex(i);
      bv.SetActive(true);
      m_VertList[i] = bv;
    }

    m_TriList.clear();
    m_TriList.resize(numOfTris);
    m_NumTriangles = numOfTris;

    for (int i = 0; i < numOfTris; i++)
    {
      int v1Idx = triangles.at(i).GetVert1Index();
      int v2Idx = triangles.at(i).GetVert2Index();
      int v3Idx = triangles.at(i).GetVert3Index();

      // make sure verts in correct range
      assert(v1Idx < numOfVerts && v2Idx < numOfVerts && v3Idx < numOfVerts);


      BasicTriangle bt(triangles.at(i));
      bt.ChangeMesh(this);
      bt.SetTriNormal(triangles.at(i).GetTriNormal());
      bt.SetIndex(i);
      //bt.CalcNormal();
      bt.CalcArea();
      bt.SetActive(true);
      m_TriList[i] = bt;

      m_VertList[v1Idx].AddTriNeighbor(i);
      m_VertList[v1Idx].AddVertNeighbor(v2Idx);
      m_VertList[v1Idx].AddVertNeighbor(v3Idx);

      m_VertList[v2Idx].AddTriNeighbor(i);
      m_VertList[v2Idx].AddVertNeighbor(v1Idx);
      m_VertList[v2Idx].AddVertNeighbor(v3Idx);

      m_VertList[v3Idx].AddTriNeighbor(i);
      m_VertList[v3Idx].AddVertNeighbor(v1Idx);
      m_VertList[v3Idx].AddVertNeighbor(v2Idx);

    }
    // Not sure if this is actually needed, disabling for now
    //CalcVertNormals();
  }



  // Recalculate the normal for one vertex
  void BasicMesh::CalcOneVertNormal(unsigned vert)
  {
    BasicVertex& v = GetVertex(vert);
    const set<int>& triset = v.GetTriNeighbors();

    set<int>::iterator iter;

    BasicVec3D vec;

    for (iter = triset.begin(); iter != triset.end(); ++iter)
    {
      // get the triangles for each vertex & add up the normals.
      vec += GetTri(*iter).GetTriNormal();
    }

    vec.Normalize(); // normalize the vertex	
    v.SetNormal(vec);
  }


  // Calculate the vertex normals after loading the mesh.
  void BasicMesh::CalcVertNormals()
  {
    // Iterate through the vertices
    for (unsigned i = 0; i < m_VertList.size(); ++i)
    {
      CalcOneVertNormal(i);
    }
  }


  // Used for debugging
  void BasicMesh::PrintStatus()
  {
    std::cout << "*** Mesh Dump ***" << std::endl;
    std::cout << "# of vertices: " << m_NumVerts << std::endl;
    std::cout << "# of triangles: " << m_NumTriangles << std::endl;
    for (unsigned i = 0; i < m_VertList.size(); ++i)
    {
      std::cout << "\tVertex " << i << ": " << m_VertList[i] << std::endl;
    }
    std::cout << std::endl;
    for (unsigned i = 0; i < m_TriList.size(); ++i)
    {
      std::cout << "\tTriangle " << i << ": " << m_TriList[i] << std::endl;
    }
    std::cout << "*** End of Mesh Dump ***" << std::endl;
    std::cout << std::endl;
  }

  // Get min, max values of all verts
  void BasicMesh::SetMinMax(float min[3], float max[3])
  {
    max[0] = max[1] = max[2] = -FLT_MAX;
    min[0] = min[1] = min[2] = FLT_MAX;

    for (unsigned int i = 0; i < m_VertList.size(); ++i)
    {
      const float* pVert = m_VertList[i].GetCoordArray();
      if (pVert[0] < min[0]) min[0] = pVert[0];
      if (pVert[1] < min[1]) min[1] = pVert[1];
      if (pVert[2] < min[2]) min[2] = pVert[2];
      if (pVert[0] > max[0]) max[0] = pVert[0];
      if (pVert[1] > max[1]) max[1] = pVert[1];
      if (pVert[2] > max[2]) max[2] = pVert[2];
    }
  }

  // Center mesh around origin.
  // Fit mesh in box from (-1, -1, -1) to (1, 1, 1)
  void BasicMesh::Normalize()  
  {
    float min[3], max[3], Scale;

    SetMinMax(min, max);

    BasicVec3D minv(min);
    BasicVec3D maxv(max);

    BasicVec3D dimv = maxv - minv;

    if (dimv.GetX() >= dimv.GetY() && dimv.GetX() >= dimv.GetZ()) 
    {
      Scale = 2.0f/dimv.GetX();
    }
    else if (dimv.GetY() >= dimv.GetX() && dimv.GetY() >= dimv.GetZ())
    {
      Scale = 2.0f/dimv.GetY();
    }
    else
    {
      Scale = 2.0f/dimv.GetZ();
    }

    BasicVec3D transv = minv + maxv;

    transv *= 0.5f;

    for (unsigned int i = 0; i < m_VertList.size(); ++i)
    {
      m_VertList[i].SetCoords(m_VertList[i].GetCoords() - transv);
      m_VertList[i].SetCoords(m_VertList[i].GetCoords() * Scale);
    }
  }

} // end of namespace
