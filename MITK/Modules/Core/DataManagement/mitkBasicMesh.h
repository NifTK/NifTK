/*=============================================================================

NifTK: A software platform for medical image computing.

Copyright (c) University College London (UCL). All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.

See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkBasicMesh_h
#define __mitkBasicMesh_h

using namespace std;

#include <vector>
#include "niftkCoreExports.h"
#include "mitkBasicVertex.h"
#include "mitkBasicTriangle.h"

namespace mitk
{

  /**
* \class BasicMesh
* \brief Simple mesh implementation that is used in the Surface Extraction 
* and surface smoothing and decimation algorithms. This stores a list of vertices
* and another list of triangles (which references the vertex list)
*/

class  NIFTKCORE_EXPORT BasicMesh
{
public:
  /// \brief Default constructor
  BasicMesh();
  /// \brief Constructor with vertices and triangles as parameters
  BasicMesh(vector<BasicVertex> &vertices, vector<BasicTriangle> &triangles, int numOfVerts, int numOfTris);
  /// \brief Destructor
  virtual ~BasicMesh();

  /// \brief Initialize the mesh with an externally defined set of triangles and vertices
  void InitWithVertsAndTris(vector<BasicVertex> &vertices, vector<BasicTriangle> &triangles, int numOfVerts, int numOfTris);

  /// \brief Copy constructor
  BasicMesh(const BasicMesh&);
  /// \brief Assignment operator
  BasicMesh& operator=(const BasicMesh&);

  /// \brief Get reference to the vertex with the specified index
  BasicVertex&         GetVertex(int index)       { return m_VertList[index]; }
  /// \brief Get const reference to the vertex with the specified index
  const BasicVertex&   GetVertex(int index) const { return m_VertList[index]; }
  /// \brief Get reference to the triangle with the specified index
  BasicTriangle&       GetTri(int index)          { return m_TriList[index]; }
  /// \brief Get const reference to the triangle with the specified index
  const BasicTriangle& GetTri(int index)    const { return m_TriList[index]; }

  /// \brief Get number of vertices
  int  GetNumVerts()          { return m_NumVerts; }
  /// \brief Set number of vertices
  void SetNumVerts(int n)     { m_NumVerts = n; }
  /// \brief Get number of triangles
  int  GetNumTriangles()      { return m_NumTriangles; }
  /// \brief Set number of triangles
  void SetNumTriangles(int n) { m_NumTriangles = n; }

  /// \brief Normalize the mesh: center the mesh around the origin and shrink to fit in [-1, 1]
  void Normalize();

  /// \brief Compute normal for the vertex with the specified index
  void CalcOneVertNormal(unsigned vert);

  /// \brief Print mesh state to cout
  void PrintStatus();

private:
  /// \brief Comparison operator
  bool operator==(const BasicMesh&);

  /// \brief Set bounding box dimensions for mesh
  void SetMinMax(float min[3], float max[3]);

  /// \brief Calculate the vertex normals after loading the mesh
  void CalcVertNormals();

private:
  vector<BasicVertex>   m_VertList; // list of vertices in mesh
  vector<BasicTriangle> m_TriList;  // list of triangles in mesh

  int m_NumVerts;
  int m_NumTriangles;
};

} // end of namespace

#endif // __mitkBasicMesh_h
