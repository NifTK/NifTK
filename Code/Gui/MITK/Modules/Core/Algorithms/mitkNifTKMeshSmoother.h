/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkNifTKMeshSmoother_h
#define mitkNifTKMeshSmoother_h

#include <mitkLogMacros.h>

#include <iostream>
#include <vector>
#include "mitkBasicVertex.h"
#include "mitkBasicTriangle.h"
#include "mitkNifTKCMC33.h"

#include "niftkCoreExports.h"

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <iomanip>
using std::setiosflags;

#include <ios>
using std::ios_base;
using std::ios;

#include <set>
using std::set;

#include <vector>
using std::vector;

#include <limits>
using std::numeric_limits;

#include <cstring> // for memcpy()
#include <cctype>

namespace mitk 
{

  /**
  * \class ordered_size_t_pair
  * \brief Utility class for storing an ordered pair of vertex indices
  */

  class ordered_size_t_pair
  {
  public:
    ordered_size_t_pair(const size_t &a, const size_t &b)
    {
      if(a < b)
      {
        indices[0] = a;
        indices[1] = b;
      }
      else
      {
        indices[0] = b;
        indices[1] = a;
      }
    }

    bool operator<(const ordered_size_t_pair &right) const
    {
      if(indices[0] < right.indices[0])
        return true;
      else if(indices[0] > right.indices[0])
        return false;

      if(indices[1] < right.indices[1])
        return true;
      else if(indices[1] > right.indices[1])
        return false;

      return false;
    }

    size_t indices[2];
  };

  /**
  * \class ordered_indexed_edge
  * \brief Utility class for storing an ordered edge
  */
  class ordered_indexed_edge
  {
  public:
    ordered_indexed_edge(const BasicVertex &a, const BasicVertex &b)
    {
      if(a.GetIndex() < b.GetIndex())
      {
        indices[0] = a.GetIndex();
        indices[1] = b.GetIndex();
      }
      else
      {
        indices[0] = b.GetIndex();
        indices[1] = a.GetIndex();
      }

      centre_point.SetX((a.GetCoordX() + b.GetCoordX())*0.5f);
      centre_point.SetY((a.GetCoordY() + b.GetCoordY())*0.5f);
      centre_point.SetZ((a.GetCoordZ() + b.GetCoordZ())*0.5f);
    }

    bool operator<(const ordered_indexed_edge &right) const
    {
      if(indices[0] < right.indices[0])
        return true;
      else if(indices[0] > right.indices[0])
        return false;

      if(indices[1] < right.indices[1])
        return true;
      else if(indices[1] > right.indices[1])
        return false;

      return false;
    }

    size_t indices[2];
    BasicVec3D centre_point;
    size_t id;
  };

  
  /**
  * \class MeshSmoother
  * \brief This class implements various mesh smoothing algorithms and it can be used to 
  * (re)compute surface and vertex normals of a BasicMesh structure.
  */

  class NIFTKCORE_EXPORT MeshSmoother
  {
  public:
    /// \brief Default constructor
    MeshSmoother();
    /// \brief Destructor
    virtual ~MeshSmoother();
    
    /// \brief Clear all internally stored data
    void Clear(void);

    /// \brief Initialize the mesh smoother over a previously defined set of vertices and triangles
    void InitWithExternalData(MeshData * data);

    /// \brief Computes vertex and triangle normals over the whole mesh
    void GenerateVertexAndTriangleNormals(void);
    /// \brief Re-orient triangle faces (CCW to CW)
    void ReOrientFaces(void);

    /// \brief Utility function to load stl for testing and verification
    bool LoadFromBinarySTL(const char *const file_name, const bool generate_normals = true, const size_t buffer_width = 65536);

    /// \brief Determines the current maximum extent of the mesh and re-scales all coordinates according to the new value
    void SetMaxExtent(float max_extent);

    /// \brief Re-scales all coordinates according to the scale parameter
    void RescaleMesh(double scale_value);

    /// \brief Implements Taubin's mesh smoothing algorithm. See: Geometric Signal Processing on Polygonal Meshes by G. Taubin
    void TaubinSmooth(const float lambda, const float mu, const size_t steps);
    /// \brief Implements algorihtm for fixing cracks in a mesh (happens with standard marchin cubes)
    void FixCracks(void);

    /// \brief Select smoothing method. 0- Classic Laplacian, 1 - Inverse Edge Length, 2 - Curvature Normal
    inline void SetSmoothingMethod(int val) { m_SmoothingMethod = val;}

    /// \brief Set the flip normals flag
    inline void SetFlipNormals(bool val) { m_FlipNormals = val;}
    /// \brief Get the flip normals flag
    inline bool GetFlipNormals(void) { return m_FlipNormals; }

  private:
    /// \brief Implements "Laplacian" mesh smoothing algorithm
    void LaplaceSmooth(const float scale);
    /// \brief Implements "Curvature Normal" mesh smoothing algorithm
    void CurvatureNormalSmooth(const float scale);
    /// \brief Implements "Inverse Edge Length" mesh smoothing algorithm
    void InverseEdgeLengthSmooth(const float scale);

    /// \brief Compute vertex normals over the whole mesh
    void GenerateVertexNormals(void);
    /// \brief Compute triangle normals over the whole mesh
    void GenerateTriangleNormals(void);
    /// \brief Re-compute both vertex and triangle normals over the whole mesh
    void RegenerateVertexAndTriangleNormalsIfExists(void);

    /// \brief Eliminates duplicate in a vector. It is used in the merging algorithm
    template<typename T> void EliminateVectorDuplicates(std::vector<T> &v);
    /// \brief Merges a pair of vertices and re-assigns triangle membership
    bool MergeVertexPair(const size_t keeper, const size_t goner);

  private:
    int  m_SmoothingMethod; // Stores which smoothing method to use
    bool m_FlipNormals;     // Flag to indicate wether we need to flip the normals

    std::vector<BasicVec3D>   m_VertexNormals;   // stores all vertex normals
    std::vector<BasicVec3D>   m_TriangleNormals; // stores all triangle normals

    mitk::MeshData          * m_MeshDataExt;  // pointer to the externally created container
  };
} //endof mitk namespace

#endif // mitkNifTKMeshSmoother_h
