/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkNifTKMeshSmoother.h"
#include <algorithm>
#include <unordered_set>

namespace std
{
  template <>
  struct hash<mitk::BasicVertex>
  {
      size_t operator()(mitk::BasicVertex const & x) const
      {
        return ((51 + std::hash<int>()(x.GetCoordX() )) * 51 + std::hash<int>()(x.GetCoordY())) * 51 + std::hash<int>()(x.GetCoordZ() );
      }
  };

  bool operator==(mitk::BasicVertex const & x, mitk::BasicVertex const & y)
  {
    return (x.GetCoordX() == y.GetCoordX() && x.GetCoordY() == y.GetCoordY() && x.GetCoordZ() == y.GetCoordZ());
  }
}


namespace mitk {

struct vec_equal : std::unary_function<mitk::BasicVertex, bool> 
{
  mitk::BasicVertex m_Vert;
  vec_equal(mitk::BasicVertex v):m_Vert(v) {}
  
  bool operator() (mitk::BasicVertex const& otherVert) const 
  {
    bool vertCoordsEqu = (m_Vert.GetCoordX() == otherVert.GetCoordX() && 
                          m_Vert.GetCoordY() == otherVert.GetCoordY() &&
                          m_Vert.GetCoordZ() == otherVert.GetCoordZ());

    bool vertNormsEqu  = (m_Vert.GetNormalX() == otherVert.GetNormalX() && 
                          m_Vert.GetNormalY() == otherVert.GetNormalY() &&
                          m_Vert.GetNormalZ() == otherVert.GetNormalZ());

    return (vertCoordsEqu && vertNormsEqu);
  }
};




MeshSmoother::MeshSmoother()
{
  m_SmoothingMethod = 0;
  m_FlipNormals     = false;
  m_MeshDataExt = 0;
}

MeshSmoother::~MeshSmoother()
{
}


void MeshSmoother::InitWithExternalData(MeshData * data)
{
  if (data != 0)
    m_MeshDataExt = data;
  else
  {
    MITK_ERROR <<"Invalid data pointer!";
    return;
  }
}

void MeshSmoother::Clear(void)
{
  m_VertexNormals.clear();
  m_TriangleNormals.clear();
}

bool MeshSmoother::LoadFromBinarySTL(const char *const file_name, const bool generate_normals, const size_t buffer_width)
{
  // Sanity check
  if (m_MeshDataExt == 0)
  {
    MITK_ERROR <<"Invalid data pointer, MeshSmoother wasn't initialized properly!";
    return false;
  }

  Clear();

  MITK_INFO << "Reading file: " << file_name << endl;

  ifstream in(file_name, ios_base::binary);

  if(in.fail())
    return false;

  const size_t header_size = 80;
  vector<char> buffer(header_size, 0);
  unsigned int num_triangles = 0; // Must be 4-byte unsigned int.

  // Read header.
  in.read(reinterpret_cast<char *>(&(buffer[0])), header_size);

  if(header_size != in.gcount())
    return false;

  if( 's' == tolower(buffer[0]) &&
    'o' == tolower(buffer[1]) && 
    'l' == tolower(buffer[2]) && 
    'i' == tolower(buffer[3]) && 
    'd' == tolower(buffer[4]) )
  {
    MITK_INFO << "Encountered ASCII STL file header -- aborting." << endl;
    return false;
  }

  // Read number of triangles.
  in.read(reinterpret_cast<char *>(&num_triangles), sizeof(unsigned int));

  if(sizeof(unsigned int) != in.gcount())
    return false;

  m_MeshDataExt->m_Triangles.resize(num_triangles);

  MITK_INFO << "Triangles:    " << m_MeshDataExt->m_Triangles.size() << endl;

  // Enough bytes for twelve 4-byte floats plus one 2-byte integer, per triangle.
  const size_t per_triangle_data_size = (12*sizeof(float) + sizeof(short unsigned int));
  const size_t buffer_size = per_triangle_data_size * buffer_width;
  buffer.resize(buffer_size, 0);

  size_t num_triangles_remaining = m_MeshDataExt->m_Triangles.size();
  size_t tri_index = 0;
  set<BasicVertex> vertex_set;

  while(num_triangles_remaining > 0)
  {
    size_t num_triangles_to_read = buffer_width;

    if(num_triangles_remaining < buffer_width)
      num_triangles_to_read = num_triangles_remaining;

    size_t data_size = per_triangle_data_size*num_triangles_to_read;

    in.read(reinterpret_cast<char *>(&buffer[0]), data_size);

    if(data_size != in.gcount() || in.fail())
      return false;

    num_triangles_remaining -= num_triangles_to_read;

    // Use a pointer to assist with the copying.
    // Should probably use std::copy() instead, but memcpy() does the trick, so whatever...
    char *cp = &buffer[0];

    for(size_t i = 0; i < num_triangles_to_read; i++)
    {
      // Skip face normal. We will calculate them manually later.
      cp += 3*sizeof(float);

      // For each of the three vertices in the triangle.
      for(short unsigned int j = 0; j < 3; j++)
      {
        BasicVertex v;

        // Get vertex components.
        float tmp = v.GetCoordX();
        memcpy(&tmp, cp, sizeof(float)); 
        cp += sizeof(float);
        
        tmp = v.GetCoordY();
        memcpy(&tmp, cp, sizeof(float));
        cp += sizeof(float);

        tmp = v.GetCoordZ();
        memcpy(&tmp, cp, sizeof(float));
        cp += sizeof(float);

        // Look for vertex in set.
        set<BasicVertex>::const_iterator find_iter = vertex_set.find(v);

        // If vertex not found in set...
        if(vertex_set.end() == find_iter)
        {
          // Assign new vertices index
          v.SetIndex(m_MeshDataExt->m_Vertices.size());

          // Add vertex to set
          vertex_set.insert(v);

          // Add vertex to vector
          BasicVertex indexless_vertex;
          indexless_vertex.SetCoordX(v.GetCoordX());
          indexless_vertex.SetCoordY(v.GetCoordY());
          indexless_vertex.SetCoordZ(v.GetCoordZ());
          m_MeshDataExt->m_Vertices.push_back(indexless_vertex);

          // Assign vertex index to triangle
          m_MeshDataExt->m_Triangles[tri_index].SetVertIndex(j, v.GetIndex());

          // Add triangle index to vertex
          vector<size_t> tri_indices;
          tri_indices.push_back(tri_index);
          m_MeshDataExt->m_VertexToTriangleIndices.push_back(tri_indices);
        }
        else
        {
          // Assign existing vertex index to triangle
          m_MeshDataExt->m_Triangles[tri_index].SetVertIndex(j, find_iter->GetIndex());

          // Add triangle index to vertex
          m_MeshDataExt->m_VertexToTriangleIndices[find_iter->GetIndex()].push_back(tri_index);
        }
      }

      // Skip attribute.
      cp += sizeof(short unsigned int);

      tri_index++;
    }
  }

  m_MeshDataExt->m_VertexToVertexIndices.resize(m_MeshDataExt->m_Vertices.size());

  for(size_t i = 0; i < m_MeshDataExt->m_VertexToTriangleIndices.size(); i++)
  {
    // Use a temporary set to avoid duplicates.
    set<size_t> m_VertexToVertexIndices_set;

    for(size_t j = 0; j < m_MeshDataExt->m_VertexToTriangleIndices[i].size(); j++)
    {
      size_t tri_index = m_MeshDataExt->m_VertexToTriangleIndices[i][j];

      for(size_t k = 0; k < 3; k++)
        if(i != m_MeshDataExt->m_Triangles[tri_index].GetVertIndex(k)) // Don't add current vertex index to its own adjacency list.
          m_VertexToVertexIndices_set.insert(m_MeshDataExt->m_Triangles[tri_index].GetVertIndex(k));
    }

    // Copy to final vector.
    for(set<size_t>::const_iterator ci = m_VertexToVertexIndices_set.begin(); ci != m_VertexToVertexIndices_set.end(); ci++)
      m_MeshDataExt->m_VertexToVertexIndices[i].push_back(*ci);
  }

  MITK_INFO << "Vertices:     " << m_MeshDataExt->m_Triangles.size()*3 << " (of which " << m_MeshDataExt->m_Vertices.size() << " are unique)" << endl;

  in.close();

  //if(true == generate_normals)
  //{
  //  MITK_INFO << "Generating normals" << endl;
  //  GenerateVertexAndTriangleNormals();
  //}

  return true;
} 

// This produces results that are practically identical to Meshlab
void MeshSmoother::LaplaceSmooth(const float scale)
{
  vector<BasicVec3D> displacements(m_MeshDataExt->m_Vertices.size(), BasicVec3D(0, 0, 0));

  // Get per-vertex displacement.
  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
  {
    // Skip rogue vertices (which were probably made rogue during a previous
    // attempt to fix mesh cracks).
    if(0 ==  m_MeshDataExt->m_VertexToVertexIndices[i].size())
      continue;

    const float weight = 1.0f / static_cast<float>( m_MeshDataExt->m_VertexToVertexIndices[i].size());

    for(size_t j = 0; j <  m_MeshDataExt->m_VertexToVertexIndices[i].size(); j++)
    {
      size_t neighbour_j =  m_MeshDataExt->m_VertexToVertexIndices[i][j];
      displacements[i] += (m_MeshDataExt->m_Vertices[neighbour_j].GetCoords() - m_MeshDataExt->m_Vertices[i].GetCoords())*weight;
    }
  }

  // Apply per-vertex displacement.
  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
    m_MeshDataExt->m_Vertices[i].SetCoords(m_MeshDataExt->m_Vertices[i].GetCoords() + displacements[i]*scale);
}

void MeshSmoother::TaubinSmooth(const float lambda, const float mu, const size_t steps)
{
  // Sanity check
  if (m_MeshDataExt == 0)
  {
    MITK_ERROR <<"Invalid data pointer, MeshSmoother wasn't initialized properly!";
    return;
  }

  switch (m_SmoothingMethod)
  {
    case 0:
      for(size_t s = 0; s < steps; s++)
      {
        LaplaceSmooth(lambda);
        LaplaceSmooth(mu);
      }
      break;

    case 1:
      for(size_t s = 0; s < steps; s++)
      {
        CurvatureNormalSmooth(lambda);
        CurvatureNormalSmooth(mu);
      }
      break;

    case 2:
      for(size_t s = 0; s < steps; s++)
      {
        InverseEdgeLengthSmooth(lambda);
        InverseEdgeLengthSmooth(mu);
      }
      break;
  }

  // Recalculate normals, if necessary.
  //RegenerateVertexAndTriangleNormalsIfExists();
}

void MeshSmoother::InverseEdgeLengthSmooth(const float scale)
{
  vector<BasicVec3D> displacements(m_MeshDataExt->m_Vertices.size(), BasicVec3D(0, 0, 0));

  // Get per-vertex displacement.
  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
  {
    // Skip rogue vertices (which were probably made rogue during a previous
    // attempt to fix mesh cracks).
    if(0 ==  m_MeshDataExt->m_VertexToVertexIndices[i].size())
      continue;

    vector<float> weights( m_MeshDataExt->m_VertexToVertexIndices[i].size(), 0.0f);

    // Calculate weights based on inverse edge lengths.
    for(size_t j = 0; j <  m_MeshDataExt->m_VertexToVertexIndices[i].size(); j++)
    {
      size_t neighbour_j =  m_MeshDataExt->m_VertexToVertexIndices[i][j];

      float edge_length = m_MeshDataExt->m_Vertices[i].GetCoords().Distance(m_MeshDataExt->m_Vertices[neighbour_j].GetCoords());

      if(0 == edge_length)
        edge_length = numeric_limits<float>::epsilon();

      weights[j] = 1.0f / edge_length;
    }

    // Normalize the weights so that they sum up to 1.
    float s = 0;

    for(size_t j = 0; j < weights.size(); j++)
      s += weights[j];

    if(0 == s)
      s = numeric_limits<float>::epsilon();

    for(size_t j = 0; j < weights.size(); j++)
      weights[j] /= s;

    // Sum the displacements.
    for(size_t j = 0; j < m_MeshDataExt->m_VertexToVertexIndices[i].size(); j++)
    {
      size_t neighbour_j = m_MeshDataExt->m_VertexToVertexIndices[i][j];
      displacements[i] += (m_MeshDataExt->m_Vertices[neighbour_j].GetCoords() - m_MeshDataExt->m_Vertices[i].GetCoords())*weights[j];
    }
  }

  // Apply per-vertex displacement.
  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
    m_MeshDataExt->m_Vertices[i].SetCoords(m_MeshDataExt->m_Vertices[i].GetCoords() + displacements[i]*scale);
}

void MeshSmoother::CurvatureNormalSmooth(const float scale)
{
  vector<BasicVec3D> displacements(m_MeshDataExt->m_Vertices.size(), BasicVec3D(0, 0, 0));

  // Get per-vertex displacement.
  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
  {
    if(0 == m_MeshDataExt->m_VertexToVertexIndices[i].size())
      continue;

    vector<float> weights(m_MeshDataExt->m_VertexToVertexIndices[i].size(), 0.0f);

    size_t angle_error = 0;

    // For each vertex pair (ie. each edge),
    // calculate weight based on the two opposing angles (ie. curvature normal scheme).
    for(size_t j = 0; j < m_MeshDataExt->m_VertexToVertexIndices[i].size(); j++)
    {
      size_t angle_count = 0;

      size_t neighbour_j = m_MeshDataExt->m_VertexToVertexIndices[i][j];

      // Find out which two triangles are shared by the edge.
      for(size_t k = 0; k < m_MeshDataExt->m_VertexToTriangleIndices[i].size(); k++)
      {
        for(size_t l = 0; l < m_MeshDataExt->m_VertexToTriangleIndices[neighbour_j].size(); l++)
        {
          size_t tri0_index = m_MeshDataExt->m_VertexToTriangleIndices[i][k];
          size_t tri1_index = m_MeshDataExt->m_VertexToTriangleIndices[neighbour_j][l];

          // This will occur twice per edge.
          if(tri0_index == tri1_index)
          {
            // Find the third vertex in this triangle (the vertex that doesn't belong to the edge).
            for(size_t m = 0; m < 3; m++)
            {
              // This will occur once per triangle.
              if(m_MeshDataExt->m_Triangles[tri0_index].GetVertIndex(m) != i && m_MeshDataExt->m_Triangles[tri0_index].GetVertIndex(m) != neighbour_j)
              {
                size_t opp_vert_index = m_MeshDataExt->m_Triangles[tri0_index].GetVertIndex(m);

                // Get the angle opposite of the edge.
                BasicVec3D a = m_MeshDataExt->m_Vertices[i].GetCoords() - m_MeshDataExt->m_Vertices[opp_vert_index].GetCoords();
                BasicVec3D b = m_MeshDataExt->m_Vertices[neighbour_j].GetCoords() - m_MeshDataExt->m_Vertices[opp_vert_index].GetCoords();
                a.Normalize();
                b.Normalize();

                float dotProd = a.Dot(b);

                if(-1 > dotProd)
                  dotProd = -1;
                else if(1 < dotProd)
                  dotProd = 1;

                float angle = acosf(dotProd);

                // Curvature normal weighting.
                float slope = tanf(angle);

                if(0 == slope)
                  slope = numeric_limits<float>::epsilon();

                // Note: Some weights will be negative, due to obtuse triangles.
                // You may wish to do weights[j] += fabsf(1.0f / slope); here.
                weights[j] += 1.0f / slope;

                angle_count++;

                break;
              }
            }

            // Since we found a triangle match, we can skip to the first vertex's next triangle.
            break;
          }
        }
      } // End of: Find out which two triangles are shared by the vertex pair.

      if(angle_count != 2)
        angle_error++;

    } // End of: For each vertex pair (ie. each edge).

    if(angle_error != 0)
    {
      MITK_INFO << "Warning: Vertex " << i << " belongs to " << angle_error << " edges that do not belong to two triangles (" << m_MeshDataExt->m_VertexToVertexIndices[i].size() - angle_error << " edges were OK)." << endl;
      MITK_INFO << "Your mesh probably has cracks or holes in it." << endl;
    }

    // Normalize the weights so that they sum up to 1.
    float s = 0;

    // Note: Some weights will be negative, due to obtuse triangles.
    // You may wish to do s += fabsf(weights[j]); here.
    for(size_t j = 0; j < weights.size(); j++)
      s += weights[j];

    if(0 == s)
      s = numeric_limits<float>::epsilon();

    for(size_t j = 0; j < weights.size(); j++)
      weights[j] /= s;

    // Sum the displacements.
    for(size_t j = 0; j < m_MeshDataExt->m_VertexToVertexIndices[i].size(); j++)
    {
      size_t neighbour_j = m_MeshDataExt->m_VertexToVertexIndices[i][j];

      displacements[i] += (m_MeshDataExt->m_Vertices[neighbour_j].GetCoords() - m_MeshDataExt->m_Vertices[i].GetCoords())*weights[j];
    }
  }

  // To do: Find out why there are cases where displacement is much, much, much larger than all edge lengths put together.

  // Apply per-vertex displacement.
  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
    m_MeshDataExt->m_Vertices[i].SetCoords(m_MeshDataExt->m_Vertices[i].GetCoords() + displacements[i]*scale);
}

void MeshSmoother::SetMaxExtent(float max_extent)
{
  float curr_x_min = numeric_limits<float>::max();
  float curr_y_min = numeric_limits<float>::max();
  float curr_z_min = numeric_limits<float>::max();
  float curr_x_max = numeric_limits<float>::min();
  float curr_y_max = numeric_limits<float>::min();
  float curr_z_max = numeric_limits<float>::min();

  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
  {
    if(m_MeshDataExt->m_Vertices[i].GetCoordX() < curr_x_min)
      curr_x_min = m_MeshDataExt->m_Vertices[i].GetCoordX();

    if(m_MeshDataExt->m_Vertices[i].GetCoordX() > curr_x_max)
      curr_x_max = m_MeshDataExt->m_Vertices[i].GetCoordX();

    if(m_MeshDataExt->m_Vertices[i].GetCoordY() < curr_y_min)
      curr_y_min = m_MeshDataExt->m_Vertices[i].GetCoordY();

    if(m_MeshDataExt->m_Vertices[i].GetCoordY() > curr_y_max)
      curr_y_max = m_MeshDataExt->m_Vertices[i].GetCoordY();

    if(m_MeshDataExt->m_Vertices[i].GetCoordZ() < curr_z_min)
      curr_z_min = m_MeshDataExt->m_Vertices[i].GetCoordZ();

    if(m_MeshDataExt->m_Vertices[i].GetCoordZ() > curr_z_max)
      curr_z_max = m_MeshDataExt->m_Vertices[i].GetCoordZ();
  }

  float curr_x_extent = fabsf(curr_x_min - curr_x_max);
  float curr_y_extent = fabsf(curr_y_min - curr_y_max);
  float curr_z_extent = fabsf(curr_z_min - curr_z_max);

  float curr_max_extent = curr_x_extent;

  if(curr_y_extent > curr_max_extent)
    curr_max_extent = curr_y_extent;

  if(curr_z_extent > curr_max_extent)
    curr_max_extent = curr_z_extent;

  float scale_value = max_extent / curr_max_extent;

  MITK_INFO << "Original max extent: " << curr_max_extent << endl;
  MITK_INFO << "Scaling all vertices by a factor of: " << scale_value << endl;
  MITK_INFO << "New max extent: " << max_extent << endl;

  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
    m_MeshDataExt->m_Vertices[i].SetCoords(m_MeshDataExt->m_Vertices[i].GetCoords() * scale_value);
}

void MeshSmoother::RescaleMesh(double scale_value)
{
  MITK_INFO << "Scaling all vertices by a factor of: " << scale_value << endl;

  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
    m_MeshDataExt->m_Vertices[i].SetCoords(m_MeshDataExt->m_Vertices[i].GetCoords() * scale_value);
}


void MeshSmoother::GenerateVertexNormals(void)
{
  if(m_MeshDataExt->m_Triangles.size() == 0 || m_MeshDataExt->m_Vertices.size() == 0)
    return;

  m_VertexNormals.clear();
  m_VertexNormals.resize(m_MeshDataExt->m_Vertices.size());

  for(size_t i = 0; i < m_MeshDataExt->m_Triangles.size(); i++)
  {
    BasicVec3D v0 = m_MeshDataExt->m_Vertices[m_MeshDataExt->m_Triangles[i].GetVert2Index()].GetCoords() - m_MeshDataExt->m_Vertices[m_MeshDataExt->m_Triangles[i].GetVert1Index()].GetCoords();
    BasicVec3D v1 = m_MeshDataExt->m_Vertices[m_MeshDataExt->m_Triangles[i].GetVert3Index()].GetCoords() - m_MeshDataExt->m_Vertices[m_MeshDataExt->m_Triangles[i].GetVert1Index()].GetCoords();
    BasicVec3D v2 = v0.Cross(v1);

    m_VertexNormals[m_MeshDataExt->m_Triangles[i].GetVert1Index()] = m_VertexNormals[m_MeshDataExt->m_Triangles[i].GetVert1Index()] + v2;
    m_VertexNormals[m_MeshDataExt->m_Triangles[i].GetVert2Index()] = m_VertexNormals[m_MeshDataExt->m_Triangles[i].GetVert2Index()] + v2;
    m_VertexNormals[m_MeshDataExt->m_Triangles[i].GetVert3Index()] = m_VertexNormals[m_MeshDataExt->m_Triangles[i].GetVert3Index()] + v2;
  }

  for(size_t i = 0; i < m_VertexNormals.size(); i++)
  {
    m_VertexNormals[i].Normalize();

    // Sometimes we must invert the normals 
    if (m_FlipNormals)
    {
      m_MeshDataExt->m_Vertices[i].SetNormalX(m_MeshDataExt->m_Vertices[i].GetNormalX() - m_VertexNormals[i].GetX());
      m_MeshDataExt->m_Vertices[i].SetNormalY(m_MeshDataExt->m_Vertices[i].GetNormalY() - m_VertexNormals[i].GetY());
      m_MeshDataExt->m_Vertices[i].SetNormalZ(m_MeshDataExt->m_Vertices[i].GetNormalZ() - m_VertexNormals[i].GetZ());
    }
    else
    {
      m_MeshDataExt->m_Vertices[i].SetNormalX(m_VertexNormals[i].GetX());
      m_MeshDataExt->m_Vertices[i].SetNormalY(m_VertexNormals[i].GetY());
      m_MeshDataExt->m_Vertices[i].SetNormalZ(m_VertexNormals[i].GetZ());
    }
  }
}

void MeshSmoother::GenerateTriangleNormals(void)
{
  if(m_MeshDataExt->m_Triangles.size() == 0)
    return;

  m_TriangleNormals.clear();
  m_TriangleNormals.resize(m_MeshDataExt->m_Triangles.size());

  for(size_t i = 0; i < m_MeshDataExt->m_Triangles.size(); i++)
  {
    // Create a temporary triangle
    BasicTriangle tmpTriangle(m_MeshDataExt->m_Triangles[i]);

    BasicVec3D vert1 = m_MeshDataExt->m_Vertices[m_MeshDataExt->m_Triangles[i].GetVert1Index()].GetCoords();
    BasicVec3D vert2 = m_MeshDataExt->m_Vertices[m_MeshDataExt->m_Triangles[i].GetVert2Index()].GetCoords();
    BasicVec3D vert3 = m_MeshDataExt->m_Vertices[m_MeshDataExt->m_Triangles[i].GetVert3Index()].GetCoords();

    BasicVec3D vec0 = vert2 - vert1;
    BasicVec3D vec1 = vert3 - vert1;
    m_TriangleNormals[i] = vec0.NormalizedCross(vec1);

    if (m_FlipNormals)
    {
      tmpTriangle.SetTriNormalX(m_MeshDataExt->m_Triangles[i].GetTriNormalX() - m_TriangleNormals[i].GetX());
      tmpTriangle.SetTriNormalY(m_MeshDataExt->m_Triangles[i].GetTriNormalY() - m_TriangleNormals[i].GetY());
      tmpTriangle.SetTriNormalZ(m_MeshDataExt->m_Triangles[i].GetTriNormalZ() - m_TriangleNormals[i].GetZ());
    }
    else
    {
      tmpTriangle.SetTriNormalX(m_TriangleNormals[i].GetX());
      tmpTriangle.SetTriNormalY(m_TriangleNormals[i].GetY());
      tmpTriangle.SetTriNormalZ(m_TriangleNormals[i].GetZ());
    }

    // This is the "d" from the plane equation ax + by + cz + d = 0;
    float dParam = -(tmpTriangle.GetTriNormal().Dot(vert1));
    tmpTriangle.SetDParam(dParam);

    m_MeshDataExt->m_Triangles.operator[](i) = tmpTriangle;
  }
}

void MeshSmoother::GenerateVertexAndTriangleNormals(void)
{
  // Sanity check
  if (m_MeshDataExt == 0)
  {
    MITK_ERROR <<"Invalid data pointer, MeshSmoother wasn't initialized properly!";
    return;
  }

  GenerateVertexNormals();
  GenerateTriangleNormals();
}

void MeshSmoother::ReOrientFaces(void)
{
  // Sanity check
  if (m_MeshDataExt == 0)
  {
    MITK_ERROR <<"Invalid data pointer, MeshSmoother wasn't initialized properly!";
    return;
  }

  for(size_t i = 0; i < m_MeshDataExt->m_Triangles.size(); i++)
  {
    size_t tmp = m_MeshDataExt->m_Triangles[i].GetVert3Index();
    m_MeshDataExt->m_Triangles[i].SetVert3Index(m_MeshDataExt->m_Triangles[i].GetVert2Index());
    m_MeshDataExt->m_Triangles[i].SetVert2Index(tmp);
  }
}


void MeshSmoother::RegenerateVertexAndTriangleNormalsIfExists(void)
{
  if(m_TriangleNormals.size() > 0)
    GenerateTriangleNormals();

  if(m_VertexNormals.size() > 0)
    GenerateVertexNormals();
}

void MeshSmoother::FixCracks(void)
{
  // Sanity check
  if (m_MeshDataExt == 0)
  {
    MITK_ERROR <<"Invalid data pointer, MeshSmoother wasn't initialized properly!";
    return;
  }

  // Find edges that belong to only one triangle.
  set<ordered_indexed_edge> problem_edges_set;
  size_t problem_edge_id = 0;

  // For each vertex.
  for(size_t i = 0; i < m_MeshDataExt->m_Vertices.size(); i++)
  {
    // For each edge.
    for(size_t j = 0; j < m_MeshDataExt->m_VertexToVertexIndices[i].size(); j++)
    {
      size_t triangle_count = 0;
      size_t neighbour_j = m_MeshDataExt->m_VertexToVertexIndices[i][j];

      // Find out which two triangles are shared by this edge.
      for(size_t k = 0; k < m_MeshDataExt->m_VertexToTriangleIndices[i].size(); k++)
      {
        for(size_t l = 0; l < m_MeshDataExt->m_VertexToTriangleIndices[neighbour_j].size(); l++)
        {
          size_t tri0_index = m_MeshDataExt->m_VertexToTriangleIndices[i][k];
          size_t tri1_index = m_MeshDataExt->m_VertexToTriangleIndices[neighbour_j][l];

          if(tri0_index == tri1_index)
          {
            triangle_count++;
            break;
          }
        }
      } // End of: Find out which two triangles are shared by this edge.

      // Found a problem edge.
      if(triangle_count != 2)
      {
        BasicVertex v0;
        v0.SetIndex(i);
        v0.SetCoordX(m_MeshDataExt->m_Vertices[i].GetCoordX());
        v0.SetCoordY(m_MeshDataExt->m_Vertices[i].GetCoordY());
        v0.SetCoordZ(m_MeshDataExt->m_Vertices[i].GetCoordZ());

        BasicVertex v1;
        v1.SetIndex(neighbour_j);
        v1.SetCoordX(m_MeshDataExt->m_Vertices[neighbour_j].GetCoordX());
        v1.SetCoordY(m_MeshDataExt->m_Vertices[neighbour_j].GetCoordY());
        v1.SetCoordZ(m_MeshDataExt->m_Vertices[neighbour_j].GetCoordZ());

        ordered_indexed_edge problem_edge(v0, v1);

        if(problem_edges_set.end() == problem_edges_set.find(problem_edge))
        {
          problem_edge.id = problem_edge_id++;
          problem_edges_set.insert(problem_edge);
        }
      } // End of: Found a problem edge.
    } // End of: For each edge.
  } // End of: For each vertex.

  if(0 == problem_edges_set.size())
  {
    //MITK_INFO << "No cracks found -- the mesh seems to be in good condition" << endl;
    return;
  }

  if(0 != problem_edges_set.size() % 2)
  {
    MITK_INFO << "Found " << problem_edges_set.size() << " problem edges" << endl;
    MITK_ERROR << "Error -- the number of problem edges must be an even number (perhaps the mesh has holes?). Aborting." << endl;
    return;
  }

  // Make a copy of the set into a vector because the edge matching will
  // run a bit faster while looping through a vector by index vs looping through
  // a set by iterator.
  vector<ordered_indexed_edge> problem_edges_vec(problem_edges_set.begin(), problem_edges_set.end());
  vector<bool> processed_problem_edges(problem_edges_set.size(), false);
  problem_edges_set.clear();

  set<ordered_size_t_pair> merge_vertices;

  //MITK_INFO << "Pairing problem edges" << endl;

  // Each problem edge is practically a duplicate of some other, but not quite exactly.
  // So, find the closest match for each problem edge.
  for(size_t i = 0; i < problem_edges_vec.size(); i++)
  {
    // This edge has already been matched up previously, so skip it.
    if(true == processed_problem_edges[problem_edges_vec[i].id])
      continue;

    float closest_dist_sq = numeric_limits<float>::max();
    size_t closest_problem_edges_vec_index = 0;

    for(size_t j = i + 1; j < problem_edges_vec.size(); j++)
    {
      // Note: Don't check to see if this edge has been processed yet.
      // Doing so will actually only slow this down further.
      // Perhaps vector<bool> is a bit of a sloth?
      //if(true == processed_problem_edges[problem_edges_vec[j].id])
      //	continue;

      float dist_sq = problem_edges_vec[i].centre_point.DistanceSquared(problem_edges_vec[j].centre_point);

      if(dist_sq < closest_dist_sq)
      {
        closest_dist_sq = dist_sq;
        closest_problem_edges_vec_index = j;
      }
    }

    processed_problem_edges[problem_edges_vec[i].id] = true;
    processed_problem_edges[problem_edges_vec[closest_problem_edges_vec_index].id] = true;

    // If edge 0 vertex 0 is further in space from edge 1 vertex 0 than from edge 1 vertex 1,
    // then swap the indices on the edge 1 -- this makes sure that the edges are not pointing
    // in opposing directions.
    BasicVec3D vert1 = m_MeshDataExt->m_Vertices[problem_edges_vec[i].indices[0]].GetCoords();
    BasicVec3D vert2 = m_MeshDataExt->m_Vertices[problem_edges_vec[closest_problem_edges_vec_index].indices[0]].GetCoords();
    BasicVec3D vert3 = m_MeshDataExt->m_Vertices[problem_edges_vec[closest_problem_edges_vec_index].indices[1]].GetCoords();
    
    if(vert1.DistanceSquared(vert2) > vert1.DistanceSquared(vert3))
    {
      size_t temp = problem_edges_vec[closest_problem_edges_vec_index].indices[0];
      problem_edges_vec[closest_problem_edges_vec_index].indices[0] = problem_edges_vec[closest_problem_edges_vec_index].indices[1];
      problem_edges_vec[closest_problem_edges_vec_index].indices[1] = temp;
    }

    // If the first indices aren't already the same, then merge them.
    if(problem_edges_vec[i].indices[0] != problem_edges_vec[closest_problem_edges_vec_index].indices[0])
      merge_vertices.insert(ordered_size_t_pair(problem_edges_vec[i].indices[0], problem_edges_vec[closest_problem_edges_vec_index].indices[0]));

    // If the second indices aren't already the same, then merge them.
    if(problem_edges_vec[i].indices[1] != problem_edges_vec[closest_problem_edges_vec_index].indices[1])
      merge_vertices.insert(ordered_size_t_pair(problem_edges_vec[i].indices[1], problem_edges_vec[closest_problem_edges_vec_index].indices[1]));
  }

  //MITK_INFO << "Merging " << merge_vertices.size() << " vertex pairs" << endl;

  for(set<ordered_size_t_pair>::const_iterator ci = merge_vertices.begin(); ci != merge_vertices.end(); ci++)
    MergeVertexPair(ci->indices[0], ci->indices[1]);

  // Recalculate normals, if necessary.
  //RegenerateVertexAndTriangleNormalsIfExists();
}

template<typename T> void MeshSmoother::EliminateVectorDuplicates(vector<T> &v)
{
  if(0 == v.size())
    return;

  set<T> s(v.begin(), v.end()); // Eliminate duplicates (and sort them)
  vector<T> vtemp(s.begin(), s.end()); // Stuff things back into a temp vector
  v.swap(vtemp); // Assign temp vector contents to destination vector
}

bool MeshSmoother::MergeVertexPair(const size_t keeper, const size_t goner)
{
  if(keeper >= m_MeshDataExt->m_Vertices.size() || goner >= m_MeshDataExt->m_Vertices.size())
    return false;

  if(keeper == goner)
    return true;

  // Merge vertex to triangle data.

  // Add goner's vertex to triangle data to keeper's triangle to vertex data,
  // and replace goner's index with keeper's index in all relevant triangles' index data.
  for(size_t i = 0; i < m_MeshDataExt->m_VertexToTriangleIndices[goner].size(); i++)
  {
    size_t tri_index = m_MeshDataExt->m_VertexToTriangleIndices[goner][i];

    m_MeshDataExt->m_VertexToTriangleIndices[keeper].push_back(tri_index);

    for(size_t j = 0; j < 3; j++)
      if(goner == m_MeshDataExt->m_Triangles[tri_index].GetVertIndex(j))
        m_MeshDataExt->m_Triangles[tri_index].SetVertIndex(j, keeper);
  }

  // Finalize keeper's vertex to triangle data.
  EliminateVectorDuplicates(m_MeshDataExt->m_VertexToTriangleIndices[keeper]);

  // Clear out goner's vertex to triangle data for good.
  m_MeshDataExt->m_VertexToTriangleIndices[goner].clear();


  // Merge vertex to vertex data.

  // Add goner's vertex to vertex data to keeper's vertex to vertex data,
  // and replace goner's index with keeper's index in all relevant vertices' vertex to vertex data.
  for(size_t i = 0; i < m_MeshDataExt->m_VertexToVertexIndices[goner].size(); i++)
  {
    size_t vert_index = m_MeshDataExt->m_VertexToVertexIndices[goner][i];

    m_MeshDataExt->m_VertexToVertexIndices[keeper].push_back(vert_index);

    for(size_t j = 0; j < m_MeshDataExt->m_VertexToVertexIndices[vert_index].size(); j++)
    {
      // Could probably break after this, but whatever...
      if(goner == m_MeshDataExt->m_VertexToVertexIndices[vert_index][j])
        m_MeshDataExt->m_VertexToVertexIndices[vert_index][j] = keeper;
    }

    EliminateVectorDuplicates(m_MeshDataExt->m_VertexToVertexIndices[vert_index]);
  }

  // Finalize keeper's vertex to vertex data.
  EliminateVectorDuplicates(m_MeshDataExt->m_VertexToVertexIndices[keeper]);

  // Clear out goner's vertex to vertex data for good.
  m_MeshDataExt->m_VertexToVertexIndices[goner].clear();

  // Note: At this point, m_Vertices[goner] is now a rogue vertex.
  // We will skip erasing it from the vertices vector because that would mean a whole lot more work
  // (we'd have to reindex every vertex after it in the vector, etc.). Instead it will simply not be 
  // referenced in the triangle list.

  return true;
}

} //endof mitk namespace