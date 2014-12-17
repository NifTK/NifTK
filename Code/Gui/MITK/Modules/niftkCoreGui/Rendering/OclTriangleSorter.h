/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#ifndef __mitkOclTriangleSorter_h
#define __mitkOclTriangleSorter_h

#include <mitkOclFilter.h>
#include <mitkOclResourceService.h>
#include <vlCore/VisualizationLibrary.hpp>

namespace mitk
{
class OclFilter;
class OclTriangleSorter;

   /**
    * \brief The OclTriangleSorter is the topmost class for all filter which take surfaces as input.
    *
    * The input surface can be intialized via an oclSurface or an mitk::Surface or vtkPolyData.
    * This makes it possible to create a filter pipeline of GPU-based filters
    * and to bind this part into the CPU (ITK) filter pipeline.
    */
class OclTriangleSorter: public OclFilter
{
public:
  OclTriangleSorter();
  virtual ~OclTriangleSorter();

  /// \brief Performs the coord tranfrom, computes triangle distances and sorts triangles based on distance
  void Update();

  /// \brief Allows to force-release the GPU buffers when needed
  void ReleaseGPUBuffers();

  /// \brief Adds a vertex buffer to be merged
  void AddVertexBuffer(const cl_mem vertBuf, unsigned int vertCount);
  /// \brief Adds an index buffer to be merged
  void AddIndexBuffer(const cl_mem idxBuf, unsigned int idxCount);
  /// \brief Adds the transform that is assigned to the object to be merged
  void AddTransform(const cl_mem trasfBuf);
  /// \brief Adds camera position that is used for distance computation
  inline void SetViewPoint(cl_float4 vp) { m_ViewPoint = vp; }
  /// \brief Gets the resulting CL mem object that contains the IBO of the merged translucent object
  // and the total num of triangles
  void GetOutput(cl_mem &mergedAndSortedIndexBuf, cl_uint &totalTriangleNum);

  /// \brief Gets the resulting CL mem object that contains the distance of each veretx
  void GetDistOutput(cl_mem &mergedAndSortedDistBuf, cl_uint &totalVertexNum);

  /// \brief Resets all buffers and internal variables
  void Reset();

protected:
  /// \brief Initialize the filter
  bool Initialize();

  /// \brief Main processing function
  void Execute();

  /// \brief
  virtual us::Module* GetModule() { return 0; }

private:
  /// \brief Compile and initialize all computing kernels
  void InitKernels();
  
  // Sorting related kernels
  void LaunchRadixSort(cl_mem bfKeyVal, cl_uint count);
  void RadixLocal(cl_uint datasetSize, cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset);
  void LocalHistogram(cl_uint datasetSize, const size_t* globalWorkSize, const size_t* localWorkSize, cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset);
  void RadixPermute(cl_uint datasetSize, const size_t* globalWorkSize, const size_t* localWorkSize, cl_mem dataIn, cl_mem dataOut, cl_mem histScan, cl_mem blockHists, int bitOffset, unsigned int numBlocks);
  void Scan(cl_uint datasetSize, cl_mem dataIn, cl_mem dataOut);

  void LaunchBitonicSort(cl_mem bfKeyVal, cl_uint count);

  /// \brief Merges the input IBO's into one large buffer while updating the triangle indices
  void MergeBuffers(cl_mem mergedIndexBuffWithDist, cl_mem mergedIndexWithDist4Sort);

  /// \brief Performs coordinate transform and computes distance of the viewpoint from each vertex. Returns with the resulting mem object
  cl_mem TransformVerticesAndComputeDistance(cl_mem vertexBuf, cl_uint numOfVertices, cl_mem transform, cl_float4 viewPoint);
  
  /// \brief Computes the triangle distances from the vertex distances. Returns the array of floats as cl_mem
  cl_mem ComputeTriangleDistances(cl_mem vertexDistances, cl_uint numOfVertices, cl_mem indexBuffer, cl_uint numOfTriangles);

  /// \brief Copies the contents of one IBO (vetex indices + distance) to the merged IBO while updating the indices
  void CopyAndUpdateIndices(cl_mem input, cl_mem output, cl_mem output4Sort, cl_uint size, cl_uint offset);

  /// \brief Copies only the vertex indices, without the distance bit
  void CopyIndicesOnly(cl_mem input, cl_mem output, cl_uint size);

  /// \brief Copies only the vertex indices and distances separately
  void CopyIndicesWithDist(cl_mem input, cl_mem output, cl_mem outputDist, cl_uint size, cl_uint sizeDist);

  // This one converts back the floats from the uint32 representation that we use in the sorting algorithm.
  // Rather ugly - but useful in debugging the distance sorting
  inline float IFloatFlip(unsigned int f)
  {
    unsigned int mask = ((f >> 31) - 1) | 0x80000000;
    unsigned int val = f ^ mask;
    float * floatVal = reinterpret_cast<float*>(&val);
    return *floatVal;
  }

private:
  bool m_KernelsReady;

  // Inputs
  std::vector<cl_mem>       m_VertexBuffers;
  std::vector<cl_mem>       m_IndexBuffers;
  std::vector<cl_mem>       m_TransformBuffers;
  std::vector<unsigned int> m_VertexCounts;
  std::vector<unsigned int> m_TriangleCounts;
  cl_float4                 m_ViewPoint;

  // Big fat buffer for storing the merged translucent object's indices
  cl_mem                    m_MergedIndexBuffer;
  // Num of triangles in the merged translucent object
  cl_uint                   m_TotalTriangleNum;
  // Big fat buffer for storing the merged translucent object's distances
  cl_mem                    m_MergedDistanceBuffer;
  // Num of vertices in the merged translucent object
  cl_uint                   m_TotalVertexNum;

  // Parallel Prefix Sum kernel
  cl_kernel    m_ckScanBlockAnyLength;

  /// Radix Sort and related kernels
  unsigned int m_SortValueSize;
  unsigned int m_SortKeySize;
  unsigned int m_SortBits;
  size_t       m_SortWorkgroupSize;

  cl_kernel    m_ckRadixLocalSort;
  cl_kernel    m_ckLocalHistogram;
  cl_kernel    m_ckRadixPermute;

  /// Bitonic Sort stuff
  cl_kernel    m_ckParallelBitonic_B2;
  cl_kernel    m_ckParallelBitonic_B4;
  cl_kernel    m_ckParallelBitonic_B8;
  cl_kernel    m_ckParallelBitonic_B16;
  cl_kernel    m_ckParallelBitonic_C4;

  std::vector<cl_kernel> m_BitonicSortKernels;

  // Custom computation kernels
  cl_kernel    m_ckTransformVertexAndComputeDistance;
  cl_kernel    m_ckComputeTriangleDistances;
  cl_kernel    m_ckCopyAndUpdateIndices;
  cl_kernel    m_ckCopyIndicesOnly;
  cl_kernel    m_ckCopyIndicesWithDist;

  cl_kernel    m_ckTest;
};

}
#endif // __mitkOclTriangleSorter_h
