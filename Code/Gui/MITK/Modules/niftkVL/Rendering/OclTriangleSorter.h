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

#include <float.h>
#include <list>

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

  /// \brief Allows to force-release the GPU buffers when needed
  void ReleaseGPUBuffers();

  /// \brief Adds a vertex buffer to be merged
  void AddGLVertexBuffer(const GLuint vertBufHandle, unsigned int vertCount);
  /// \brief Adds an index buffer to be merged
  void AddGLIndexBuffer(const GLuint idxBufHandle, unsigned int idxCount);
  
  /// \brief Adds camera position that is used for distance computation
  inline void SetViewPoint(cl_float4 vp) { m_ViewPoint = vp; }
  
  /// \brief Sorts the merged index buffer based on triangle distance
  void SortIndexBufferByDist(cl_mem &mergedIndexBuf, cl_mem &mergedVertexBuf, cl_uint triCount, cl_uint vertCount);

  /// \brief Merges index buffers that were previously set to be merged
  void MergeIndexBuffers(cl_mem &mergedBuffer, cl_uint &numOfElements);

  /// \brief Gets the resulting CL mem object that contains the distance of each triangle
  void GetTriangleDistOutput(cl_mem &mergedAndSortedDistBuf, cl_uint &totalTriangleNum);

  /// \brief Resets all buffers and internal variables
  void Reset();

  // This one converts back the floats from the uint32 representation that we use in the sorting algorithm.
  // Rather ugly - but useful in debugging the distance sorting
  inline static float IFloatFlip(unsigned int f)
  {
    unsigned int mask = ((f >> 31) - 1) | 0x80000000;
    unsigned int val = f ^ mask;
    float * floatVal = reinterpret_cast<float*>(&val);
    return *floatVal;
  }

protected:
  /// \brief Initialize the filter
  bool Initialize();

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

  /// \brief Performs coordinate transform and computes distance of the viewpoint from each vertex. Returns with the resulting mem object
  void TransformVerticesAndComputeDistance(cl_mem vertexBuf, cl_uint numOfVertices, cl_float4 viewPoint);
  
  /// \brief Computes the triangle distances from the vertex distances. Returns the array of floats as cl_mem
  void ComputeTriangleDistances(cl_mem vertexDistances, cl_uint numOfVertices, cl_mem indexBuffer, cl_uint numOfTriangles);

  /// \brief Copies the contents of one IBO (vetex indices + distance) to the merged IBO while updating the indices
  void CopyAndUpdateIndices(cl_mem input, cl_mem output, cl_uint size, cl_uint triOffset, cl_uint vertOffset);

  /// \brief Copies only the vertex indices, without the distance bit
  void CopyIndicesOnly(cl_mem input, cl_mem output, cl_uint size);

  /// \brief Copies only the vertex indices and distances separately
  void CopyIndicesWithDist(cl_mem input, cl_mem output, cl_mem outputDist, cl_uint size);

private:
  bool m_KernelsReady;

  // Inputs
  std::vector<GLuint>       m_GLVertexBuffers;
  std::vector<GLuint>       m_GLIndexBuffers;
  std::vector<unsigned int> m_VertexCounts;
  std::vector<unsigned int> m_TriangleCounts;
  cl_float4                 m_ViewPoint;


  // Buffer for storing translucent triangles' distances
  cl_mem                    m_VertexDistances;
  // Buffer for storing the index buffer together with distance for sorting
  cl_mem                    m_IndexBufferWithDist;

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

  // Release temporary objects
  cl_mem m_bfRadixHist1;
  cl_mem m_bfRadixHist2;
  cl_mem m_bfRadixHist1Scan;
  cl_mem m_bfDataA;
  cl_mem m_bfDataB;

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
