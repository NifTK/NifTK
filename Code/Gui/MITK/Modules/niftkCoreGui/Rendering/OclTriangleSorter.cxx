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

#include "OclTriangleSorter.h"
#include <mitkOclFilter.h>

#include <iostream>
#include <fstream>
#include <iomanip>

enum Kernels {
  PARALLEL_BITONIC_B2_KERNEL,
  PARALLEL_BITONIC_B4_KERNEL,
  PARALLEL_BITONIC_B8_KERNEL,
  PARALLEL_BITONIC_B16_KERNEL,
  PARALLEL_BITONIC_C4_KERNEL,
  NB_KERNELS
};

const char * KernelNames[NB_KERNELS+1] = 
{
  "ParallelBitonic_B2",
  "ParallelBitonic_B4",
  "ParallelBitonic_B8",
  "ParallelBitonic_B16",
  "ParallelBitonic_C4",
  0 
};


mitk::OclTriangleSorter::OclTriangleSorter()
{
  this->AddSourceFile(":/NewVisualization/TriangleSorter.cl");
  this->m_FilterID = "TriangleSorter";

  // Prefix Sum
  m_ckScanBlockAnyLength  = 0;
  // Radix sort
  m_ckRadixLocalSort  = 0;
  m_ckLocalHistogram  = 0;
  m_ckRadixPermute    = 0;

  // Bitonic Sort
  m_ckParallelBitonic_B2 = 0;
  m_ckParallelBitonic_B4 = 0;
  m_ckParallelBitonic_B8 = 0;
  m_ckParallelBitonic_B16 = 0;
  m_ckParallelBitonic_C4 = 0;

  // This value (m_SortBits) used to be 32. We have to reduce to 28, otherwise the sorting
  // of the floats-converted-to-uints fails.
  m_SortBits          = 28;
  m_SortWorkgroupSize = 32;
  m_SortValueSize     = 4;
  m_SortKeySize       = 4;

  // Custom computations
  m_ckTransformVertexAndComputeDistance = 0;
  m_ckComputeTriangleDistances = 0;
  m_ckCopyAndUpdateIndices = 0;
  m_ckCopyIndicesOnly = 0;
  m_ckTest = 0;

  m_MergedIndexBuffer = 0;
  m_TotalTriangleNum = 0;

  m_KernelsReady = false;
}

mitk::OclTriangleSorter::~OclTriangleSorter()
{
  clFinish(m_CommandQue);

  // Prefix Sum
  if (m_ckScanBlockAnyLength)  clReleaseKernel(m_ckScanBlockAnyLength);
 
  // Radix Sort
  if (m_ckRadixLocalSort) clReleaseKernel(m_ckRadixLocalSort);
  if (m_ckLocalHistogram) clReleaseKernel(m_ckLocalHistogram);
  if (m_ckRadixPermute)   clReleaseKernel(m_ckRadixPermute);

  if (m_ckParallelBitonic_B2)  clReleaseKernel(m_ckParallelBitonic_B2);
  if (m_ckParallelBitonic_B4)  clReleaseKernel(m_ckParallelBitonic_B4);
  if (m_ckParallelBitonic_B8)  clReleaseKernel(m_ckParallelBitonic_B8);
  if (m_ckParallelBitonic_B16) clReleaseKernel(m_ckParallelBitonic_B16);
  if (m_ckParallelBitonic_C4)  clReleaseKernel(m_ckParallelBitonic_C4);

  // Custom computations
  if (m_ckTransformVertexAndComputeDistance) clReleaseKernel(m_ckTransformVertexAndComputeDistance);
  if (m_ckComputeTriangleDistances) clReleaseKernel(m_ckComputeTriangleDistances);
  if (m_ckCopyAndUpdateIndices) clReleaseKernel(m_ckCopyAndUpdateIndices);
  if (m_ckCopyIndicesOnly) clReleaseKernel(m_ckCopyIndicesOnly);

  if (m_ckTest) clReleaseKernel(m_ckTest);

  if (m_MergedIndexBuffer)
    clReleaseMemObject(m_MergedIndexBuffer);

  m_MergedIndexBuffer = 0;
}

void mitk::OclTriangleSorter::Reset()
{
  m_VertexBuffers.clear();
  m_IndexBuffers.clear();
  m_TransformBuffers.clear();
  m_VertexCounts.clear();
  m_TriangleCounts.clear();
  cl_float4 float4Max = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
  m_ViewPoint = float4Max;

  if (m_MergedIndexBuffer)
    clReleaseMemObject(m_MergedIndexBuffer);

  m_MergedIndexBuffer = 0;
  m_TotalTriangleNum = 0;
}

void mitk::OclTriangleSorter::Update()
{
  //Check if context & program available
  if (!this->Initialize())
  {
    //// clean-up also the resources
    //resources->InvalidateStorage();
    mitkThrow() <<"Filter is not initialized. Cannot update.";
  } 
  else
  {
    // Execute
    this->Execute();
  }
}

bool mitk::OclTriangleSorter::Initialize()
{
  cl_int clErr = 0;

  if (m_IndexBuffers.size() == 0 || m_VertexBuffers.size() == 0)
    mitkThrow() << "Input buffers aren't set.";

  m_TotalTriangleNum = 0;
  for (unsigned int i = 0; i < m_VertexBuffers.size(); i++)
  {
    m_TotalTriangleNum += m_TriangleCounts[i];
  }

  //! [Initialize]
  if (!OclFilter::Initialize())
  {
    MITK_ERROR << "Caught exception while initializing filter: " << CHECK_OCL_ERR(clErr);
    return false;
  }

  if (!m_KernelsReady)
    InitKernels();

  if (!m_KernelsReady)
  {
    return false;
  }

  return true;
}

void mitk::OclTriangleSorter::InitKernels()
{
  int buildErr = 0;
  cl_int clErr = 0;

  // Prefix Sum
  this->m_ckScanBlockAnyLength = clCreateKernel( this->m_ClProgram, "ckScanBlockAnyLength", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  /// Sort kernels
  this->m_ckRadixLocalSort = clCreateKernel( this->m_ClProgram, "ckRadixLocalSort", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  this->m_ckLocalHistogram = clCreateKernel( this->m_ClProgram, "ckLocalHistogram", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  this->m_ckRadixPermute = clCreateKernel( this->m_ClProgram, "ckRadixPermute", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  this->m_ckParallelBitonic_B2 = clCreateKernel( this->m_ClProgram, "ckParallelBitonic_B2", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;
  m_BitonicSortKernels.push_back(m_ckParallelBitonic_B2);

  this->m_ckParallelBitonic_B4 = clCreateKernel( this->m_ClProgram, "ckParallelBitonic_B4", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;
  m_BitonicSortKernels.push_back(m_ckParallelBitonic_B4);

  this->m_ckParallelBitonic_B8 = clCreateKernel( this->m_ClProgram, "ckParallelBitonic_B8", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;
  m_BitonicSortKernels.push_back(m_ckParallelBitonic_B8);

  this->m_ckParallelBitonic_B16 = clCreateKernel( this->m_ClProgram, "ckParallelBitonic_B16", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;
  m_BitonicSortKernels.push_back(m_ckParallelBitonic_B16);

  this->m_ckParallelBitonic_C4 = clCreateKernel( this->m_ClProgram, "ckParallelBitonic_C4", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;
  m_BitonicSortKernels.push_back(m_ckParallelBitonic_C4);

  this->m_ckTransformVertexAndComputeDistance = clCreateKernel(this->m_ClProgram, "ckTransformVertexAndComputeDistance", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  this->m_ckComputeTriangleDistances = clCreateKernel(this->m_ClProgram, "ckComputeTriangleDistances", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  this->m_ckCopyAndUpdateIndices = clCreateKernel(this->m_ClProgram, "ckCopyAndUpdateIndices", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  this->m_ckCopyIndicesOnly = clCreateKernel(this->m_ClProgram, "ckCopyIndicesOnly", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  this->m_ckTest = clCreateKernel(this->m_ClProgram, "ckTest", &clErr);
  CHECK_OCL_ERR( clErr );
  buildErr |= clErr;

  if (buildErr != 0)
  {
    MITK_ERROR <<"Error while compiling OpenCL kernels!";
    m_KernelsReady = false;
  }
  else m_KernelsReady = true;
}


/// \brief Main processing function
void mitk::OclTriangleSorter::Execute()
{
  cl_int clErr = 0;
  cl_mem mergedIndexBuffWithDist  = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, m_TotalTriangleNum* sizeof(cl_uint4), 0, &clErr);
  cl_mem mergedIndexWithDist4Sort = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, m_TotalTriangleNum* sizeof(cl_uint2), 0, &clErr);

  //// Merge input buffers in one fat buffer, that includes the distance of the triangle as val.x
  MergeBuffers(mergedIndexBuffWithDist, mergedIndexWithDist4Sort);

  //// Sort the triangles based on the distance
  //LaunchBitonicSort(mergedIndexBuffWithDist, m_TotalTriangleNum);
  LaunchRadixSort(mergedIndexBuffWithDist, m_TotalTriangleNum);
  
  if (m_MergedIndexBuffer)
    clReleaseMemObject(m_MergedIndexBuffer);
  m_MergedIndexBuffer = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, m_TotalTriangleNum*3*sizeof(cl_uint), 0, &clErr);

  MITK_INFO <<"Creating m_MergedIndexBuffer " <<m_TotalTriangleNum <<" " <<m_TotalTriangleNum*3*sizeof(cl_uint);

  CopyIndicesOnly(mergedIndexBuffWithDist, m_MergedIndexBuffer, m_TotalTriangleNum);


/*
  cl_uint * buff = new cl_uint[m_TotalTriangleNum*3];
  clErr = clEnqueueReadBuffer(m_CommandQue, m_MergedIndexBuffer, true, 0, m_TotalTriangleNum*3* sizeof(cl_uint), buff, 0, 0, 0);
  CHECK_OCL_ERR(clErr);

  std::ofstream outfile0;
  outfile0.open ("d://SortedMergedIBO.txt", std::ios::out);

      
  // Write out filtered volume
  for (int r = 0 ; r < m_TotalTriangleNum; r++)
  {
    outfile0 <<"Index: " <<r <<" Indices: " <<buff[r*3+0] <<" " <<buff[r*3+1] <<" " <<buff[r*3+2] <<"\n";
  }

  outfile0.close();
*/

/*
  cl_uint * buff2 = new cl_uint[m_TotalTriangleNum*4];
  clErr = clEnqueueReadBuffer(m_CommandQue, mergedIndexBuffWithDist, true, 0, m_TotalTriangleNum*4* sizeof(cl_uint), buff2, 0, 0, 0);
  CHECK_OCL_ERR(clErr);

  std::ofstream outfile1;
  outfile1.open ("d://SortedMergedIBO2.txt", std::ios::out);

      
  // Write out filtered volume
  for (int r = 0 ; r < m_TotalTriangleNum; r++)
  {
    outfile1 <<"Index: " <<r <<" Dist: " <<std::setprecision(10) <<IFloatFlip(buff2[r*4+0]) <<" Indices: " <<buff2[r*4+1] <<" " <<buff2[r*4+2] <<" " <<buff2[r*4+3] <<"\n";
  }

  outfile1.close();
*/

/*
  cl_uint * buff3 = new cl_uint[m_TotalTriangleNum*2];
  clErr = clEnqueueReadBuffer(m_CommandQue, mergedIndexWithDist4Sort, true, 0, m_TotalTriangleNum*2* sizeof(cl_uint), buff3, 0, 0, 0);
  CHECK_OCL_ERR(clErr);

  std::ofstream outfile2;
  outfile2.open ("d://SortedMergedIBO3.txt", std::ios::out);

      
  // Write out filtered volume
  for (int r = 0 ; r < m_TotalTriangleNum; r++)
  {
    outfile2 <<"Index: " <<r <<" Dist: " <<std::setprecision(10) <<IFloatFlip(buff3[r*2+0]) <<" Indices: " <<buff3[r*2+1]<<"\n";
  }

  outfile2.close();
*/
  clReleaseMemObject(mergedIndexBuffWithDist);
  clReleaseMemObject(mergedIndexWithDist4Sort);
}

void mitk::OclTriangleSorter::AddVertexBuffer(const cl_mem vertBuf, unsigned int vertCount)
{ 
  m_VertexBuffers.push_back(vertBuf);
  m_VertexCounts.push_back(vertCount);
} 

void mitk::OclTriangleSorter::AddIndexBuffer(const cl_mem idxBuf, unsigned int idxCount)
{ 
  m_IndexBuffers.push_back(idxBuf);
  m_TriangleCounts.push_back(idxCount);
} 

void mitk::OclTriangleSorter::AddTransform(const cl_mem trasfBuf) 
{ 
  m_TransformBuffers.push_back(trasfBuf);
} 

void mitk::OclTriangleSorter::GetOutput(cl_mem &mergedAndSortedIndexBuf, cl_uint &totalTriangleNum)
{
  // Compute mem size that will get copied to output
  size_t indexBufSize = m_TotalTriangleNum*3*sizeof(cl_uint);

  // Copy to merged buffer into output buffer
  cl_int clErr = clEnqueueCopyBuffer(m_CommandQue, m_MergedIndexBuffer, mergedAndSortedIndexBuf, 0, 0, indexBufSize, 0, 0, 0);
  CHECK_OCL_ERR(clErr);

  totalTriangleNum = m_TotalTriangleNum;
}

void mitk::OclTriangleSorter::CopyAndUpdateIndices(cl_mem input, cl_mem output, cl_mem output4Sort,  cl_uint size, cl_uint offset)
{
  cl_int clErr = 0;

  cl_int clStatus;
  unsigned int a = 0;

  int workgroupSize = 128;
  size_t global_128[1] = {ToMultipleOf(size, workgroupSize)};
  size_t local_128[1] = {workgroupSize};

  clStatus  = clSetKernelArg(m_ckCopyAndUpdateIndices, a++, sizeof(cl_mem), (const void*)&input);
  clStatus |= clSetKernelArg(m_ckCopyAndUpdateIndices, a++, sizeof(cl_mem), (const void*)&output);
  clStatus |= clSetKernelArg(m_ckCopyAndUpdateIndices, a++, sizeof(cl_mem), (const void*)&output4Sort);
  clStatus |= clSetKernelArg(m_ckCopyAndUpdateIndices, a++, sizeof(cl_uint), (const void*)&size);
  clStatus |= clSetKernelArg(m_ckCopyAndUpdateIndices, a++, sizeof(cl_uint), (const void*)&offset);
  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckCopyAndUpdateIndices, 1, NULL, global_128, local_128, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);
}

void mitk::OclTriangleSorter::CopyIndicesOnly(cl_mem input, cl_mem output, cl_uint size)
{
  cl_int clErr = 0;

  cl_int clStatus;
  unsigned int a = 0;

  int workgroupSize = 128;
  size_t global_128[1] = {ToMultipleOf(size, workgroupSize)};
  size_t local_128[1] = {workgroupSize};

  clStatus  = clSetKernelArg(m_ckCopyIndicesOnly, a++, sizeof(cl_mem), (const void*)&input);
  clStatus |= clSetKernelArg(m_ckCopyIndicesOnly, a++, sizeof(cl_mem), (const void*)&output);
  clStatus |= clSetKernelArg(m_ckCopyIndicesOnly, a++, sizeof(cl_uint), (const void*)&size);
  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckCopyIndicesOnly, 1, NULL, global_128, local_128, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);
}

void mitk::OclTriangleSorter::MergeBuffers(cl_mem mergedIndexBuffWithDist, cl_mem mergedIndexWithDist4Sort)
{
  cl_int clErr = 0;

  cl_uint offset = 0;
  for (unsigned int i = 0; i < m_VertexBuffers.size(); i++)
  {
    cl_mem vertexDistances = TransformVerticesAndComputeDistance(m_VertexBuffers[i], m_VertexCounts[i], m_TransformBuffers[i], m_ViewPoint);
    cl_mem indexBufferWithDist = ComputeTriangleDistances(vertexDistances, m_VertexCounts[i], m_IndexBuffers[i], m_TriangleCounts[i]);

    CopyAndUpdateIndices(indexBufferWithDist, mergedIndexBuffWithDist, mergedIndexWithDist4Sort, m_TriangleCounts[i], offset);

    offset += m_TriangleCounts[i];

    clReleaseMemObject(vertexDistances);
    clReleaseMemObject(indexBufferWithDist);
  }

/*
  cl_uint * buff = new cl_uint[m_TotalTriangleNum*4];
  clErr = clEnqueueReadBuffer(m_CommandQue, mergedIndexBuffWithDist, true, 0, m_TotalTriangleNum*4* sizeof(cl_uint), buff, 0, 0, 0);
  CHECK_OCL_ERR(clErr);

  std::ofstream outfile0;
  outfile0.open ("d://IBOwDist.txt", std::ios::out);
    
  // Write out filtered volume
  for (int r = 0 ; r < m_TotalTriangleNum; r++)
  {
    outfile0 <<"Index: " <<r <<" Dist: " <<std::setprecision(10) <<IFloatFlip(buff[r*4+0]) <<" Indices: " <<buff[r*4+1] <<" " <<buff[r*4+2] <<" " <<buff[r*4+3] <<"\n";
  }

  outfile0.close();

  cl_uint * buff2 = new cl_uint[m_TotalTriangleNum*2];
  clErr = clEnqueueReadBuffer(m_CommandQue, mergedIndexWithDist4Sort, true, 0, m_TotalTriangleNum*2* sizeof(cl_uint), buff2, 0, 0, 0);
  CHECK_OCL_ERR(clErr);

  std::ofstream outfile1;
  outfile1.open ("d://IndexwDist.txt", std::ios::out);
    
  // Write out filtered volume
  for (int r = 0 ; r < m_TotalTriangleNum; r++)
  {
    outfile1 <<"Index: " <<r <<" Dist: " <<std::setprecision(10) <<IFloatFlip(buff2[r*2+0]) <<" Indices: " <<buff2[r*2+1] <<"\n";
  }

  outfile1.close();
*/
}


inline int roundUpDiv(int A, int B) { return (A + B - 1) / (B); }

cl_mem mitk::OclTriangleSorter::TransformVerticesAndComputeDistance(cl_mem vertexBuf, cl_uint numOfVertices, cl_mem transform, cl_float4 viewPoint)
{
  cl_int clStatus = 0;
  cl_mem vertexDistances = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, numOfVertices*sizeof(cl_float), 0, &clStatus);
  CHECK_OCL_ERR( clStatus );

  unsigned int a = 0;

  int workgroupSize = 128;
  size_t global_128 = ToMultipleOf(numOfVertices, workgroupSize);
  size_t local_128  = workgroupSize;

  clStatus  = clSetKernelArg(m_ckTransformVertexAndComputeDistance, a++, sizeof(cl_mem), &vertexDistances);
  clStatus |= clSetKernelArg(m_ckTransformVertexAndComputeDistance, a++, sizeof(cl_mem), &vertexBuf);
  clStatus |= clSetKernelArg(m_ckTransformVertexAndComputeDistance, a++, sizeof(cl_mem), &transform);
  clStatus |= clSetKernelArg(m_ckTransformVertexAndComputeDistance, a++, sizeof(cl_float4), &viewPoint);
  clStatus |= clSetKernelArg(m_ckTransformVertexAndComputeDistance, a++, sizeof(cl_uint), &numOfVertices);

  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckTransformVertexAndComputeDistance, 1, NULL, &global_128, &local_128, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);

/*
  cl_float * buff = new cl_float[numOfVertices];
  clStatus = clEnqueueReadBuffer(m_CommandQue, vertexDistances, CL_TRUE, 0, numOfVertices*sizeof(cl_float), buff, 0, 0, 0);
  CHECK_OCL_ERR(clStatus);

  MITK_INFO <<"numOfVertices: " <<numOfVertices;
  std::ofstream outfile0;
  outfile0.open ("d://verticesWithDists.txt", std::ios::out);
    
  // Write out filtered volume
  for (int r = 0 ; r < numOfVertices; r++)
  {
    outfile0 <<"Index: " <<r <<" Dist: " <<std::setprecision(10) <<IFloatFlip(buff[r]) <<"\n";// <<" Vert: " <<buff[r*4+1] <<" " <<buff[r*4+2] <<" " <<buff[r*4+3] <<"\n";
  }

  outfile0.close();
*/

  return vertexDistances;
}
cl_mem mitk::OclTriangleSorter::ComputeTriangleDistances(
    cl_mem vertexDistances,
    cl_uint numOfVertices,
    cl_mem indexBuffer,
    cl_uint numOfTriangles)
{
  cl_int clErr = 0;
  cl_mem indexBufWithDist = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, numOfTriangles*sizeof(cl_uint4), 0, &clErr);
  CHECK_OCL_ERR( clErr );

  cl_int clStatus;
  unsigned int a = 0;

  int workgroupSize = 128;
  size_t global_128[1] = {ToMultipleOf(numOfTriangles, workgroupSize)};
  size_t local_128[1] = {workgroupSize};

  clStatus  = clSetKernelArg(m_ckComputeTriangleDistances, a++, sizeof(cl_mem),  (const void*)&indexBufWithDist);
  clStatus |= clSetKernelArg(m_ckComputeTriangleDistances, a++, sizeof(cl_mem),  (const void*)&vertexDistances);
  clStatus |= clSetKernelArg(m_ckComputeTriangleDistances, a++, sizeof(cl_mem),  (const void*)&indexBuffer);
  clStatus |= clSetKernelArg(m_ckComputeTriangleDistances, a++, sizeof(cl_uint), (const void*)&numOfVertices);
  clStatus |= clSetKernelArg(m_ckComputeTriangleDistances, a++, sizeof(cl_uint), (const void*)&numOfTriangles);
  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckComputeTriangleDistances, 1, NULL, global_128, local_128, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);

/*
  cl_uint * buff = new cl_uint[numOfTriangles*4];
  clStatus = clEnqueueReadBuffer(m_CommandQue, indexBufWithDist, CL_TRUE, 0, numOfTriangles*4*sizeof(cl_uint), buff, 0, 0, 0);
  CHECK_OCL_ERR(clStatus);

 
  MITK_INFO <<"numOfTriangles: " <<numOfTriangles;
  std::ofstream outfile0;
  outfile0.open ("d://trianglesWithDists.txt", std::ios::out);
    
  // Write out filtered volume
  for (int r = 0 ; r < numOfTriangles; r++)
  {
    outfile0 <<"Index: " <<r <<" Dist: " <<buff[r*4+0] <<" Indices: " <<buff[r*4+1] <<" " <<buff[r*4+2] <<" " <<buff[r*4+3] <<"\n";
  }

  outfile0.close();
*/
  return indexBufWithDist;

}

// Allowed "Bx" kernels (bit mask)
#define ALLOWB (2+4+8)

void mitk::OclTriangleSorter::LaunchBitonicSort(cl_mem bfKeyVal, cl_uint _datasetSize)
{
  unsigned int _valueSize = 4;
  unsigned int _keySize  = 4;

  int keyValueSize = _valueSize+_keySize;

  for(int length = 1; length < _datasetSize; length <<= 1)
  {
    int inc = length;
    std::list<int> strategy; // vector defining the sequence of reductions
    {
      int ii = inc;
      while (ii>0)
      {
        if (ii==128 || ii==32 || ii==8) { strategy.push_back(-1); break; } // C kernel
        int d = 1; // default is 1 bit
        if (0) d = 1;
#if 1
        // Force jump to 128
        else if (ii==256) d = 1;
        else if (ii==512 && (ALLOWB & 4)) d = 2;
        else if (ii==1024 && (ALLOWB & 8)) d = 3;
        else if (ii==2048 && (ALLOWB & 16)) d = 4;
#endif
        else if (ii>=8 && (ALLOWB & 16)) d = 4;
        else if (ii>=4 && (ALLOWB & 8)) d = 3;
        else if (ii>=2 && (ALLOWB & 4)) d = 2;
        else d = 1;

        strategy.push_back(d);
        ii >>= d;
      }
    }

    while (inc > 0)
    {
      int ninc = 0;
      int kid = -1;
      int doLocal = 0;
      int nThreads = 0;
      int d = strategy.front(); strategy.pop_front();

      switch (d)
      {
      case -1:
        kid = PARALLEL_BITONIC_C4_KERNEL;
        ninc = -1; // reduce all bits
        doLocal = 4;
        nThreads = _datasetSize >> 2;
        break;
      case 4:
        kid = PARALLEL_BITONIC_B16_KERNEL;
        ninc = 4;
        nThreads = _datasetSize >> ninc;
        break;
      case 3:
        kid = PARALLEL_BITONIC_B8_KERNEL;
        ninc = 3;
        nThreads = _datasetSize >> ninc;
        break;
      case 2:
        kid = PARALLEL_BITONIC_B4_KERNEL;
        ninc = 2;
        nThreads = _datasetSize >> ninc;
        break;
      case 1:
        kid = PARALLEL_BITONIC_B2_KERNEL;
        ninc = 1;
        nThreads = _datasetSize >> ninc;
        break;
      default:
        printf("Strategy error!\n");
        break;
      }

      //---- Execute the kernel
      size_t wg;
      clGetKernelWorkGroupInfo(m_BitonicSortKernels[kid], m_OclService->GetCurrentDevice(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wg, 0);
      wg = wg < 256 ? wg : 256;
      wg = wg < nThreads ? wg : nThreads;

      cl_int clStatus = 0;
      unsigned int pId = 0;
      clStatus |= clSetKernelArg(m_BitonicSortKernels[kid], pId++, sizeof(cl_mem), (const void*)&bfKeyVal);
      clStatus |= clSetKernelArg(m_BitonicSortKernels[kid], pId++, sizeof(int), &inc);		// INC passed to kernel
      
      int lenght2 = length << 1;
      clStatus |= clSetKernelArg(m_BitonicSortKernels[kid], pId++, sizeof(int), &lenght2);	// DIR passed to kernel
      
      if (doLocal>0)
        clStatus |= clSetKernelArg(m_BitonicSortKernels[kid], pId++, doLocal * wg * keyValueSize, 0);
      
      clStatus |= clSetKernelArg(m_BitonicSortKernels[kid], pId++, sizeof(unsigned int), (const void*)&_datasetSize);

      size_t globalWs[1] = {nThreads};
      size_t localWs[1] = {wg};
      clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_BitonicSortKernels[kid], 1, 0, globalWs, localWs, 0, NULL, NULL);

      // Sync
      clStatus |= clEnqueueBarrier(m_CommandQue);

      if (ninc < 0) break; // done
      inc >>= ninc;
    }
  }
}



void mitk::OclTriangleSorter::LaunchRadixSort(cl_mem bfKeyVal, cl_uint datasetSize)
{
  // Satish et al. empirically set b = 4. The size of a work-group is in hundreds of
  // work-items, depending on the concrete device and each work-item processes more than one
  // stream element, usually 4, in order to hide latencies.
  cl_int clStatus;
  unsigned int numBlocks = roundUpDiv(datasetSize, m_SortWorkgroupSize * 4);
  unsigned int Ndiv4     = roundUpDiv(datasetSize, 4);

  size_t globalWorkSize[1] = {ToMultipleOf(Ndiv4, m_SortWorkgroupSize)};
  size_t localWorkSize[1]  = {m_SortWorkgroupSize};

  // Store to working variable for swapping
  cl_mem dataA = bfKeyVal;

  // Create a new temporary array of the same size as the input data
  cl_uint * zeros  = new cl_uint[datasetSize*4];
  memset(zeros, 0, datasetSize*4);

  cl_mem dataB = clCreateBuffer(m_Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * datasetSize*4, zeros, &clStatus);
  CHECK_OCL_ERR(clStatus);

  // histogram : 16 values per block
  cl_mem bfRadixHist1 = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint4) * 16 * numBlocks, NULL, &clStatus);
  CHECK_OCL_ERR(clStatus);
  cl_mem bfRadixHist2 = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint4) * 16 * numBlocks, NULL, &clStatus);
  CHECK_OCL_ERR(clStatus);
  cl_mem bfRadixHist1Scan = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(cl_uint4) * 16 * numBlocks, NULL, &clStatus);
  CHECK_OCL_ERR(clStatus);
  
  for (unsigned int bitOffset = 0; bitOffset < m_SortBits; bitOffset += 4)
  {
    //MITK_INFO <<"bitOffset: " <<bitOffset <<" m_SortBits: " <<m_SortBits;

    // 1) Each workgroup sorts its tile by using local memory
    // 2) Create an histogram of d=2^b digits entries
    RadixLocal(datasetSize, dataA, bfRadixHist1, bfRadixHist2, bitOffset);

    LocalHistogram(datasetSize, globalWorkSize, localWorkSize, dataA, bfRadixHist1, bfRadixHist2, bitOffset);
    
    Scan(16 * numBlocks, bfRadixHist1, bfRadixHist1Scan);
    
    RadixPermute(datasetSize, globalWorkSize, localWorkSize, dataA, dataB, bfRadixHist1Scan, bfRadixHist2, bitOffset, numBlocks);
   
    // Swap buffers for the next iteration
    std::swap(dataA, dataB);
  }

  clFinish(m_CommandQue);

  // Check how many times we swapped.. if odd we need to swap once more
  // Otherwise the wrong buffer will get released.
  if (((m_SortBits  / 4) % 2) != 0)
    std::swap(dataA, dataB);
  
  if (bfRadixHist1)
    clReleaseMemObject(bfRadixHist1);

  if (bfRadixHist2)
    clReleaseMemObject(bfRadixHist2);

  if (bfRadixHist1Scan)
    clReleaseMemObject(bfRadixHist1Scan);

  if (dataB)
    clReleaseMemObject(dataB);

  delete zeros;
}

void mitk::OclTriangleSorter::RadixLocal(cl_uint datasetSize, cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset)
{
  cl_int clStatus;
  unsigned int a = 0;

  int workgroupSize = 128;

  unsigned int Ndiv = roundUpDiv(datasetSize, 4); // Each work item handle 4 entries
  size_t global_128[1] = {ToMultipleOf(Ndiv, workgroupSize)};
  size_t local_128[1] = {workgroupSize};

  clStatus  = clSetKernelArg(m_ckRadixLocalSort, a++, sizeof(cl_mem), (const void*)&data);
  clStatus |= clSetKernelArg(m_ckRadixLocalSort, a++, sizeof(int), (const void*)&bitOffset);
  clStatus |= clSetKernelArg(m_ckRadixLocalSort, a++, sizeof(cl_uint), (const void*)&datasetSize);
  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckRadixLocalSort, 1, NULL, global_128, local_128, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);

}

void mitk::OclTriangleSorter::LocalHistogram(cl_uint datasetSize, const size_t* globalWorkSize, const size_t* localWorkSize, cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset)
{
  cl_int clStatus;
  clStatus  = clSetKernelArg(m_ckLocalHistogram, 0, sizeof(cl_mem), (const void*)&data);
  clStatus |= clSetKernelArg(m_ckLocalHistogram, 1, sizeof(int), (const void*)&bitOffset);
  clStatus |= clSetKernelArg(m_ckLocalHistogram, 2, sizeof(cl_mem), (const void*)&hist);
  clStatus |= clSetKernelArg(m_ckLocalHistogram, 3, sizeof(cl_mem), (const void*)&blockHists);
  clStatus |= clSetKernelArg(m_ckLocalHistogram, 4, sizeof(unsigned int), (const void*)&datasetSize);
  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckLocalHistogram, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);

}

void mitk::OclTriangleSorter::RadixPermute(cl_uint datasetSize, const size_t* globalWorkSize, const size_t* localWorkSize, cl_mem dataIn, cl_mem dataOut, cl_mem histScan, cl_mem blockHists, int bitOffset, unsigned int numBlocks)
{
  cl_int clStatus;
  clStatus  = clSetKernelArg(m_ckRadixPermute, 0, sizeof(cl_mem), (const void*)&dataIn);
  clStatus |= clSetKernelArg(m_ckRadixPermute, 1, sizeof(cl_mem), (const void*)&dataOut);
  clStatus |= clSetKernelArg(m_ckRadixPermute, 2, sizeof(cl_mem), (const void*)&histScan);
  clStatus |= clSetKernelArg(m_ckRadixPermute, 3, sizeof(cl_mem), (const void*)&blockHists);
  clStatus |= clSetKernelArg(m_ckRadixPermute, 4, sizeof(int), (const void*)&bitOffset);
  clStatus |= clSetKernelArg(m_ckRadixPermute, 5, sizeof(unsigned int), (const void*)&datasetSize);
  clStatus |= clSetKernelArg(m_ckRadixPermute, 6, sizeof(unsigned int), (const void*)&numBlocks);
  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckRadixPermute, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);

}

void mitk::OclTriangleSorter::Scan(cl_uint datasetSize, cl_mem dataIn, cl_mem dataOut)
{
  cl_int clStatus;
  size_t _workgroupSize = 256;
  //clGetKernelWorkGroupInfo(m_ckScanBlockAnyLength, resources->GetCurrentDevice(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_workgroupSize, 0);

  cl_uint blockSize = datasetSize / _workgroupSize;
  cl_uint B = blockSize * _workgroupSize;
  if ((datasetSize % _workgroupSize) > 0) { blockSize++; };
  size_t localWorkSize = {_workgroupSize};
  size_t globalWorkSize = {ToMultipleOf(datasetSize / blockSize, _workgroupSize)};

  clStatus = clSetKernelArg(m_ckScanBlockAnyLength, 0, _workgroupSize * sizeof(cl_uint), 0);
  CHECK_OCL_ERR(clStatus);
  clStatus |= clSetKernelArg(m_ckScanBlockAnyLength, 1, sizeof(cl_mem), &dataIn);
  CHECK_OCL_ERR(clStatus);
  clStatus |= clSetKernelArg(m_ckScanBlockAnyLength, 2, sizeof(cl_mem), &dataOut);
  CHECK_OCL_ERR(clStatus);
  clStatus |= clSetKernelArg(m_ckScanBlockAnyLength, 3, sizeof(cl_uint), &B);
  CHECK_OCL_ERR(clStatus);
  clStatus |= clSetKernelArg(m_ckScanBlockAnyLength, 4, sizeof(cl_uint), &datasetSize);
  CHECK_OCL_ERR(clStatus);
  clStatus |= clSetKernelArg(m_ckScanBlockAnyLength, 5, sizeof(cl_uint), &blockSize);
  CHECK_OCL_ERR(clStatus);

  clStatus |= clEnqueueNDRangeKernel(m_CommandQue, m_ckScanBlockAnyLength, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
  CHECK_OCL_ERR(clStatus);

  clFinish(m_CommandQue);

}
