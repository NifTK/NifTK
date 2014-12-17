#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// 
#define PI 3.1415926535897932384626433832795f

//------------------------------------------------------------
// Purpose :
// ---------
//
// Algorithm :
// -----------
// Radix sort algorithm for key-value pairs. This work is based on the Blelloch
// paper and optimized with the technique described in the Satish/Harris/Garland paper.
//
// References :
// ------------
// Designing Efficient Sorting Algorithms for Manycore GPUs. Nadathur Satish, Mark Harris, Michael Garland. http://mgarland.org/files/papers/gpusort-ipdps09.pdf
// http://www.sci.utah.edu/~csilva/papers/cgf.pdf
// Radix Sort For Vector Multiprocessors, Marco Zagha and Guy E. Blelloch
//------------------------------------------------------------

// To do : visiting logic and multi-scan.

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define WGZ 32
#define WGZ_x2 (WGZ*2)
#define WGZ_x3 (WGZ*3)
#define WGZ_x4 (WGZ*4)
#define WGZ_1 (WGZ-1)
#define WGZ_2 (WGZ-2)
#define WGZ_x2_1 (WGZ_x2-1)
#define WGZ_x3_1 (WGZ_x3-1)
#define WGZ_x4_1 (WGZ_x4-1)
#define WGZ_x4_2 (WGZ_x4-2)

//// CASE UINT2
//#define KV_TYPE       uint2
//#define MAX_KV_TYPE ((uint2)(0x7FFFFFFF, 0xFFFFFFFF))

//// CASE UINT4
#define KV_TYPE       uint4
#define MAX_KV_TYPE ((uint4)(0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF))

//#define MAX_KV_TYPE2 ((uint2)(0x7FFFFFFF, 0xFFFFFFFF))
//#define MAX_KV_TYPE  ((uint) (0x7FFFFFFF))

//#ifdef KEYS_ONLY
//#define KEY(DATA) (DATA)
//#else
#define KEY(DATA) (DATA.x)
//#endif

#define EXTRACT_KEY_BIT(VALUE,BIT) ((KEY(VALUE)>>BIT)&0x1)
#define EXTRACT_KEY_4BITS(VALUE,BIT) ((KEY(VALUE)>>BIT)&0xF)

// Because our workgroup size = SIMT size, we use the natural synchronization provided by SIMT.
// So, we don't need any barrier to synchronize
#define BARRIER_LOCAL barrier(CLK_LOCAL_MEM_FENCE)
#define SIMT 32
#define SIMT_1 (SIMT-1)
#define SIMT_2 (SIMT-2)
#define COMPUTE_UNITS 4
#define TPG (COMPUTE_UNITS * SIMT)
#define TPG_2 (TPG-2)

//------------------------------------------------------------
// exclusive_scan_128
//
// Purpose : Do a scan of 128 elements in once.
//------------------------------------------------------------

inline
  uint4 inclusive_scan_128(volatile __local uint* localBuffer, const uint tid, uint block, uint lane, uint4 initialValue, __local uint* bitsOnCount)
{
  //---- scan : 4 bits
  uint4 localBits = initialValue;
  localBits.y += localBits.x;
  localBits.z += localBits.y;
  localBits.w += localBits.z;

  //---- scan the last 4x32 bits (The sum in the previous scan)

  // The following is the same as 2 * SIMT_SIZE * simtId + threadInSIMT =
  // 64*(threadIdx.x >> 5) + (threadIdx.x & (:WARP_SIZE - 1))
  //int localId = get_local_id(0);
  //int idx = 2 * localId - (localId & (WARP_SIZE - 1));
  //uint tid2 = 2 * tid - lane;

  uint tid2 = block * 2 * SIMT + lane;

  localBuffer[tid2] = 0;
  tid2 += SIMT;
  localBuffer[tid2] = localBits.w;

  localBuffer[tid2] += localBuffer[tid2 - 1];
  localBuffer[tid2] += localBuffer[tid2 - 2];
  localBuffer[tid2] += localBuffer[tid2 - 4];
  localBuffer[tid2] += localBuffer[tid2 - 8];
  localBuffer[tid2] += localBuffer[tid2 - 16];

  //---- Add the sum to create a scan of 128 bits
  return localBits + localBuffer[tid2 - 1];
}

inline
  uint4 exclusive_scan_512(volatile __local uint* localBuffer, const uint tid, uint4 initialValue, __local uint* bitsOnCount)
{
  uint lane = tid & SIMT_1;
  uint block = tid >> 5;

  uint4 localBits = inclusive_scan_128(localBuffer, tid, block, lane, initialValue, bitsOnCount);

  barrier(CLK_LOCAL_MEM_FENCE);

  //---- Scan 512
  if (lane > SIMT_2)
  {
    localBuffer[block] = 0;
    localBuffer[4 + block] = localBits.w;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Use the SIMT capabilities
  if (tid < 4)
  {
    uint tid2 = tid + 4;
    localBuffer[tid2] += localBuffer[tid2 - 1];
    localBuffer[tid2] += localBuffer[tid2 - 2];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Add the sum
  localBits += localBuffer[block + 4 - 1];

  // Total number of '1' in the array, retreived from the inclusive scan
  if (tid > TPG_2)
    bitsOnCount[0] = localBits.w;

  // To exclusive scan
  return localBits - initialValue;
}

//------------------------------------------------------------
// kernel ckRadixLocalSort
//
// Purpose :
// 1) Each workgroup sorts its tile by using local memory
// 2) Create an histogram of d=2^b digits entries
//------------------------------------------------------------

__kernel
  void ckRadixLocalSort(
  //__local KV_TYPE* localDataOLD,
  __global KV_TYPE* data,
  const int bitOffset,
  const int N)
{
  const uint tid = (uint)get_local_id(0);
  const uint4 tid4 = (const uint4)(tid << 2) + (const uint4)(0,1,2,3);
  const uint4 gid4 = (const uint4)(get_global_id(0) << 2) + (const uint4)(0,1,2,3);

  // Local memory
  __local KV_TYPE localDataArray[TPG*4*2]; // Faster than using it as a parameter !!!
  __local KV_TYPE* localData = localDataArray;
  __local KV_TYPE* localTemp = localData + TPG * 4;
  __local uint bitsOnCount[1];
  __local uint localBuffer[TPG*2];

  // Each thread copies 4 (Cell,Tri) pairs into local memory
  if (gid4.x < N)
    localData[tid4.x] = data[gid4.x];
  else
    localData[tid4.x] = MAX_KV_TYPE;

  if (gid4.y < N)
    localData[tid4.y] = data[gid4.y];
  else
    localData[tid4.y] = MAX_KV_TYPE;

  if (gid4.z < N)
    localData[tid4.z] = data[gid4.z];
  else
    localData[tid4.z] = MAX_KV_TYPE;

  if (gid4.w < N)
    localData[tid4.w] = data[gid4.w];
  else
    localData[tid4.w] = MAX_KV_TYPE;

  //-------- 1) 4 x local 1-bit split
#pragma unroll
  for(uint shift = bitOffset; shift < (bitOffset+4); shift++) // Radix 4
  {
    //barrier(CLK_LOCAL_MEM_FENCE);

    //---- Setup the array of 4 bits (of level shift)
    // Create the '1s' array as explained at : http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
    // In fact we simply inverse the bits
    // Local copy and bits extraction
    uint4 flags;
    flags.x = ! EXTRACT_KEY_BIT(localData[tid4.x], shift);
    flags.y = ! EXTRACT_KEY_BIT(localData[tid4.y], shift);
    flags.z = ! EXTRACT_KEY_BIT(localData[tid4.z], shift);
    flags.w = ! EXTRACT_KEY_BIT(localData[tid4.w], shift);

    //---- Do a scan of the 128 bits and retreive the total number of '1' in 'bitsOnCount'
    uint4 localBitsScan = exclusive_scan_512(localBuffer, tid, flags, bitsOnCount);

    // Waiting for 'bitsOnCount'
    barrier(CLK_LOCAL_MEM_FENCE);

    //---- Relocate to the right position
    uint4 offset = (1-flags) * ((uint4)(bitsOnCount[0]) + tid4 - localBitsScan) + flags * localBitsScan;
    localTemp[offset.x] = localData[tid4.x];
    localTemp[offset.y] = localData[tid4.y];
    localTemp[offset.z] = localData[tid4.z];
    localTemp[offset.w] = localData[tid4.w];

    // Wait before swapping the 'local' buffer pointers. They are shared by the whole local context
    barrier(CLK_LOCAL_MEM_FENCE);

    // Swap the buffer pointers
    __local KV_TYPE* swBuf = localData;
    localData = localTemp;
    localTemp = swBuf;
  }

  //barrier(CLK_LOCAL_MEM_FENCE);

  // Write sorted data back to global memory
  if (gid4.x < N) data[gid4.x] = localData[tid4.x];
  if (gid4.y < N) data[gid4.y] = localData[tid4.y];
  if (gid4.z < N) data[gid4.z] = localData[tid4.z];
  if (gid4.w < N) data[gid4.w] = localData[tid4.w];
}

//------------------------------------------------------------
// kernel ckLocalHistogram
//
// Purpose :
//
// Given an array of 'locally sorted' blocks of keys (according to a 4-bit radix), for each
// block we counts the number of keys that fall into each radix, and finds the starting
// offset of each radix in the block.
//
// It then writes the radix counts to the 'radixCount' array, and the starting offsets to the 'radixOffsets' array.
//------------------------------------------------------------

__kernel
  void ckLocalHistogram(__global KV_TYPE* data, const int bitOffset, __global int* radixCount, __global int* radixOffsets, const int N)
{
  const int tid = (int)get_local_id(0);
  const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
  const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);
  const int blockId = (int)get_group_id(0);

  __local uint localData[WGZ_x4];

  // Contains the 2 histograms (16 values)
  __local int localHistStart[16]; // 2^4 = 16
  __local int localHistEnd[16];

  //---- Extract the radix
  localData[tid4.x] = (gid4.x < N) ? EXTRACT_KEY_4BITS(data[gid4.x], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
  localData[tid4.y] = (gid4.y < N) ? EXTRACT_KEY_4BITS(data[gid4.y], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
  localData[tid4.z] = (gid4.z < N) ? EXTRACT_KEY_4BITS(data[gid4.z], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
  localData[tid4.w] = (gid4.w < N) ? EXTRACT_KEY_4BITS(data[gid4.w], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);

  //---- Create the histogram

  barrier(CLK_LOCAL_MEM_FENCE);

  // Reset the local histogram
  if (tid < 16)
  {
    localHistStart[tid] = 0;
    localHistEnd[tid] = -1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Finds the position where the localData entries differ and stores it in 'start index' (localHistStart) for each radix.
  // This way, for the first 'instance' of a radix, we store its index.
  // We also store where each radix ends in 'localHistEnd'.
  //
  // And so, if we use end-start+1 we have the histogram value to store.

  if (tid4.x > 0 && localData[tid4.x] != localData[tid4.x-1])
  {
    localHistStart[localData[tid4.x]] = tid4.x;
    localHistEnd[localData[tid4.x-1]] = tid4.x - 1;
  }
  //BARRIER_LOCAL;

  if (localData[tid4.y] != localData[tid4.x])
  {
    localHistStart[localData[tid4.y]] = tid4.y;
    localHistEnd[localData[tid4.x]] = tid4.x;
  }
  //BARRIER_LOCAL;

  if (localData[tid4.z] != localData[tid4.y])
  {
    localHistStart[localData[tid4.z]] = tid4.z;
    localHistEnd[localData[tid4.y]] = tid4.y;
  }
  //BARRIER_LOCAL;

  if (localData[tid4.w] != localData[tid4.z])
  {
    localHistStart[localData[tid4.w]] = tid4.w;
    localHistEnd[localData[tid4.z]] = tid4.z;
  }
  //BARRIER_LOCAL;

  // First and last histogram values
  if (tid < 1)
  {
    localHistStart[localData[0]] = 0;
    localHistEnd[localData[WGZ_x4-1]] = WGZ_x4 - 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  //---- Write histogram to global memory
  // Write the 16 histogram values to the global buffers
  if (tid < 16)
  {
    radixCount[tid * get_num_groups(0) + blockId] = localHistEnd[tid] - localHistStart[tid] + 1;
    radixOffsets[(blockId << 4) + tid] = localHistStart[tid];
  }
}

//------------------------------------------------------------
// kernel ckRadixPermute
//
// Purpose : Prefix sum results are used to scatter each work-group's elements to their correct position.
//------------------------------------------------------------

__kernel
  void kernel ckRadixPermute(
  __global const KV_TYPE* dataIn, // size 4*4 int2s per block
  __global KV_TYPE* dataOut, // size 4*4 int2s per block
  __global const int* histSum, // size 16 per block (64 B)
  __global const int* blockHists, // size 16 int2s per block (64 B)
  const int bitOffset, // k*4, k=0..7
  const int N,
  const int numBlocks)
{
  const int tid = get_local_id(0);
  const int groupId = get_group_id(0);
  const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
  const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);


  __local int sharedHistSum[16];
  __local int localHistStart[16];

  // Fetch per-block KV_TYPE histogram and int histogram sums
  if (tid < 16)
  {
    sharedHistSum[tid] = histSum[tid * numBlocks + groupId];
    localHistStart[tid] = blockHists[(groupId << 4) + tid]; // groupId * 32 + tid
  }

  BARRIER_LOCAL;

  KV_TYPE myData;
  int myShiftedKeys;
  int finalOffset;

  if (gid4.x < N)
    myData = dataIn[gid4.x];
  else
     myData = MAX_KV_TYPE;

  myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
  finalOffset = tid4.x - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
  if (finalOffset < N) dataOut[finalOffset] = myData;

  if (gid4.y < N)
    myData = dataIn[gid4.y];
  else
    myData = MAX_KV_TYPE;

  myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
  finalOffset = tid4.y - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
  if (finalOffset < N) dataOut[finalOffset] = myData;

  if (gid4.z < N)
    myData = dataIn[gid4.z];
  else
    myData = MAX_KV_TYPE;

  myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
  finalOffset = tid4.z - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
  if (finalOffset < N) dataOut[finalOffset] = myData;

  if (gid4.w < N)
    myData = dataIn[gid4.w];
  else
    myData = MAX_KV_TYPE;

  myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
  finalOffset = tid4.w - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
  if (finalOffset < N) dataOut[finalOffset] = myData;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


//------------------------------------------------------------
// Purpose :
// ---------
// Prefix sum or prefix scan is an operation where each output element contains the sum of all input elements preceding it.
//
// Algorithm :
// -----------
// The parallel prefix sum has two principal parts, the reduce phase (also known as the up-sweep phase) and the down-sweep phase.
//
// In the up-sweep reduction phase we traverse the computation tree from bottom to top, computing partial sums.
// After this phase, the last element of the array contains the total sum.
//
// During the down-sweep phase, we traverse the tree from the root and use the partial sums to build the scan in place.
//
// Because the scan pictured is an exclusive sum, a zero is inserted into the last element before the start of the down-sweep phase.
// This zero is then propagated back to the first element.
//
// In our implementation, each compute unit loads and sums up two elements (for the deepest depth). Each subsequent depth during the up-sweep
// phase is processed by half of the compute units from the deeper level and the other way around for the down-sweep phase.
//
// In order to be able to scan large arrays, i.e. arrays that have many more elements than the maximum size of a work-group, the prefix sum has to be decomposed.
// Each work-group computes the prefix scan of its sub-range and outputs a single number representing the sum of all elements in its sub-range.
// The workgroup sums are scanned using exactly the same algorithm.
// When the number of work-group results reaches the size of a work-group, the process is reversed and the work-group sums are
// propagated to the sub-ranges, where each work-group adds the incoming sum to all its elements, thus producing the final scanned array.
//
// References :
// ------------
// http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=1041
//
// To read :
// http://developer.apple.com/library/mac/#samplecode/OpenCL_Parallel_Prefix_Sum_Example/Listings/scan_kernel_cl.html#//apple_ref/doc/uid/DTS40008183-scan_kernel_cl-DontLinkElementID_5
// http://developer.apple.com/library/mac/#samplecode/OpenCL_Parallel_Reduction_Example/Listings/reduce_int4_kernel_cl.html
//------------------------------------------------------------

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define T uint
#define OPERATOR_APPLY(A,B) A+B
#define OPERATOR_IDENTITY 0

//#define VOLATILE volatile
#define VOLATILE

//------------------------------------------------------------
// kernel__scanInter
//
// Purpose : do a scan on a chunck of data.
//------------------------------------------------------------

inline T scan_simt_exclusive(__local VOLATILE T* input, size_t idx, const uint lane)
{
  if (lane > 0 ) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
  if (lane > 1 ) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
  if (lane > 3 ) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
  if (lane > 7 ) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
  if (lane > 15) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);

  return (lane > 0) ? input[idx-1] : OPERATOR_IDENTITY;
}

inline T scan_simt_inclusive(__local VOLATILE T* input, size_t idx, const uint lane)
{
  if (lane > 0 ) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
  if (lane > 1 ) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
  if (lane > 3 ) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
  if (lane > 7 ) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
  if (lane > 15) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);

  return input[idx];
}

inline T scan_workgroup_exclusive(__local T* localBuf, const uint idx, const uint lane, const uint simt_bid)
{
  // Step 1: Intra-warp scan in each warp
  T val = scan_simt_exclusive(localBuf, idx, lane);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Step 2: Collect per-warp partial results (the sum)
  if (lane > 30) localBuf[simt_bid] = localBuf[idx];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Step 3: Use 1st warp to scan per-warp results
  if (simt_bid < 1) scan_simt_inclusive(localBuf, idx, lane);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Step 4: Accumulate results from Steps 1 and 3
  if (simt_bid > 0) val = OPERATOR_APPLY(localBuf[simt_bid-1], val);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Step 5: Write and return the final result
  localBuf[idx] = val;
  barrier(CLK_LOCAL_MEM_FENCE);

  return val;
}

__kernel
  void ckScanBlockAnyLength(
  __local T* localBuf,
  __global T* dataSetIn,
  __global T* dataSetOut,
  const uint B,
  uint size,
  const uint passesCount
  )
{
  size_t idx = get_local_id(0);
  const uint bidx = get_group_id(0);
  const uint TC = get_local_size(0);

  const uint lane = idx & 31;
  const uint simt_bid = idx >> 5;

  T reduceValue = OPERATOR_IDENTITY;

  //#pragma unroll 4
  for (uint i = 0; i < passesCount; ++i)
  {
    const uint offset    = i * TC + (bidx * B);
    const uint offsetIdx = offset + idx;

    if (offsetIdx > size-1)
      return;

    // Step 1: Read TC elements from global (off-chip) memory to local memory (on-chip)
    localBuf[idx] = dataSetIn[offsetIdx];
    T input       = localBuf[idx];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Perform scan on TC elements
    T val = scan_workgroup_exclusive(localBuf, idx, lane, simt_bid);

    // Step 3: Propagate reduced result from previous block of TC elements
    val = OPERATOR_APPLY(val, reduceValue);

    // Step 4: Write out data to global memory
    dataSetOut[offsetIdx] = val;

    // Step 5: Choose reduced value for next iteration
    if (idx == (TC-1))
    {
      //localBuf[idx] = (Kind == exclusive) ? OPERATOR_APPLY(input, val) : val;
      localBuf[idx] = OPERATOR_APPLY(input, val);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    reduceValue = localBuf[TC-1];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Adapted from the Eric Bainville code.
//
// Copyright (c) Eric Bainville - June 2011
// http://www.bealto.com/gpu-sorting_intro.html

#define KV_TYPE2 uint2
#define getKey(a) ((a).x)
#define getValue(a) ((a).y)
#define makeData(k,v) ((uint2)((k),(v)))

#ifndef BLOCK_FACTOR
#define BLOCK_FACTOR 1
#endif

#define ORDER(a,b) { bool swap = reverse ^ (getKey(a)<getKey(b)); KV_TYPE2 auxa = a; KV_TYPE2 auxb = b; a = (swap)?auxb:auxa; b = (swap)?auxa:auxb; }

// N/2 threads
__kernel
  void ckParallelBitonic_B2(__global KV_TYPE2* data, int inc, int dir, uint datasetSize)
{
  int t = get_global_id(0); // thread index
  int low = t & (inc - 1); // low order bits (below INC)
  int i = (t<<1) - low; // insert 0 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data += i; // translate to first value

  // Load
  KV_TYPE2 x0 = data[  0];
  KV_TYPE2 x1 = data[inc];

  // Sort
  ORDER(x0,x1)

    // Store
    data[0  ] = x0;
  data[inc] = x1;
}

// N/4 threads
__kernel
  void ckParallelBitonic_B4(__global KV_TYPE2 * data,int inc,int dir, uint datasetSize)
{
  inc >>= 1;
  int t = get_global_id(0); // thread index
  int low = t & (inc - 1); // low order bits (below INC)
  int i = ((t - low) << 2) + low; // insert 00 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data += i; // translate to first value

  // Load
  KV_TYPE2 x0 = data[    0];
  KV_TYPE2 x1 = data[  inc];
  KV_TYPE2 x2 = data[2*inc];
  KV_TYPE2 x3 = data[3*inc];

  // Sort
  ORDER(x0,x2)
    ORDER(x1,x3)
    ORDER(x0,x1)
    ORDER(x2,x3)

    // Store
    data[    0] = x0;
  data[  inc] = x1;
  data[2*inc] = x2;
  data[3*inc] = x3;
}

#define ORDERV(x,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b])); KV_TYPE2 auxa = x[a]; KV_TYPE2 auxb = x[b]; x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb; }
#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }

// N/8 threads
__kernel
  void ckParallelBitonic_B8(__global KV_TYPE2 * data,int inc,int dir, uint datasetSize)
{
  inc >>= 2;
  int t = get_global_id(0); // thread index
  int low = t & (inc - 1); // low order bits (below INC)
  int i = ((t - low) << 3) + low; // insert 000 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data += i; // translate to first value

  // Load
  KV_TYPE2 x[8];
  for (int k=0;k<8;k++) x[k] = data[k*inc];

  // Sort
  B8V(x,0)

    // Store
    for (int k=0;k<8;k++) data[k*inc] = x[k];
}

// N/16 threads
__kernel
  void ckParallelBitonic_B16(__global KV_TYPE2 * data,int inc,int dir, uint datasetSize)
{
  inc >>= 3;
  int t = get_global_id(0); // thread index
  int low = t & (inc - 1); // low order bits (below INC)
  int i = ((t - low) << 4) + low; // insert 0000 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data += i; // translate to first value

  // Load
  KV_TYPE2 x[16];
  for (int k=0;k<16;k++) x[k] = data[k*inc];

  // Sort
  B16V(x,0)

    // Store
    for (int k=0;k<16;k++) data[k*inc] = x[k];
}

__kernel
  void ckParallelBitonic_C4(__global KV_TYPE2 * data, int inc0, int dir, __local KV_TYPE2* aux, uint datasetSize)
{
  int t = get_global_id(0); // thread index
  int wgBits = 4 * get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
  int inc,low,i;
  bool reverse;
  KV_TYPE2 x[4];

  // First iteration, global input, local output
  inc = inc0>>1;
  low = t & (inc - 1); // low order bits (below INC)
  i = ((t - low) << 2) + low; // insert 00 at position INC
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k = 0; k < 4; k++) x[k] = data[i+k*inc];
  B4V(x,0);
  for (int k = 0; k < 4; k++) aux[(i+k*inc) & wgBits] = x[k];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Internal iterations, local input and output
  for(;inc > 1; inc >>= 2)
  {
    low = t & (inc - 1); // low order bits (below INC)
    i = ((t - low) << 2) + low; // insert 00 at position INC
    reverse = ((dir & i) == 0); // asc/desc order
    for (int k=0;k<4;k++) x[k] = aux[(i+k*inc) & wgBits];
    B4V(x,0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k=0;k<4;k++) aux[(i+k*inc) & wgBits] = x[k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Final iteration, local input, global output, INC=1
  i = t << 2;
  reverse = ((dir & i) == 0); // asc/desc order
  for (int k = 0;k < 4; k++) x[k] = aux[(i+k) & wgBits];
  B4V(x,0);
  for (int k = 0;k < 4; k++) data[i+k] = x[k];
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__kernel
  void ckTransformVertexAndComputeDistance(
  __global float * vertexDistances,
  __global float * vertexBuf,
  __global float * transform,
  const  float4    viewPoint,
  const  uint      numOfVertices
  )
{
  uint idx = get_global_id(0);

  if (idx >= numOfVertices)
    return;

  viewPoint.w = 0.0f;
  float4 transformedVertexCoords;
  float4 vertexCoords;
  vertexCoords.x = vertexBuf[idx*3+0];
  vertexCoords.y = vertexBuf[idx*3+1];
  vertexCoords.z = vertexBuf[idx*3+2];
  vertexCoords.w = 0.0f;

  float value = 0.0f;
  value += transform[0] * vertexCoords.x;
  value += transform[1] * vertexCoords.y;
  value += transform[2] * vertexCoords.z;
  value += transform[3] * vertexCoords.w;
  transformedVertexCoords.x = value;

  value = 0.0f;
  value += transform[4] * vertexCoords.x;
  value += transform[5] * vertexCoords.y;
  value += transform[6] * vertexCoords.z;
  value += transform[7] * vertexCoords.w;
  transformedVertexCoords.y = value;

  value = 0.0f;
  value += transform[8] * vertexCoords.x;
  value += transform[9] * vertexCoords.y;
  value += transform[10] * vertexCoords.z;
  value += transform[11] * vertexCoords.w;
  transformedVertexCoords.z = value;

  value = 0.0f;
  value += transform[12] * vertexCoords.x;
  value += transform[13] * vertexCoords.y;
  value += transform[14] * vertexCoords.z;
  value += transform[15] * vertexCoords.w;
  transformedVertexCoords.w = value;

  transformedVertexCoords.x /= transformedVertexCoords.w;
  transformedVertexCoords.y /= transformedVertexCoords.w;
  transformedVertexCoords.z /= transformedVertexCoords.w;
  
  //transformedVertexCoords.x += 1.0f;
  //transformedVertexCoords.y += 1.0f;
  transformedVertexCoords.z += 1.0f;
  transformedVertexCoords.w = 0.0f;

  float4 zero = 0.0f;
  vertexDistances[idx] = fast_distance(zero, transformedVertexCoords);
  //vertexBuf[idx*3+0] = vertexCoords.x;
  //vertexBuf[idx*3+1] = vertexCoords.y;
  //vertexBuf[idx*3+2] = vertexCoords.z;
}

inline unsigned int FloatFlip(float f)
{
  int value = as_int(f) >> 31;
  unsigned int mask = -value | 0x80000000;
  return as_int(f) ^ mask;
}

inline float IFloatFlip(unsigned int f)
{
  unsigned int mask = ((f >> 31) - 1) | 0x80000000;
  return as_float(f ^ mask);
}

__kernel
  void ckComputeTriangleDistances(
  __global uint   * indexBufWithDist,
  __global float  * vertexDist,
  __global uint   * indexBuf,
  const  uint       numOfVertices,
  const  uint       numOfTriangles
  )
{
  size_t idx = get_global_id(0);

  if (idx >= numOfTriangles)
    return;

  uint3 vertIndices;
  vertIndices.x = indexBuf[idx*3 +0];
  vertIndices.y = indexBuf[idx*3 +1];
  vertIndices.z = indexBuf[idx*3 +2];

  float triDist = (vertexDist[vertIndices.x] + vertexDist[vertIndices.y] + vertexDist[vertIndices.z]) / 3.0f;

  indexBufWithDist[idx*4 +0] = FloatFlip(triDist);
  indexBufWithDist[idx*4 +1] = vertIndices.x;
  indexBufWithDist[idx*4 +2] = vertIndices.y;
  indexBufWithDist[idx*4 +3] = vertIndices.z;
}

__kernel
  void ckCopyAndUpdateIndicesOrig(
  __global uint4  * input,
  __global uint4  * output,
  const  uint     size,
  const  uint     offset
  )
{
  size_t idx = get_global_id(0);

  if (idx >= size)
    return;

  output[idx + offset].x = input[idx].x;
  output[idx + offset].y = input[idx].y + offset; 
  output[idx + offset].z = input[idx].z + offset; 
  output[idx + offset].w = input[idx].w + offset; 
}

__kernel
  void ckCopyAndUpdateIndices(
  __global uint4  * input,
  __global uint4  * output,
  __global uint2  * outputForSorting,
  const  uint     size,
  const  uint     offset
  )
{
  size_t idx = get_global_id(0);

  if (idx >= size)
    return;

  output[idx + offset].x = input[idx].x;
  output[idx + offset].y = input[idx].y + offset;
  output[idx + offset].z = input[idx].z + offset;
  output[idx + offset].w = input[idx].w + offset;

  outputForSorting[idx + offset].x = input[idx].x;
  outputForSorting[idx + offset].y = idx + offset;
}

__kernel
  void ckCopyIndicesOnly(
  __global uint4  * input,
  __global uint   * output,
    const  uint     size
  )
{
  size_t idx = get_global_id(0);

  if (idx >= size)
    return;

  size_t flippedIndex = ((size-1) -idx)*3;
  //size_t flippedIndex = idx * 3;
  output[flippedIndex +0] = input[idx].y; 
  output[flippedIndex +1] = input[idx].z; 
  output[flippedIndex +2] = input[idx].w; 
}

__kernel
  void ckCopyIndicesWithDist(
  __global uint4  * input,
  __global uint   * output,
  __global float  * outputDist,
    const  uint     size,
    const  uint     sizeDist
  )
{
  size_t idx = get_global_id(0);

  if (idx >= size)
    return;

  size_t flippedIndex = ((size-1) -idx)*3;
  //size_t flippedIndex = idx * 3;
  output[flippedIndex +0] = input[idx].y; 
  output[flippedIndex +1] = input[idx].z; 
  output[flippedIndex +2] = input[idx].w; 

  outputDist[(size-1) -idx] = input[idx].x; 
}






__kernel
  void ckTest(__global float * dataIn, __global float * dataOut, uint datasetSize)
{
  int t = get_global_id(0); // thread index
  if (t >= datasetSize)
    return;

  dataOut[t] = dataIn[0] + t;
}