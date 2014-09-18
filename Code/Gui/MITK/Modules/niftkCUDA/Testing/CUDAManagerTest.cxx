/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <CUDAManager/CUDAManager.h>
#include <cassert>
#include <mitkTestingMacros.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkDataNode.h>


void Producer(const mitk::StandaloneDataStorage::Pointer& datastorage)
{
  CUDAManager*    cm = CUDAManager::GetInstance();

  //ScopedCUDADevice  sd = cm->ActivateDevice(0);

  cudaError_t   err = cudaSuccess;

  WriteAccessor wa = cm->RequestOutputImage(1920, 1080, 4);

  // the stream is where we queue commands for the gpu. they will be processed in sequence.
  // however, commands on multiple streams may be processed in parallel.
  cudaStream_t stream = cm->GetStream("producer-test");

  err = cudaMemsetAsync(wa.m_DevicePointer, 1, wa.m_SizeInBytes, stream);
  assert(err == cudaSuccess);


  // when our processing has finished it will signal an event object that we use
  // to synchronise later dependent work.
  err = cudaEventRecord(wa.m_ReadyEvent, stream);
  assert(err == cudaSuccess);

  // finalising an output image means that it will become available for read-requests.
  // the actual kernel producing it does not need to have finished yet! that's what we have
  // the ReadyEvent for.
  LightweightCUDAImage lwci = cm->Finalise(wa, stream);

  // the lightweight image, we can stuff into mitk's heavy DataNode and BaseData infrastructure.
  CUDAImage::Pointer    cudaImage(CUDAImage::New());
  cudaImage->SetLightweightCUDAImage(lwci);

  mitk::DataNode::Pointer   node(mitk::DataNode::New());
  node->SetName("cudaimagetest");
  node->SetData(cudaImage);
  datastorage->Add(node);
}


void Consumer(const mitk::StandaloneDataStorage::Pointer& datastorage)
{
  mitk::DataNode::Pointer node = datastorage->GetNamedNode("cudaimagetest");
  assert(node.IsNotNull());

  CUDAImage::Pointer  cudaImage = dynamic_cast<CUDAImage*>(node->GetData());
  assert(cudaImage.IsNotNull());

  LightweightCUDAImage  lwci = cudaImage->GetLightweightCUDAImage();

  CUDAManager*    cm = CUDAManager::GetInstance();

  //ScopedCUDADevice  sd = cm->ActivateDevice(0);

  cudaError_t   err = cudaSuccess;
  //cm->RequestReadAccess(lwci);


  cudaStream_t stream = cm->GetStream("consumer-test");

  WriteAccessor wa = cm->RequestOutputImage(1920, 1080, 4);
}


int CUDAManagerTest(int /*argc*/, char* /*argv*/[])
{
  MITK_TEST_BEGIN("CUDAManagerTest");

  mitk::StandaloneDataStorage::Pointer    datastorage(mitk::StandaloneDataStorage::New());
  Producer(datastorage);
  Consumer(datastorage);

  MITK_TEST_END();

  return EXIT_SUCCESS;
}
