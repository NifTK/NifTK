/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <CUDAManager/niftkCUDAManager.h>
#include <cassert>
#include <mitkTestingMacros.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkDataNode.h>

namespace niftk
{

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

  LightweightCUDAImage  lwciInput = cudaImage->GetLightweightCUDAImage();

  CUDAManager*    cm = CUDAManager::GetInstance();

  //ScopedCUDADevice  sd = cm->ActivateDevice(0);

  cudaStream_t stream = cm->GetStream("consumer-test");

  ReadAccessor ra = cm->RequestReadAccess(lwciInput);
  WriteAccessor wa = cm->RequestOutputImage(1920, 1080, 4);

  cudaError_t err = cudaSuccess;
  // make sure input image has actually finished processing before we start our work.
  // (remember: streams can run in parallel.)
  err = cudaStreamWaitEvent(stream, ra.m_ReadyEvent, 0);
  assert(err == cudaSuccess);

  err = cudaMemcpyAsync(wa.m_DevicePointer, ra.m_DevicePointer, 1, cudaMemcpyDeviceToDevice, stream);

  // Finalise*() does an automatic ready-event queueing so no need to do this here.
  // (Producer() above did it only to illustrate how).
  LightweightCUDAImage lwciOutput = cm->FinaliseAndAutorelease(wa, ra, stream);

  // replace image on node.
  cudaImage->SetLightweightCUDAImage(lwciOutput);
}


void AnotherConsumer(const mitk::StandaloneDataStorage::Pointer& datastorage)
{
  mitk::DataNode::Pointer node = datastorage->GetNamedNode("cudaimagetest");
  assert(node.IsNotNull());
  CUDAImage::Pointer  cudaImage = dynamic_cast<CUDAImage*>(node->GetData());
  assert(cudaImage.IsNotNull());

  LightweightCUDAImage  lwciInput = cudaImage->GetLightweightCUDAImage();
  CUDAManager*          cm = CUDAManager::GetInstance();

  cudaStream_t stream = cm->GetStream("anotherconsumer-test");

  ReadAccessor ra = cm->RequestReadAccess(lwciInput);
  cudaError_t err = cudaSuccess;
  err = cudaStreamWaitEvent(stream, ra.m_ReadyEvent, 0);
  assert(err == cudaSuccess);

  // no output this time

  cm->Autorelease(ra, stream);
}


void WaitForResult(const mitk::StandaloneDataStorage::Pointer& datastorage)
{
  mitk::DataNode::Pointer   node = datastorage->GetNamedNode("cudaimagetest");
  assert(node.IsNotNull());
  CUDAImage::Pointer        cudaImage = dynamic_cast<CUDAImage*>(node->GetData());
  assert(cudaImage.IsNotNull());
  LightweightCUDAImage      lwciInput = cudaImage->GetLightweightCUDAImage();

  cudaError_t err = cudaSuccess;
  err = cudaEventSynchronize(lwciInput.GetReadyEvent());
  assert(err == cudaSuccess);
}


void RefcountTest(const mitk::StandaloneDataStorage::Pointer& datastorage)
{
  CUDAManager*          cm = CUDAManager::GetInstance();

  {
    WriteAccessor         wa = cm->RequestOutputImage(1, 1, 4);
    cudaStream_t          stream = cm->GetStream("refcount-test");
    LightweightCUDAImage  lwci = cm->Finalise(wa, stream);

    CUDAImage::Pointer    cudaImage(CUDAImage::New());
    cudaImage->SetLightweightCUDAImage(lwci);

    mitk::DataNode::Pointer   node(mitk::DataNode::New());
    node->SetName("cudaimagetest");
    node->SetData(cudaImage);
    datastorage->Add(node);
  }

  // at this point, the above lwci should have a refcount of 1
}

} // end namespace

int niftkCUDAManagerTest(int /*argc*/, char* /*argv*/[])
{
  MITK_TEST_BEGIN("niftkCUDAManagerTest");

  mitk::StandaloneDataStorage::Pointer    datastorage(mitk::StandaloneDataStorage::New());
  niftk::RefcountTest(datastorage);
  niftk::Producer(datastorage);
  niftk::Consumer(datastorage);
  niftk::AnotherConsumer(datastorage);
  niftk::WaitForResult(datastorage);
  MITK_TEST_END();

  return EXIT_SUCCESS;
}
