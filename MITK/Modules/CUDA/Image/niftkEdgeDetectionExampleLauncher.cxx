/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkEdgeDetectionExampleLauncher.h"
#include <niftkCUDAManager.h>
#include <niftkCUDAImage.h>
#include <niftkLightweightCUDAImage.h>
#include <Example/niftkEdgeDetectionKernel.h>
#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
void EdgeDetectionExampleLauncher(mitk::DataStorage* dataStorage,
                                  mitk::DataNode* node,
                                  const mitk::BaseRenderer* renderer)
{
  mitk::DataNode::Pointer fbonode = dataStorage->GetNamedNode("vl-framebuffer");
  if (fbonode.IsNotNull())
  {
    niftk::CUDAImage::Pointer  cudaImg = dynamic_cast<niftk::CUDAImage*>(fbonode->GetData());
    if (cudaImg.IsNotNull())
    {
      niftk::LightweightCUDAImage    inputLWCI = cudaImg->GetLightweightCUDAImage();
      if (inputLWCI.GetId() != 0)
      {
        niftk::CUDAManager*    cudamanager = niftk::CUDAManager::GetInstance();
        cudaStream_t    mystream    = cudamanager->GetStream("vl example");
        ReadAccessor    inputRA     = cudamanager->RequestReadAccess(inputLWCI);
        WriteAccessor   outputWA    = cudamanager->RequestOutputImage(inputLWCI.GetWidth(), inputLWCI.GetHeight(), 4);

        // this is important: it will make our kernel call below wait for vl to finish the fbo copy.
        cudaError_t err = cudaStreamWaitEvent(mystream, inputRA.m_ReadyEvent, 0);
        if (err != cudaSuccess)
        {
          // flood the log
          MITK_WARN << "cudaStreamWaitEvent failed with error code " << err;
        }

        RunEdgeDetectionKernel(
          (char*) outputWA.m_DevicePointer, outputWA.m_BytePitch,
          (const char*) inputRA.m_DevicePointer, inputRA.m_BytePitch,
          inputLWCI.GetWidth(), inputLWCI.GetHeight(), mystream);

        // finalise() will queue an event-signal on our stream for us, so that future processing steps can
        // synchronise, just like we did above before starting our kernel.
        niftk::LightweightCUDAImage outputLWCI  = cudamanager->FinaliseAndAutorelease(outputWA, inputRA, mystream);
        mitk::DataNode::Pointer     node        = dataStorage->GetNamedNode("vl-cuda-interop sample");
        bool                        isNewNode   = false;
        if (node.IsNull())
        {
          isNewNode = true;
          node = mitk::DataNode::New();
          node->SetName("vl-cuda-interop sample");
        }
        niftk::CUDAImage::Pointer  img = dynamic_cast<niftk::CUDAImage*>(node->GetData());
        if (img.IsNull())
          img = niftk::CUDAImage::New();
        img->SetLightweightCUDAImage(outputLWCI);
        node->SetData(img);
        if (isNewNode)
          dataStorage->Add(node);
      }
    }
  }
}

} // end namespace
