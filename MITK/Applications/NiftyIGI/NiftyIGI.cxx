/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBaseApplication.h>

#include <QVariant>
#include <QStringList>

/// \file NiftyIGI.cxx
/// \brief Main entry point for NiftyIGI application.
int main(int argc, char** argv)
{
  niftk::BaseApplication app(argc, argv);
  app.setApplicationName("NiftyIGI");
  app.setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftyigi");

  // Horrible work-around to force NiftyIGI to preload niftknvapi.
  // I (Matt) seem to have static initialiser problems.
  // When NiftyIGI loads libuk_ac_ucl_cmic_igidatasources.dylib
  // this causes all the data sources in sub-dir niftkIGIDataSources
  // to dynamically load. This includes the niftkNVidiaSDIDataSourceService 
  // which in turn causes niftknvapi.dll to be loaded. But niftnvapi.dll
  // has the rgba2nv12.cu which has a static global texture variable.
  // The construction of this global variable seems to cause a crash when
  // NiftyIGI starts up. THis link:
  // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
  // suggests that, as of CUDA 5.0 and Kepler architectures we should be able
  // to re-write the code to use dynamically constructed texture variables.
  // However, we already have a migration plan to get to CUDA 7.5 using Pankaj's
  // new encoder/decoder. So, the following code is a simple hack to force pre-loading
  // niftknvapi directly when main starts. This avoids loading/unloading during the
  // ctkPluginFramework's plugin discovery phase. 
  //
  // Can be removed when we finally upgrade to CUDA 7.5 on all our machines.
#ifdef NIFTYIGI_USE_NVAPI
  QStringList preLoadLibraries;
  preLoadLibraries << "niftknvapi";
  app.setPreloadLibraries(preLoadLibraries);
#endif

  return app.run();
}
