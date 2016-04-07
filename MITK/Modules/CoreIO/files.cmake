#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(H_FILES
)

set(CPP_FILES
  Internal/niftkCoordinateAxesDataReaderService.cxx
  Internal/niftkCoordinateAxesDataWriterService.cxx
  Internal/niftkPNMReaderService.cxx
  Internal/niftkPNMWriterService.cxx
  Internal/niftkCoreIOMimeTypes.cxx
  Internal/niftkCoreIOActivator.cxx
  Internal/niftkCoreIOObjectFactory.cxx
  Internal/mitkCoordinateAxesDataSerializer.cxx
  Internal/mitkLabeledLookupTablePropertySerializer.cxx
  Internal/mitkNamedLookupTablePropertySerializer.cxx
)
