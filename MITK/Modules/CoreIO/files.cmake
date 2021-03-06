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
  Internal/niftkCoordinateAxesDataSerializer.cxx
  Internal/niftkCoordinateAxesDataWriterService.cxx
  Internal/niftkCoreIOMimeTypes.cxx
  Internal/niftkCoreIOActivator.cxx
  Internal/niftkCoreIOObjectFactory.cxx
  Internal/niftkLabeledLookupTablePropertySerializer.cxx
  Internal/niftkNamedLookupTablePropertySerializer.cxx
  Internal/niftkVLPropertySerializers.cxx
  Internal/niftkPNMReaderService.cxx
  Internal/niftkPNMWriterService.cxx
  Internal/niftkLabelMapReader.cxx
  Internal/niftkLabelMapWriter.cxx
  Internal/niftkLookupTableProviderServiceImpl.cxx
)
