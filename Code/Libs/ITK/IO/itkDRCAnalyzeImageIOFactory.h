/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKDRCANALYZEIMAGEIOFACTORY_H
#define ITKDRCANALYZEIMAGEIOFACTORY_H

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include "itkAnalyzeImageIOFactory.h"

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "itkObjectFactoryBase.h"
#include "itkAnalyzeImageIO.h"

namespace itk
{
/** \class DRCAnalyzeImageIOFactory
   * \brief Create instances of DRCAnalyzeImageIO objects using an object factory.
   */
class NIFTKITK_WINEXPORT ITK_EXPORT DRCAnalyzeImageIOFactory : public AnalyzeImageIOFactory
{
public:
  /** Standard class typedefs. */
  typedef DRCAnalyzeImageIOFactory    Self;
  typedef AnalyzeImageIOFactory       Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DRCAnalyzeImageIOFactory, AnalyzeImageIOFactory);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
    {
    DRCAnalyzeImageIOFactory::Pointer metaFactory = DRCAnalyzeImageIOFactory::New();
    ObjectFactoryBase::RegisterFactory(metaFactory);
    }

protected:
  DRCAnalyzeImageIOFactory();
  ~DRCAnalyzeImageIOFactory();

private:
  DRCAnalyzeImageIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


} // end namespace itk

#endif
