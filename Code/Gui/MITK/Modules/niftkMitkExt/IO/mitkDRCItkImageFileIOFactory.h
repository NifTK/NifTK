/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-08 16:23:32 +0100 (Thu, 08 Sep 2011) $
 Revision          : $Revision: 7267 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKDRCITKIMAGEUIFACTORY_H
#define MITKDRCITKIMAGEUIFACTORY_H

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "niftkMitkExtExports.h"
#include "itkObjectFactoryBase.h"
#include "mitkBaseData.h"

namespace mitk
{

/**
 * \class DRCItkImageFileIOFactory
 * \brief A DRC specific MITK Object factory, to instantiate file readers based on ITK technology,
 * where the functionality will differ from the standard MITK Object factory, due to the
 * Analyze images being flipped/converted as per DRC "standards".
 */
class NIFTKMITKEXT_EXPORT DRCItkImageFileIOFactory : public itk::ObjectFactoryBase
{
public:

  /** Standard class typedefs. */
  typedef DRCItkImageFileIOFactory       Self;
  typedef itk::ObjectFactoryBase         Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion(void) const;
  virtual const char* GetDescription(void) const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);
  static DRCItkImageFileIOFactory* FactoryNew() { return new DRCItkImageFileIOFactory;}

  /** Run-time type information (and related methods). */
  itkTypeMacro(DRCItkImageFileIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    DRCItkImageFileIOFactory::Pointer factory = DRCItkImageFileIOFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

protected:
  DRCItkImageFileIOFactory();
  ~DRCItkImageFileIOFactory();

private:
  DRCItkImageFileIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


} // end namespace mitk

#endif // MITKDRCITKIMAGEUIFACTORY_H
