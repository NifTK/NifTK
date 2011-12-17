/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-24 15:23:38 +0100 (Wed, 24 Aug 2011) $
 Revision          : $Revision: 7154 $
 Last modified by  : $Author: be $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkNIFTKTransformIOFactory_h
#define __itkNIFTKTransformIOFactory_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "itkObjectFactoryBase.h"
#include "itkTransformIOBase.h"
#include "niftkITKWin32ExportHeader.h"

namespace itk
{
/** \class NIFTKTransformIOFactory
   * \brief Create instances of UCLTransformIO objects using an object factory.
   */
class NIFTKITK_WINEXPORT ITK_EXPORT NIFTKTransformIOFactory : public ObjectFactoryBase
{
public:
  /** Standard class typedefs. */
  typedef NIFTKTransformIOFactory    Self;
  typedef ObjectFactoryBase        Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion(void) const;
  virtual const char* GetDescription(void) const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NIFTKTransformIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
    {
    NIFTKTransformIOFactory::Pointer metaFactory = NIFTKTransformIOFactory::New();
    ObjectFactoryBase::RegisterFactory(metaFactory);
    }

protected:
  NIFTKTransformIOFactory();
  ~NIFTKTransformIOFactory();
  virtual void PrintSelf(std::ostream& os, Indent indent) const;

private:
  NIFTKTransformIOFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


} // end namespace itk

#endif
