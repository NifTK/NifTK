/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 13:12:13 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8041 $
 Last modified by  : $Author: jhh $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkUCLRegularStepOptimizer_h
#define __itkUCLRegularStepOptimizer_h

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include <log4cplus/logger.h>
#include "itkSingleValuedNonLinearOptimizer.h"

#include <fstream>

namespace itk
{
  
/** 
 * \class ConstrainedRegStepOptimizer
 * \brief Implement a Regular Step Size optimizer.
 *
 * \ingroup Numerics Optimizers
 */
class NIFTKITK_WINEXPORT ITK_EXPORT ConstrainedRegStepOptimizerForSDM :
  public SingleValuedNonLinearOptimizer
{
public:
  /** Standard "Self" typedef. */
  typedef ConstrainedRegStepOptimizerForSDM        Self;
  typedef SingleValuedNonLinearOptimizer           Superclass;
  typedef SmartPointer<Self>                       Pointer;
  typedef SmartPointer<const Self>                 ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( ConstrainedRegStepOptimizerForSDM, 
        	SingleValuedNonLinearOptimizer );

  /** Codes of stopping conditions. */
  typedef enum {
    StepTooSmall = 1,
    ImageNotAvailable = 2,
    CostFunctionError = 3,
    MaximumNumberOfIterations = 4,
    Unknown = 5
  } StopConditionType;

  /** Specify whether to minimize or maximize the cost function. */
  itkSetMacro( Maximize, bool );
  itkGetConstReferenceMacro( Maximize, bool );
  itkBooleanMacro( Maximize );
  bool GetMinimize( ) const
    { return !m_Maximize; }
  void SetMinimize(bool v)
    { this->SetMaximize(!v); }
  void    MinimizeOn(void) 
    { SetMaximize( false ); }
  void    MinimizeOff(void) 
    { SetMaximize( true ); }

  /** Start optimization. */
  void    StartOptimization( void );

  /** Resume previously stopped optimization with current parameters.
   * \sa StopOptimization */
  void    ResumeOptimization( void );

  /** Stop optimization.
   * \sa ResumeOptimization */
  void    StopOptimization( void );

  void SetProgressFileName ( std::string name )
    { m_progressFileName = name; }

 
  void SetNumberOfPCAComponents(unsigned int numOfComponents)
  {
     m_NumberOfPCAComponents = numOfComponents;
  }

  /** Set/Get parameters to control the optimization process. */
  itkSetMacro( MaximumStepLength, double );
  itkSetMacro( MinimumStepLength, double );
  itkSetMacro( RelaxationFactor, double );
  itkSetMacro( NumberOfIterations, unsigned long );
  itkGetConstReferenceMacro( CurrentStepLength, double);
  itkGetConstReferenceMacro( MaximumStepLength, double );
  itkGetConstReferenceMacro( MinimumStepLength, double );
  itkGetConstReferenceMacro( RelaxationFactor, double );
  itkGetConstReferenceMacro( NumberOfIterations, unsigned long );
  itkGetConstMacro( CurrentIteration, unsigned int );
  itkGetConstReferenceMacro( StopCondition, StopConditionType );
  itkGetConstReferenceMacro( Value, MeasureType );
  
protected:
  ConstrainedRegStepOptimizerForSDM();
  virtual ~ConstrainedRegStepOptimizerForSDM() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:  
  ConstrainedRegStepOptimizerForSDM(const Self&); //purposely not implemented
  void operator=(const Self&);//purposely not implemented

  /** Do all logging using log4cplus. */
  static log4cplus::Logger s_Logger;
  
protected:
  bool                          m_Stop;
  bool                          m_Maximize;
  MeasureType                   m_Value;
  MeasureType                   m_BestSoFarValue;
  ParametersType                m_BestSoFarParameters;
  double                        m_MaximumStepLength;
  double                        m_MinimumStepLength;
  double                        m_CurrentStepLength;
  double                        m_RelaxationFactor;
  StopConditionType             m_StopCondition;
  unsigned long                 m_NumberOfIterations;
  unsigned long                 m_CurrentIteration;
  
  std::ofstream m_progressFile;
  std::string m_progressFileName;

  unsigned int m_NumberOfPCAComponents;

  static const float compMin = -3.;
  static const float compMax = 3.;

};

} // end namespace itk

#endif
