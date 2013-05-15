/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkUCLRegularStepOptimizer_h
#define __itkUCLRegularStepOptimizer_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <itkSingleValuedNonLinearOptimizer.h>

namespace itk
{
  
/** 
 * \class UCLRegularStepOptimizer
 * \brief Implement a Regular Step Size optimizer.
 *
 * \ingroup Numerics Optimizers
 */
class NIFTKITK_WINEXPORT ITK_EXPORT UCLRegularStepOptimizer :
  public SingleValuedNonLinearOptimizer
{
public:
  /** Standard "Self" typedef. */
  typedef UCLRegularStepOptimizer   Self;
  typedef SingleValuedNonLinearOptimizer           Superclass;
  typedef SmartPointer<Self>                       Pointer;
  typedef SmartPointer<const Self>                 ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( UCLRegularStepOptimizer, 
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
  UCLRegularStepOptimizer();
  virtual ~UCLRegularStepOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:  
  UCLRegularStepOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&);//purposely not implemented

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


};

} // end namespace itk

#endif
