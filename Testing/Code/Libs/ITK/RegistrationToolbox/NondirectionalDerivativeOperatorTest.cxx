/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include "itkDerivativeOperator.h"
#include "itkNondirectionalDerivativeOperator.h"

int NondirectionalDerivativeOperatorTest(int argc, char * argv[])
{
  typedef itk::DerivativeOperator< double, 2 > DerivativeOperatorType; 
  DerivativeOperatorType derivativeOperator; 
  DerivativeOperatorType::SizeType radius; 
  
  derivativeOperator.SetDirection(0);
  derivativeOperator.SetOrder(1);
  derivativeOperator.CreateDirectional();
  derivativeOperator.FlipAxes();
  radius = derivativeOperator.GetRadius();
  std::cout << "operator=" << derivativeOperator << std::endl;
  
  DerivativeOperatorType::ConstIterator it = derivativeOperator.Begin();
  
  for (; it != derivativeOperator.End() ; ++it)
  {
    std::cout << *it << " "; 
  }
  std::cout << std::endl;
  
  radius[0] = 4;
  derivativeOperator.CreateToRadius(radius);
  derivativeOperator.FlipAxes();
  std::cout << "operator=" << derivativeOperator << std::endl;
  it = derivativeOperator.Begin();
  for (; it != derivativeOperator.End() ; ++it)
  {
    std::cout << *it << " "; 
  }
  std::cout << std::endl;
  
  
  typedef itk::NondirectionalDerivativeOperator < double, 3 > NondirectionalDerivativeOperatorType; 
  NondirectionalDerivativeOperatorType nondirectionalDerivativeOperator; 
  NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType term; 
  NondirectionalDerivativeOperatorType::SingleDerivativeTermInfoType::DerivativeOrderType order; 
  
  order[0] = 2;
  order[1] = 2;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(1.0);
  nondirectionalDerivativeOperator.AddSingleDerivativeTerm(term);
  nondirectionalDerivativeOperator.CreateToRadius(1);
  
  NondirectionalDerivativeOperatorType::ConstIterator nondirectionalDerivativeOperatorIt; 
  int index;
  
  double expectedAnswer220[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 1, -2, 4, -2, 1, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  for (nondirectionalDerivativeOperatorIt = nondirectionalDerivativeOperator.Begin(), index = 0; 
       nondirectionalDerivativeOperatorIt != nondirectionalDerivativeOperator.End(); 
       ++nondirectionalDerivativeOperatorIt, index++)
  {
    std::cout << *nondirectionalDerivativeOperatorIt << " "; 
    if (fabs(expectedAnswer220[index] - *nondirectionalDerivativeOperatorIt) > 0.000001)
      return EXIT_FAILURE; 
  }
  std::cout << std::endl;
  
  order[0] = 1;
  order[1] = 1;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(1.0);
  nondirectionalDerivativeOperator.ClearDerivativeTerm();
  nondirectionalDerivativeOperator.AddSingleDerivativeTerm(term);
  nondirectionalDerivativeOperator.CreateToRadius(1);
  
  double expectedAnswer110[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0, -0.25, 0, 0, 0, -0.25, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  for (nondirectionalDerivativeOperatorIt = nondirectionalDerivativeOperator.Begin(), index = 0; 
       nondirectionalDerivativeOperatorIt != nondirectionalDerivativeOperator.End(); 
       ++nondirectionalDerivativeOperatorIt, index++)
  {
    std::cout << *nondirectionalDerivativeOperatorIt << " "; 
    if (fabs(expectedAnswer110[index] - *nondirectionalDerivativeOperatorIt) > 0.000001)
      return EXIT_FAILURE; 
  }
  std::cout << std::endl;
  
  order[0] = 2;
  order[1] = 2;
  order[2] = 2;
  term.SetDervativeOrder(order);
  term.SetConstant(1.0);
  nondirectionalDerivativeOperator.ClearDerivativeTerm();
  nondirectionalDerivativeOperator.AddSingleDerivativeTerm(term);
  nondirectionalDerivativeOperator.CreateToRadius(1);
  
  double expectedAnswer222[] = { 1, -2, 1, -2, 4, -2, 1, -2, 1, 
                                -2,  4,-2,  4,-8,  4,-2,  4,-2, 
                                 1, -2, 1, -2, 4, -2, 1, -2, 1 };
  for (nondirectionalDerivativeOperatorIt = nondirectionalDerivativeOperator.Begin(), index = 0; 
       nondirectionalDerivativeOperatorIt != nondirectionalDerivativeOperator.End(); 
       ++nondirectionalDerivativeOperatorIt, index++)
  {
    std::cout << *nondirectionalDerivativeOperatorIt << " "; 
    if (fabs(expectedAnswer222[index] - *nondirectionalDerivativeOperatorIt) > 0.000001)
      return EXIT_FAILURE; 
  }
  std::cout << std::endl;
  
  order[0] = 2;
  order[1] = 2;
  order[2] = 2;
  term.SetDervativeOrder(order);
  term.SetConstant(2.0);
  nondirectionalDerivativeOperator.ClearDerivativeTerm();
  nondirectionalDerivativeOperator.AddSingleDerivativeTerm(term);
  nondirectionalDerivativeOperator.CreateToRadius(1);
  
  for (nondirectionalDerivativeOperatorIt = nondirectionalDerivativeOperator.Begin(), index = 0; 
       nondirectionalDerivativeOperatorIt != nondirectionalDerivativeOperator.End(); 
       ++nondirectionalDerivativeOperatorIt, index++)
  {
    std::cout << *nondirectionalDerivativeOperatorIt << " "; 
    if (fabs(2.0*expectedAnswer222[index] - *nondirectionalDerivativeOperatorIt) > 0.000001)
      return EXIT_FAILURE; 
  }
  std::cout << std::endl;
  
  
  order[0] = 2;
  order[1] = 0;
  order[2] = 0;
  term.SetDervativeOrder(order);
  term.SetConstant(1.0);
  nondirectionalDerivativeOperator.ClearDerivativeTerm();
  nondirectionalDerivativeOperator.AddSingleDerivativeTerm(term);
  order[0] = 0;
  order[1] = 2;
  order[2] = 0;
  term.SetDervativeOrder(order);
  nondirectionalDerivativeOperator.AddSingleDerivativeTerm(term);
  order[0] = 0;
  order[1] = 0;
  order[2] = 2;
  term.SetDervativeOrder(order);
  nondirectionalDerivativeOperator.AddSingleDerivativeTerm(term);
  nondirectionalDerivativeOperator.CreateToRadius(1);
  
  double expectedAnswer222Sum[] = { 0,  0, 0,  0, 1,  0, 0,  0, 0, 
                                    0,  1, 0,  1,-6,  1, 0,  1, 0, 
                                    0,  0, 0,  0, 1,  0, 0,  0, 0 };
  for (nondirectionalDerivativeOperatorIt = nondirectionalDerivativeOperator.Begin(), index = 0; 
       nondirectionalDerivativeOperatorIt != nondirectionalDerivativeOperator.End(); 
       ++nondirectionalDerivativeOperatorIt, index++)
  {
    std::cout << *nondirectionalDerivativeOperatorIt << " "; 
    if (fabs(expectedAnswer222Sum[index] - *nondirectionalDerivativeOperatorIt) > 0.000001)
      return EXIT_FAILURE; 
  }
  std::cout << std::endl;
  

  std::cout << "Test PASSED !" << std::endl;
  return EXIT_SUCCESS;
}





