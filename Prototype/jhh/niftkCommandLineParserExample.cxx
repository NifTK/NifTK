/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include <iostream>
using namespace std;


/* This structure defines the command line options for the
   program. The fields are as follows:

   int type;		The type of option eg. 'OPT_INT' for a single
                        integer or 'OPT_FLOATx3' for a comman
                        separated list of three floats. The various
                        variable types can be OR'ed with (a)
                        'OPT_LONELY' to specfiy that no option key is 
                        associated with this command line option
                        and/or (b) 'OPT_REQ' to indicate that this
                        option is mandatory. 'Lonely' options
                        should come at the end of the command
                        line. 'OPT_MORE' can be used to specify that
                        there are multiple 'lonely' command line
                        options at the end of the command line. This 
                        is useful for specifying multiple filenames 
                        since the string option 'OPT_STRING' only 
                        allows one string to be input.
  
   char *key;		The option key string, eg. "option" will give
                        "-option". If 'type' is OR'ed with
                        'OPT_LONELY'  then the option key should be 'NULL'. 
  
   char *parm;		The parameters the option takes. This string
                        is for descriptive purposes only, for instance
                        indicating to the user that the value required
                        for this option should be a 'filename'. 
  
   char *helpmsg;	A string describing the function of this option. */ 


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "sw", 0, "A binary switch command line argument."},

  {OPT_INT,    "int", "value", "An integer command line argument."},
  {OPT_INTx2,  "intx2", "val1,val2", "A command line argument consisting of a pair of integers."},
  {OPT_INTx3,  "intx3", "val1,val2,val3", "A command line argument consisting of three integers."},
  {OPT_INTx4,  "intx4", "val1,val2,val3,val4", "A command line argument consisting of four integers."},

  {OPT_LONG,   "long", "value", "A long integer command line argument."},

  {OPT_FLOAT,   "float", "value", "A float command line argument."},
  {OPT_FLOATx2, "floatx2", "val1,val2", "A command line argument consisting of a pair of floats."},
  {OPT_FLOATx3, "floatx3", "val1,val2,val3", "A command line argument consisting of three floats."},
  {OPT_FLOATx4, "floatx4", "val1,val2,val3,val4", "A command line argument consisting of four floats."},

  {OPT_DOUBLE, "double", "value", "A double command line argument."},
  {OPT_DOUBLEx2, "doublex2", "val1,val2", "A command line argument consisting of a pair of doubles."},
  {OPT_DOUBLEx3, "doublex3", "val1,val2,val3", "A command line argument consisting of three doubles."},
  {OPT_DOUBLEx4, "doublex4", "val1,val2,val3,val4", "A command line argument consisting of four doubles."},

  {OPT_STRING, "string", "filename", "A character string command line argument."},
  
  {OPT_INT|OPT_LONELY, NULL, "int", "A single integer with no option key (hence 'lonely')."},
  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "string(s)", "Multiple strings (e.g. filenames)."},
  
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Program to illustrate the various command line parsing options in class 'niftk::CommandLineParser'.\n\n"

   "Example command line:\n"
   "niftkCommandLineParserExample \\\n"
   "\t-sw -int 11 -long 1234567890 -float 0.0018 -double 0.00101010101010 \\\n"
   "\t-string hello -intx2 3,4 -intx3 1,2,3 -intx4 1,2,3,4 -floatx2 1.2,3.4 \\\n"
   "\t-floatx3 2.4,5.6,6.7 -floatx4 5.6,7.8,3.4,5.6 -doublex2 1.2,3.4 \\\n"
   "\t-doublex3 3.2,3.4,5.6 -doublex4 3.4,2.3,5.6,6.7 1 string1 string2 string3\n"

  }
};



/* This enumerated list is simply used to simplify indexing the list
   of options in the 'CommandLineArgumentDescription' above via the
   CommandLineParser method 'GetArgument()'. Care
   should be taken to ensure that each value corresponds to the correct
   value in the 'CommandLineArgumentDescription' list. */

enum {
  O_SWITCH = 0,

  O_INT,
  O_INTx2,
  O_INTx3,
  O_INTx4,

  O_LONG,

  O_FLOAT,
  O_FLOATx2,
  O_FLOATx3,
  O_FLOATx4,

  O_DOUBLE,
  O_DOUBLEx2,
  O_DOUBLEx3,
  O_DOUBLEx4,

  O_STRING,
  O_LONELY_INT,
  O_REQ_STRING,

  O_MORE
};


int main(int argc, char** argv)
{
  bool aSwitch;			// Binary/boolean switch argument

  int anInteger;		// An integer argument
  int *twoIntegers = 0;		// A pair of integers
  int *threeIntegers = 0;	// Three integers
  int *fourIntegers = 0;	// Four integers

  int i;			// Loop counter
  int arg;			// Index of arguments in command line 
  int nStrings = 0;		// Number of 'multiple strings' on the
				// command line

  long int aLongInteger;	// A long integer argument

  float aFloat;			// A float argument
  float *twoFloats = 0;		// A pair of floats
  float *threeFloats = 0;	// Three floats
  float *fourFloats = 0;	// Four floats

  double aDouble;		// A double argument
  double *twoDoubles = 0;		// A pair of doubles
  double *threeDoubles = 0;	// Three doubles
  double *fourDoubles = 0;	// Four doubles

  char *aString = 0;		// A character string argument
  char *aRequiredString = 0;	// A mandatory character string argument
  char **multipleStrings = 0;	// Multiple character string arguments

  std::string aStdString;	// A standard string argument

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  // The boolean 'OPT_SWITCH' argument sets the value of 'aSwitch' to
  // true or false depending upon whether this option is present or
  // absent from the command line.
  CommandLineOptions.GetArgument(O_SWITCH, aSwitch);

  CommandLineOptions.GetArgument(O_INT,   anInteger);
  CommandLineOptions.GetArgument(O_INTx2, twoIntegers);
  CommandLineOptions.GetArgument(O_INTx3, threeIntegers);
  CommandLineOptions.GetArgument(O_INTx4, fourIntegers);

  CommandLineOptions.GetArgument(O_LONG, aLongInteger);

  CommandLineOptions.GetArgument(O_FLOAT, aFloat);
  CommandLineOptions.GetArgument(O_FLOATx2, twoFloats);
  CommandLineOptions.GetArgument(O_FLOATx3, threeFloats);
  CommandLineOptions.GetArgument(O_FLOATx4, fourFloats);

  CommandLineOptions.GetArgument(O_DOUBLE, aDouble);
  CommandLineOptions.GetArgument(O_DOUBLEx2, twoDoubles);
  CommandLineOptions.GetArgument(O_DOUBLEx3, threeDoubles);
  CommandLineOptions.GetArgument(O_DOUBLEx4, fourDoubles);

  CommandLineOptions.GetArgument(O_STRING, aString);
  CommandLineOptions.GetArgument(O_STRING, aStdString);

  CommandLineOptions.GetArgument(O_LONELY_INT, anInteger);
  CommandLineOptions.GetArgument(O_REQ_STRING, aRequiredString);

  // Call the 'OPT_MORE' option to determine the position of the list
  // of extra command line options ('arg').
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {            // Many strings
    nStrings = argc - arg + 1;
    multipleStrings = &argv[arg-1];

    std::cout << std::endl << "Input strings: " << std::endl;
    for (i=0; i<nStrings; i++)
      std::cout << "   " << i+1 << " " << multipleStrings[i] << std::endl;
  }
  else if (aRequiredString) {	// Single string
    nStrings = 1;
    multipleStrings = &aRequiredString;

    std::cout << std::endl << "Input string: " << multipleStrings[0] << std::endl;
  }
  else {
    nStrings = 0;
    multipleStrings = 0;
  }

  return 0;
}

