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
#ifndef __NIFTK_COMMANDLINEPARSER_H_
#define __NIFTK_COMMANDLINEPARSER_H_

#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"

#include <ostream>
#include <stdio.h>
#include <string>

namespace niftk
{


/* For use in flag argument of printhelp() */
#define HLP_BRIEF 0x0000
#define HLP_VERBOSE 0x0001


/** Each possible argument on command line must have one of these
    defined (in an array terminated with OPT_DONE). 

    The fields are as follows:

    int type;	      The type of option eg. 'OPT_INT' for a single
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

    char *key;	      The option key string, eg. "option" will give
                      "-option". If 'type' is OR'ed with
                      'OPT_LONELY'  then the option key should be 'NULL'. 

    char *parm;	      The parameters the option takes. This string
                      is for descriptive purposes only, for instance
                      indicating to the user that the value required
                      for this option should be a 'filename'. 

    char *helpmsg;	A string describing the function of this option.
*/

struct CommandLineArgumentDescription {
  int type;			///< Type of option.
  const char *key;		///< option key, eg. "a" will give "-a".
  const char *parm;		///< The parameters the option takes.
  const char *helpmsg;		///< The help messages.
};

/* type field takes the following arguments. */

#define OPT_TYPEMASK 0x00ff	/* Mask type of opt. */


/** @name Option flags.

    Flags indicating the type of each command line option.
 */

//@{

#define OPT_SWITCH   0x0000	///< Option is boolean switch.
#define OPT_INT      0x0001	///< Option takes one integer.
#define OPT_FLOAT    0x0002	///< Option takes one float.
#define OPT_STRING   0x0003	///< Option takes one string.
#define OPT_VARIANT  0x0004	///< Option type is variant.
#define OPT_MULT     0x0005	///< Option takes chars stright after option.
#define OPT_DOUBLE   0x0006	///< Simliar to OPT_FLOAT but returns double.
#define OPT_LONG     0x0007	///< Simliar to OPT_INT but returns long.
#define OPT_INTx2    0x0008	///< Option takes two ints '(<a>,<b>)'
#define OPT_INTx3    0x0009	///< Option takes three ints '(<a>,<b>,<c>)'
#define OPT_INTx4    0x000a	///< Option takes four ints '(<a>,<b>,<c>,<d)'
#define OPT_FLOATx2  0x000b	///< Option takes two float '(<a>,<b>)'
#define OPT_FLOATx3  0x000c	///< Option takes three floats '(<a>,<b>,<c>)'
#define OPT_FLOATx4  0x000d	///< Option takes four floats '(<a>,<b>,<c>,<d)'
#define OPT_DOUBLEx2 0x000e	///< Option takes two double '(<a>,<b>)'
#define OPT_DOUBLEx3 0x000f	///< Option takes three doubles '(<a>,<b>,<c>)'
#define OPT_DOUBLEx4 0x0010	///< Option takes four doubles '(<a>,<b>,<c>,<d)'
#define OPT_DONE (~0)		///< No more options - must terminate array.
//@}

/** @name Option qualifier flags.

    These flags can be used in conjuction with the above basic types.
    They should be bitwise OR-ed with the basic type. 
*/
//@{

#define OPT_CHARBCK  0x0800	///< OPT_MULT return char not integer index.
#define OPT_UNIQUE   0x1000	///< Force OPT_MULT to have one entry only.
#define OPT_LONELY   0x2000	///< Parameter doesn't have key.
#define OPT_REQ      0x4000	///< Option is always required.
#define OPT_NIN      0x8000	///< Next option only if this option is.
#define OPT_MORETYPE 0x0011	///< Use OPT_MORE described below

/** This can be used as an option type too. 
    Extra arguments can be on command line,
    returns index of where 'more' args start. */
#define OPT_MORE (OPT_MORETYPE | OPT_LONELY)
//@}


/** Structures for returning parsed command line. These should be
   treated as ADTs and the internal fields never referenced. The
   function calls in class CommandLineParser should be used instead. 
*/
//@{

union ad {
  long along;
  double afloat;
  char *astr;
  int *mult;
  int *pint;
  float *pfloat;
  double *pdouble;
};

struct ml {
  struct ml *next;
  struct CommandLineArgumentDescription *opt;
  int num;
  union ad data;
};
//@}

#define ML_END (~0)

/** 
 * \class CommandLineParser
 * \brief Parser for extracting program arguments from the command line.
 * 
 * This class parses the command line using an array 'clArgList' of structures of type
 * 'CommandLineArgumentDescription' terminated in:
 *
 * '{OPT_DONE, 0, 0, "Insert brief program description here"}'.
 *
 * For an example of how to use the range of command line option types
 * please see program 'NifTK/Prototype/jhh/niftkCommandLineParserExample'.
 * 
 * @see CommandLineArgumentDescription
 * @see niftkCommandLineParserExample.cxx
 */

class NIFTKCOMMON_WINEXPORT CommandLineParser {

public:

  /** @name Special Member Functions */
  //@{
  /// CommandLineParser Constructor
  CommandLineParser(int argc, char *argv[],
		    struct CommandLineArgumentDescription *clArgList, bool flgVerbose);

  /// CommandLineParser Destructor
  ~CommandLineParser();                  

  /// CommandLineParser copy constructor
  CommandLineParser(const CommandLineParser &source);

  /// Printing member function for CommandLineParser     
  void PrintUsage(void) {PrintHelp(m_OptionList->opt, HLP_VERBOSE);}
  //@}

  /** Get back a particular argument out of parsed command line. This
      method is deprecated, use overloaded versions instead. */
  int GetArgument(int, void *);

  /// Get back a bool argument out of parsed command line.
  int GetArgument(int, bool &value);
  /// Get back an unsigned int argument out of parsed command line.
  int GetArgument(int, unsigned int &value);
  /// Get back an int argument out of parsed command line.
  int GetArgument(int, int &value);
  /// Get back an int pointer argument out of parsed command line.
  int GetArgument(int, int *&value);
  /// Get back an float argument out of parsed command line.
  int GetArgument(int, float &value);
  /// Get back an float pointer argument out of parsed command line.
  int GetArgument(int, float *&value);
  /// Get back an double argument out of parsed command line.
  int GetArgument(int, double &value);
  /// Get back an double pointer argument out of parsed command line.
  int GetArgument(int, double *&value);
  /// Get back a char string argument out of parsed command line.
  int GetArgument(int, char *&value);
  /// Get back a standard string argument out of parsed command line.
  int GetArgument(int, std::string &value);
  /// Get back a long int argument out of parsed command line.
  int GetArgument(int, long int &value);
  
private:

  /// The number of command line arguments
  int m_Argc;
  /// The command line arguments
  char **m_Argv;

  /// The name of the program
  char *m_ProgramName;

  /// Parsed option specification list.
  struct ml *m_OptionList;		

  /// Verbose flag.
  int m_flgVerbose;

  /// Print help message.
  void PrintHelp(struct CommandLineArgumentDescription *clArgList, int flag);

  /// Parse arguments on command line
  struct ml *ParseArguments(int argc, char *argv[],
			    struct CommandLineArgumentDescription *clArgList);

  /// Free resources allocated by parsearg().
  void FreeArguments(struct ml *mlist);

  int MakeOption(char *str, struct CommandLineArgumentDescription *opt, int *delayed);
  void ParseStandardArguments(char *key, struct CommandLineArgumentDescription *optlist);
  void ReadParameter(char *argv[], int arg, int type, struct ml *ml);

  struct ml *allocmlist(struct CommandLineArgumentDescription *);
  struct ml *locateml(struct ml *, struct CommandLineArgumentDescription *);
  struct ml *createml(struct ml *, struct CommandLineArgumentDescription *);
  int incarg(int, int);
  void chknotopt(char *argv[], int arg);
  int *get_intx2(char *str);
  int *get_intx3(char *str);
  int *get_intx4(char *str);
  float *get_floatx2(char *str);
  float *get_floatx3(char *str);
  float *get_floatx4(char *str);
  double *get_doublex2(char *str);
  double *get_doublex3(char *str);
  double *get_doublex4(char *str);
  void errmsg(const char *s, ...);
  void DumpInfoMessage(struct ml *mlptr, const char *argDesc, std::string value);
  void dumpmlist(void);
  void dumpmlptr(struct ml *mlptr);
};
  
} // end namespace niftk

#endif
