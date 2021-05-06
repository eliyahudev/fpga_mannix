#ifndef CNN_INC_H
#define CNN_INC_H

#ifdef FPGNIX
#define BARE_METAL
#endif


#ifndef BARE_METAL
  #include "stdio.h"
  #include <stdlib.h>
  #include <windows.h>
  #pragma warning(disable : 4996)
#else
  #include <gpio.h>        // simple SOC gpio interface
  #include <iosim.h>       // Simulated IO (basic terminal and file access)  over gpio interface
  #include <bm_printf.h>   // bare-metal printf
  #include <uart.h>
  #include <rcg.h>
  #include <hamsa_config.h>
  #include <utils.h>
  #include <cycle_count_access.h>
  
  
  #define CLK_PERIOD_NS CLK_PERIOD
  #define NS_PER_BIT ((1000000000)/BAUD_RATE)
  #define UART_CLK (((NS_PER_BIT/CLK_PERIOD))-1)  

#endif



#include "man_def.h"

#ifdef BARE_METAL
 #define  PULP_EXT
 #include "conv_5x5.h"
#endif

#include "man_struct.h"

#ifndef MEM_LOAD_MODE
   #include "read_csv.h"
#endif

#include "mannixlib.h"
#include "mannix_matrix.h"
 
#include "mannix_tensor.h"
#include "mannix_4dtensor.h"

//#include "cnn.h"  // not needed for now

#endif 