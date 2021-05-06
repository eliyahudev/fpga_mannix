// MANNIX i/o addresses

#define GPP_BASE_ADDR   0x1A106000 // TODO 
#define CNN_ADDRX       GPP_BASE_ADDR +0X0004
#define CNN_ADDRY       GPP_BASE_ADDR +0X0008
#define CNN_ADDRZ       GPP_BASE_ADDR +0X000C
#define CNN_XM          GPP_BASE_ADDR +0X0010
#define CNN_XN          GPP_BASE_ADDR +0X0014
#define CNN_YM          GPP_BASE_ADDR +0X0018
#define CNN_YN          GPP_BASE_ADDR +0X001C
#define CNN_STRAT       GPP_BASE_ADDR +0X0020
#define CNN_DONE        GPP_BASE_ADDR +0X0024

#define POOL_ADDRX       GPP_BASE_ADDR +0X0028
#define POOL_ADDRZ      GPP_BASE_ADDR +0X002C
#define POOL_XM         GPP_BASE_ADDR +0X0030
#define POOL_XN         GPP_BASE_ADDR +0X0034  
#define POOL_PM         GPP_BASE_ADDR +0X0038
#define POOL_PN         GPP_BASE_ADDR +0X003C
#define POOL_START      GPP_BASE_ADDR +0X0040
#define POOL_DONE       GPP_BASE_ADDR +0X0044

#define ACTIV_ADDRX     GPP_BASE_ADDR +0X0048
#define ACTIV_XM        GPP_BASE_ADDR +0X004C
#define ACTIV_XN        GPP_BASE_ADDR +0X0050
#define ACTIV_START     GPP_BASE_ADDR +0X0054
#define ACTIV_DONE      GPP_BASE_ADDR +0X0058

#define FC_ADDRX        GPP_BASE_ADDR +0X005C
#define FC_ADDRY        GPP_BASE_ADDR +0X0060
#define FC_ADDRB        GPP_BASE_ADDR +0X0064
#define FC_ADDRZ        GPP_BASE_ADDR +0X0068
#define FC_XM           GPP_BASE_ADDR +0X006C
#define FC_YM           GPP_BASE_ADDR +0X0070
#define FC_YN           GPP_BASE_ADDR +0X0074
#define FC_BN           GPP_BASE_ADDR +0X0078
#define FC_START        GPP_BASE_ADDR +0X007C
#define FC_DONE         GPP_BASE_ADDR +0X0080

//---- program define ----
// #define MANNIX_CNN_F
// #define MANNIX_ACTIVE_F
// #define MANNIX_PULL_F
// #define MANNIX_FC_F

//---- dubaging defines ----
// #define TEST // result example of few pictures
// #define CMP_TEST //
//#define DEBUG // debug variable

#ifdef CMP_TEST
#define IFDEF_CMP_TEST(dump_call) dump_call
#else
#define IFDEF_CMP_TEST(dump_call) 
#endif


//---- OS ----
#define WINDOWS_MANNIX
// #define DEBIAN

//---- environment setups----

//#define VS_MANNIX

//#define MEM_DUMP_MODE // Dumps the model parameters loadable data vector , run once per model configuration.

#define MEM_LOAD_MODE  // Model parameters and data to be actively loaded , skip CSV read

#define MODEL_PARAMS_FILE "fashion_mnist_model/model_params_db/model_params_mfdb.txt"

#define DATASET_FILE "fashion_mnist_model/test_src/fashion_mnist_V1_mfds.txt"

// #define ONLY_KERNEL_5X5_USED

// ----- environment defines ----------
// memory size definitions
#define MANNIX_DATA_SIZE 53000 // TODO - ask Udi memory can be decreased to 44000  
#define MANNIX_MAT_SIZE 2000
#define MANNIX_TEN_SIZE 500
// activation
#define WB_LOG2_SCALE 7
#define UINT_DATA_WIDTH 8
#define LOG2_RELU_FACTOR 1
#define EXP_UINT_DATA_WIDTH 1 << UINT_DATA_WIDTH
#define MAX_DATA_RANGE (EXP_UINT_DATA_WIDTH) - 1  

// BARE-METAL BARE_METAL MODE

#ifndef BARE_METAL
   #define FILE_PTR FILE*
   #define fopen_w(fileName) fopen(fileName,"w")
   #define fopen_r(fileName) fopen(fileName,"r")  
   #pragma warning(disable : 4996)
#else
   #define printf bm_printf
   #define fprintf bm_fprintf   
   #define exit(X) sim_finish()
   #define fopen_w(fileName) bm_fopen_w(fileName)
   #define fopen_r(fileName) bm_fopen_r(fileName)  
   #define fclose bm_fclose   
   #define FILE_PTR int
   
   // UART DEFINES
   #define CLK_PERIOD_NS CLK_PERIOD
   #define NS_PER_BIT ((1000000000)/BAUD_RATE)
   #define UART_CLK (((NS_PER_BIT/CLK_PERIOD))-1)  

   
   #define NULL -1 // TMP, TODO - CHECK
#endif


