

//#include "../include/cnn_inc.h"
#include "include/cnn_inc.h" // Use this for BARE METAL env.


#ifndef FPGNIX
int main(int argc, char const *argv[]) {
#else
int main() {
#endif //FPGNIX

#ifdef BARE_METAL
init_pll_rcg();   
uart_set_cfg(0, UART_CLK);
#endif // BARE_METAL


printf("HELLO FASHION MNIST\n");


// like-dynamic memory allocation for the program
#ifndef BARE_METAL
#include "include/tensor_allocation_setup.h"
#else
#include "include/tensor_allocation_setup.h"
#endif // !BARE_METAL

    // declare 4D tensors
    Tensor4D_uint8 image[1];
    Tensor4D_int8 conv_weight[2];
    Tensor4D_int32 result_4D_tensor[2];
    Tensor4D_uint8 result_maxPool_tensor[2];
    Tensor4D_uint8 result_4D_tensor_uint8[2];

    // declare matrix bias [for each matrix there is one bias value, for example for image->matrix[0] we add the same value bias->data[0] to all cells]
    Matrix_int32 conv_bias[2];
    
    Matrix_int8 fc_weight[3];
    Matrix_int32 fc_bias[3];
    Matrix_int32 result_matrix[3];
    Matrix_uint8 result_matrix_uint8[3];
    
    // import matrices

    char* path_in = { "../../../python/csv_dumps/scaled_int/" };

#ifndef MEM_LOAD_MODE
  #ifdef VS_MANNIX
      #ifdef CMP_TEST  
      FILE_PTR imageFilePointer = fopen("../../test_src/img_3673.csv", "r");
      char file_out[80] ;
      #else
      FILE_PTR imageFilePointer = fopen("../../test_src/data_set_256_fasion_emnist.csv", "r");
      #endif // !CMP_TEST
      char * path_out = {"../../test_products/"} ;    
  #else // !VS_MANNIX
      char* path_in = { "../../python/csv_dumps/scaled_int/" };
      #ifdef CMP_TEST  
      FILE_PTR imageFilePointer = fopen("../../../../test_and_delete/mannix_test/inference/img_csv_dumps/img_3673.csv", "r");
      #else
      FILE_PTR imageFilePointer = fopen("../test_src/data_set_256_fasion_emnist.csv", "r");
      #endif //CMP_TEST
      char * path_out = {"../test_products/"} ;
  #endif // !VS_MANNIX
#else // MEM_LOAD_MODE
      FILE_PTR imageFilePointer = fopen_r(DATASET_FILE);  // @MEM_LOAD_MODE (File name defined at man_def.h)
    char * path_out = {"../test_products/"} ;
    char file_out[80] ;

#endif //!MEM_LOAD_MODE

    // allocate memory for image, conv_weight and bias
    printf("allocating memory\n");
    create4DTensor_uint8(&image[0], 28, 28, 1, 1,    (Allocator_uint8*)al, (MatAllocator_uint8*)mat_al, (TensorAllocator_uint8*)tens_alloc);  
    create4DTensor_int8(&conv_weight[0], 5, 5, 1, 6, (Allocator_int8*)al, (MatAllocator_int8*)mat_al, (TensorAllocator_int8*)tens_alloc);
    create4DTensor_int8(&conv_weight[1], 5, 5, 6, 12,(Allocator_int8*)al, (MatAllocator_int8*)mat_al, (TensorAllocator_int8*)tens_alloc);
    creatMatrix_int32(6, 1,   &conv_bias[0],   (Allocator_int32*) al);
    creatMatrix_int32(12, 1,  &conv_bias[1],   (Allocator_int32*) al);
    creatMatrix_int8(120, 192,&fc_weight[0],   (Allocator_int8*)  al);
    creatMatrix_int8(64, 120, &fc_weight[1],   (Allocator_int8*)  al);
    creatMatrix_int8(10, 64,  &fc_weight[2],   (Allocator_int8*)  al);
    creatMatrix_int32(120,1,  &fc_bias[0],     (Allocator_int32*) al);
    creatMatrix_int32(64,1,   &fc_bias[1],     (Allocator_int32*) al);
    creatMatrix_int32(10,1,   &fc_bias[2],     (Allocator_int32*) al);
 
#ifndef MEM_LOAD_MODE
    // set values from csv table
    printf("setting weights and bais\n");
    setFilter(&conv_weight[0], path_in, 1);
    setFilter(&conv_weight[1], path_in, 2);
    setBias(&conv_bias[1], path_in, "conv", 2, "b");
    setBias(&conv_bias[0], path_in, "conv", 1, "b");
    setBias(&fc_bias[0], path_in, "fc", 1, "b");
    setBias(&fc_bias[1], path_in, "fc", 2, "b");
    setBias(&fc_bias[2], path_in, "fc", 3, "b");
    setWeight(&fc_weight[0], path_in, "fc", 1, "w");
    setWeight(&fc_weight[1], path_in, "fc", 2, "w");
    setWeight(&fc_weight[2], path_in, "fc", 3, "w");
#endif

    #ifdef MEM_DUMP_MODE // Dumps the model parameters loadable db , run once per model configuration.
    dump_model_params_mfdb(al,MODEL_PARAMS_FILE);  // dump mannix format data base
    #endif
    

    #ifdef MEM_LOAD_MODE // Load the model parameters pre-dumped db
    load_model_params_mfdb(al,MODEL_PARAMS_FILE);  // load mannix format data base
    #endif
  
    int success_count = 0;
    int fail_count = 0;

    unsigned char* reset_mannix_data = al->data;
    int reset_mannix_data_index = al->index ;
    Matrix_uint8* reset_mannix_matrix = mat_al->matrix ;
    int reset_mannix_matrix_index = mat_al->index ;
    Tensor_uint8* reset_mannix_tensor = tens_alloc->tensor ;
    int reset_mannix_tensor_index = tens_alloc->index ;

    // print4DTensor_int8(&conv_weight[0]);
    // print4DTensor_int8(&conv_weight[1]);
    // bm_printf("conv_bias[1]\n");
    // printMatrix_int32(&conv_bias[1]);
    // bm_printf("\n\n");
    // bm_printf("conv_bias[0]\n");
    // printMatrix_int32(&conv_bias[0]);
    // bm_printf("\n\n");
    // bm_printf("fc_bias[0]\n");
    // printMatrix_int32(&fc_bias[0]);
    // bm_printf("\n\n");
    // bm_printf("fc_bias[1]\n");
    // printMatrix_int32(&fc_bias[1]);
    // bm_printf("\n\n");
    // bm_printf("fc_bias[2]\n");
    // printMatrix_int32(&fc_bias[2]);
    // bm_printf("\n\n");
    // bm_printf("fc_weight[0]\n");
    // printMatrix_int8(&fc_weight[0]);
    // bm_printf("\n\n");
    // bm_printf("fc_weight[1]\n");
    // printMatrix_int8(&fc_weight[1]);
    // bm_printf("\n\n");
    // bm_printf("fc_weight[2]\n");
    // printMatrix_int8(&fc_weight[2]);
    // bm_printf("\n");

    printf("============================================================================\n");
    printf("=============== starting test (it could take some time...): ================\n");
    printf("============================================================================\n\n");

    #ifdef BARE_METAL
          int start_cycle, end_cycle ;          
          ENABLE_CYCLE_COUNT  ; // Enable the cycle counter
          RESET_CYCLE_COUNT  ; // Reset counter to ensure 32 bit counter does not wrap in-between start and end.           
          GET_CYCLE_COUNT_START(start_cycle) ; 
    #endif

  #ifdef TEST
      for (int i = 0; i < 1; i++) {
  #else
      #ifdef CMP_TEST
          for (int a = 0; a < 1; a++) {
      #else
          int i=0;
          #ifndef BARE_METAL          
          while (!feof(imageFilePointer)) {
          #else
          int num_img_per_cnt_msg = 100 ;
          while (i<10000) {  // TMP NEED ROUBUST BARE-METAL EOF EQUIVELENT
          #endif // !BARE_METAL
              if (((i%num_img_per_cnt_msg)==0)&(i>0)) {
                  printf("Checked %d images ...\n",i) ;
                  GET_CYCLE_COUNT_END(end_cycle) ;             
                  bm_printf("%d cycles for last %d images\n", end_cycle-start_cycle,num_img_per_cnt_msg) ;   
                  RESET_CYCLE_COUNT  ; // Reset counter to ensure 32 bit counter does not wrap in-between start and end.           
                  GET_CYCLE_COUNT_START(start_cycle) ; // start measuring for next num_img_per_cnt_msg images                  
              }   
              i++ ;
      #endif   //!CMP_TEST          
   #endif //!TEST

        al->data = reset_mannix_data ;
        al->index = reset_mannix_data_index ;
        mat_al->matrix = reset_mannix_matrix ;
        mat_al->index = reset_mannix_matrix_index ;
        tens_alloc->tensor = reset_mannix_tensor ;
        tens_alloc->index = reset_mannix_tensor_index ;

        int real_label = setImage(&image[0], imageFilePointer);
        // print4DTensor_uint8(image);

        int sc = LOG2_RELU_FACTOR ;  // 'sc' (extra scale) is determined by LOG2_RELU_FACTOR , consider maling this layer specific;

    //  convolution layer   
        Tensor4D_uint8 * actResult_4D_tensor = tensor4DConvNActivate(image, &conv_weight[0], &conv_bias[0], result_4D_tensor_uint8, (Allocator_int32 *)al, (MatAllocator_int32 *)mat_al, (TensorAllocator_int32 *)tens_alloc, sc);
        // print4DTensor_uint8(actResult_4D_tensor);
        IFDEF_CMP_TEST( writeTensor4DToCsv_uint8 (actResult_4D_tensor, path_out, "conv1_relu_out"); )
        
        Tensor4D_uint8 * maxPoolResult_tensor =  tensor4DMaxPool(actResult_4D_tensor, &result_maxPool_tensor[0], 2, 2, 2, al, mat_al, tens_alloc);
        // print4DTensor_uint8(maxPoolResult_tensor);
        IFDEF_CMP_TEST(writeTensor4DToCsv_uint8(maxPoolResult_tensor, path_out, "conv1_pool2d_out"); )

        actResult_4D_tensor = tensor4DConvNActivate(maxPoolResult_tensor, &conv_weight[1], &conv_bias[1], &result_4D_tensor_uint8[1], (Allocator_int32 *)al, (MatAllocator_int32 *)mat_al, (TensorAllocator_int32 *)tens_alloc, sc);
        // print4DTensor_uint8(actResult_4D_tensor);
        IFDEF_CMP_TEST( writeTensor4DToCsv_uint8 (actResult_4D_tensor  , path_out, "conv2_relu_out"); )

        maxPoolResult_tensor =  tensor4DMaxPool(actResult_4D_tensor, & result_maxPool_tensor[1], 2, 2, 2, al, mat_al, tens_alloc);
    //  print4DTensor_uint8(maxPoolResult_tensor);
        IFDEF_CMP_TEST( writeTensor4DToCsv_uint8 (maxPoolResult_tensor, path_out, "conv2_pool2d_out"); )


        // fully-connected layer 
        tensor4Dflatten(maxPoolResult_tensor);
        
        Matrix_uint8 * actResultMatrix  = matrixFCNActivate(maxPoolResult_tensor->tensor->matrix, &fc_weight[0], &fc_bias[0], &result_matrix_uint8[0], (Allocator_int32 *)al, sc);

        IFDEF_CMP_TEST(strcpy(file_out,path_out);writeMatrixToCsv_uint8(actResultMatrix,strcat(file_out,"fc1_relu_out.csv")); ) 
        
        actResultMatrix  = matrixFCNActivate(&result_matrix_uint8[0], &fc_weight[1], &fc_bias[1], &result_matrix_uint8[1], (Allocator_int32 *)al, sc);

        IFDEF_CMP_TEST(strcpy(file_out,path_out);writeMatrixToCsv_uint8(actResultMatrix,strcat(file_out,"fc2_relu_out.csv")); ) 

        actResultMatrix  = matrixFCNActivate(&result_matrix_uint8[1], &fc_weight[2], &fc_bias[2], &result_matrix_uint8[2], (Allocator_int32 *)al, sc);
        
        IFDEF_CMP_TEST(strcpy(file_out,path_out);writeMatrixToCsv_uint8(&result_matrix_uint8[2],strcat(file_out,"fc3_out.csv")); ) 

        int detected_label = maxElement_uint8(actResultMatrix) ;
        
        const char itemsStrVec[10][20]  = {"target-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"};

        
        if (real_label == detected_label) { 
		   
														
	  
            success_count++;
            #if defined(TEST) || defined(BARE_METAL) 
            printf("image %5d : OK    %s\n",i,itemsStrVec[detected_label]);
            #endif
        }
        else {
		   
													  
	  
            fail_count++;
            #if defined (TEST) || defined (BARE_METAL)
            printf("image %5d : WRONG %s EXPECTED %s\n",i,itemsStrVec[detected_label],itemsStrVec[real_label]);
            #endif
        }
    }

#ifndef MEM_LOAD_MODE
    fclose(imageFilePointer);
#endif

																																								
  
#ifndef BARE_METAL
  printf(" done!\n ======================= total success : %f%% ========================\n", (float)(100* success_count) / (float)(success_count + fail_count));
#else // Avoid float
    printf(" done!\n ======================= total success : %d%% ========================\n", (100*success_count) / (success_count + fail_count) );
#endif


bm_quit_app();  // uart message to simulation/pyshell to quit
return 0;
}

