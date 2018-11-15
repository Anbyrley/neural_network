#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include "cblas.h"


//================================================================================================//
//===========================================MACROS===============================================//
//================================================================================================//

#define BLURT printf ("This is line %d of file %s (function %s)\n",\
                      __LINE__, __FILE__, __func__)

#define UNIT_TESTS 0
#define DEBUG 0

#define MAX_LAYER_NODES 10
#define MIN_HIDDEN_LAYERS 1
#define MAX_HIDDEN_LAYERS 10
#define MAX_LAYERS MAX_HIDDEN_LAYERS+2

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))



//================================================================================================//
//======================================Data Structures===========================================//
//================================================================================================//

//================================================================================================//
/** @struct neural_layer_t
*   @brief This structure comprises the functionality of a neural network layer.
*
*/
//================================================================================================//
typedef struct neural_layer_s neural_layer_t;
typedef struct neural_layer_s{
	double input[MAX_LAYER_NODES];
	double activation[MAX_LAYER_NODES+1];
	double derivative[MAX_LAYER_NODES+1];
	double weight_matrix[(MAX_LAYER_NODES+1)*(MAX_LAYER_NODES+1)];
	double weight_update[(MAX_LAYER_NODES+1)*(MAX_LAYER_NODES+1)];
	double delta[MAX_LAYER_NODES];
	neural_layer_t* previous_layer;
	neural_layer_t* next_layer;
	double (*activate)(double);
	double (*derivate)(double);
	double* learning_rate;
	unsigned int num_nodes;
} neural_layer_t;


//================================================================================================//
/** @struct neural_network_t
*   @brief This structure comprises the functionality of a neural network.
*
*	This object coordinates the activities of multiple neural_layer_t objects.
*/
//================================================================================================//
typedef struct neural_network_s neural_network_t;
typedef struct neural_network_s{
	neural_layer_t layer[MAX_HIDDEN_LAYERS+2];
	double* input;
	double* output;
	double* error;
	double learning_rate;
	unsigned int num_hidden_layers;
} neural_network_t;


//================================================================================================//
/** @struct neural_network_parameters_t
*   @brief This structure comprises the creation parameters of a neural network.
*
*	Use this structure to initialize a neural network.
*/
//================================================================================================//
typedef struct neural_network_parameters_s neural_network_parameters_t;
typedef struct neural_network_parameters_s{
	unsigned int num_nodes[MAX_HIDDEN_LAYERS+2];
	unsigned int num_hidden_layers;
	double learning_rate;
} neural_network_parameters_t; 



//================================================================================================//
//===================================Function Definitions=========================================//
//================================================================================================//


//================================================================================================//
/**
* @brief This function initializes a neural_layer_t object.
*
* If errors occur, the function exits.
*
* @param[in,out] neural_layer_t* self
* @param[in] unsigned int num_nodes
* @param[in] neural_layer_t* previous_layer
* @param[in] neural_layer_t* next_layer
*
* @return neural_layer_t* self
*/
//================================================================================================//
void initialize_neural_layer(neural_layer_t*, unsigned int, neural_layer_t*, neural_layer_t*);


//================================================================================================//
/**
* @brief This function initializes a neural_layer_t's weight matrix.
*
* If errors occur, the function exits.
*
* @param[in,out] neural_layer_t* self
*
* @return NONE
*/
//================================================================================================//
void initialize_weight_matrix(neural_layer_t*);


void print_weight_matrix( neural_layer_t* self );

//================================================================================================//
/**
* @brief This function sets a neural_layer_t's weight matrix.
*
* If errors occur, the function exits.
*
* @param[in,out] neural_layer_t* self
* @param[in] double* matrix
*
* @return NONE
*/
//================================================================================================//
void set_weight_matrix(neural_layer_t*, double*);


//================================================================================================//
/**
* @brief This function feeds data through a neural_layer_t forward.
*
* If errors occur, the function exits.
*
* @param[in,out] neural_layer_t* self
*
* @return NONE
*/
//================================================================================================//
void feed_layer_forward(neural_layer_t*);


//================================================================================================//
/**
* @brief This function allocates a neural_network_parameters_t object.
*
* If errors occur, the function exits.
*
* @param[in] int num_hidden_layers
* @param[in] int* num_nodes
* @param[in] double learning_rate
*
* @return neural_network_parameters_t* self
*/
//================================================================================================//
neural_network_parameters_t* create_neural_network_parameters(unsigned int, unsigned int*, double);


//================================================================================================//
/**
* @brief This function allocates a neural_network_t object.
*
* If errors occur, the function exits.
*
* @param[in] neural_network_parameters_t* self
*
* @return neural_network_t* self
*/
//================================================================================================//
neural_network_t* create_neural_network(neural_network_parameters_t*);


void print_weight_matrices(neural_network_t*);


//================================================================================================//
/**
* @brief This function runs an update iteration for the neural_network_t object.
*
* If errors occur, the function exits.
*
* @param[in,out] neural_network_t* self
* @param[in] double* input
* @param[in] double* true_decision
*
* @return NONE
*/
//================================================================================================//
void iterate_network(neural_network_t*, double*, double*);


void feed_forward( neural_network_t* self, 
				   double* input );

//================================================================================================//
/**
* @brief This function runs the unit test for the neural_network_t object
*
* If errors occur, the function exits.
*
* @return NONE
*/
//================================================================================================//
void test_neural_network();



#endif //NEURAL_NETWORK_H//
