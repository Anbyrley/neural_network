#ifndef HELPER_H
#define HELPER_H


//================================================================================================//
/**
* @brief This function prints a vector to stdout.
*
* If errors occur, the function exits.
*
* @param[in] double* vector
* @param[in] int vector_size
*
* @return NONE
*/
//================================================================================================//
void print_vector( double* vector,
				   int vector_size );


//================================================================================================//
/**
* @brief This function prints a matrix in row major order to stdout.
*
* If errors occur, the function exits.
*
* @param[in] double* matrix
* @param[in] int num_rows
* @param[in] int num_cols
*
* @return NONE
*/
//================================================================================================//
void print_matrix( double* matrix,
				   int num_rows,
				   int num_cols );


//================================================================================================//
/**
* @brief This function multiplies a vector by a matrix.
*
* If errors occur, the function exits.
*
* @param[in] double* vector
* @param[in] int vector_size
* @param[in] double* matrix
* @param[in] int matrix_rows
* @param[in] int matrix_columns
* @param[in,out] double* result
*
* @return NONE
*/
//================================================================================================//
void vector_matrix_multiply( double* vector,
							 int vector_size,
							 double* matrix,
							 int matrix_rows,
							 int matrix_columns,
							 double* result );


//================================================================================================//
/**
* @brief This function tests the vector_matrix_multiply
*
* If errors occur, the function exits.
*
* @return NONE
*/
//================================================================================================//
void test_vector_matrix_multiply();




#endif //HELPER_H//
