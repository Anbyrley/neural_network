//================================================================================================//
//======================================Helper Functions==========================================//
//================================================================================================//

void print_vector( double* vector,
				   int vector_size )
{
	int i;
	fprintf(stdout, "\n");
	for (i=0; i<vector_size; i++){
		fprintf(stdout, "%+lf ", vector[i]);
	}
	fprintf(stdout, "\n");
	
	return;
}

void print_matrix( double* matrix,
				   int num_rows,
				   int num_cols )
{
	int i, j;
	fprintf(stdout, "\n");
	for (i=0; i<num_rows; i++){
		for (j=0; j<num_cols; j++){
			fprintf(stdout, "%+lf ", matrix[j + i*num_cols]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");

	return;
}

void vector_matrix_multiply( double* vector,
							 int vector_size,
							 double* matrix,
							 int matrix_rows,
							 int matrix_columns,
							 double* result )
{

	if (matrix_rows != vector_size){
		fprintf(stderr, "Vector-Matrix Sizes Are Incompatible In Function -- vector_matrix_multiply\n");
		return;
	}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				1, matrix_columns, vector_size, 
				1.0, vector, vector_size, 
				matrix, matrix_columns, 
				0.0, result, matrix_columns);
	
	return;
}

void test_vector_matrix_multiply()
{
	double vector[4];
	double matrix[4*5];
	double result[5];

	//===Make Vector===//
	vector[0] = 1;
	vector[1] = 2;
	vector[2] = 3;
	vector[3] = 4;
	
	//===Make Matrix===//
	matrix[0] = 1; matrix[1] = 2; matrix[2] = 3; matrix[3] = 4; matrix[4] = 5; 
	matrix[5] = 6; matrix[6] = 7; matrix[7] = 8; matrix[8] = 9; matrix[9] = 10; 
	matrix[10] = 11; matrix[11] = 12; matrix[12] = 13; matrix[13] = 14; matrix[14] = 15; 
	matrix[15] = 16; matrix[16] = 17; matrix[17] = 18; matrix[18] = 19; matrix[19] = 20;


	vector_matrix_multiply( vector,
							4,
							matrix,
							4,
							5,
							result);

	if ( (int)result[0] != 110 ||
		 (int)result[1] != 120 ||
		 (int)result[2] != 130 ||
		 (int)result[3] != 140 ||
		 (int)result[4] != 150 ){
		fprintf(stderr, "Error: Function vector_matrix_multiply Has Failed!\n");
	} 

	print_matrix(matrix, 4, 5);
	fprintf(stdout, "\n");

	return;
}

void matrix_vector_multiply( double* vector,
							 int vector_size,
							 double* matrix,
							 int matrix_rows,
							 int matrix_columns,
							 double* result )
{


	if (matrix_columns != vector_size){
		fprintf(stderr, "Matrix-Vector Sizes Are Incompatible In Function -- matrix_vector_multiply\n");
		return;
	}


 //zgemv (transa, 
  //       m,     //cblas -- number of rows in A //acml -- number of columns
  //       n,     //cblas -- number of columns in A// acml -- number of rows
  //       alpha, //A constant
  //       a,     //A itself
  //       lda,   //leading dimension of A cblas -- columns of A // acml -- rows of A
  //       x,     //vector x itself
  //       incx,  //spacing between elements of x
  //       beta,  //Another constant set it to zero
  //       y,     //result vector
  //       incy); //spacing between results
 
	cblas_dgemv(CblasRowMajor,
		         CblasNoTrans, matrix_rows, matrix_columns,
		         1.0, matrix, matrix_columns,
		         vector, 1, 0.0,
		         result, 1);

	return;
}


void test_matrix_vector_multiply()
{

	double vector[3];
	double matrix[2*3];
	double result[2];

	//===Make Vector===//
	vector[0] = 2;
	vector[1] = 1;
	vector[2] = 0;
	
	//===Make Matrix===//
	matrix[0] = 1; matrix[1] = -1; matrix[2] = 2; 
	matrix[3] = 0; matrix[4] = -3; matrix[5] = 1;  


	matrix_vector_multiply( vector,
							3,
							matrix,
							2,
							3,
							result );


	if ( (int)result[0] != 1 || (int)result[1] != -3){
		fprintf(stderr, "Error: Function matrix_vector_multiply Has Failed!\n");
	} 

 	//print_vector( result, 2 );
	//print_matrix(matrix, 2, 3);
	//fprintf(stdout, "\n");

	return;
}

void matrix_matrix_multiply( double* matrix1,
							 int matrix1_rows,
							 int matrix1_columns,
							 double* matrix2,
							 int matrix2_rows,
							 int matrix2_columns,
							 double* result )
{

	if (matrix1_columns != matrix2_rows){
		fprintf(stderr, "Matrix-Matrix Sizes Are Incompatible In Function -- matrix_matrix_multiply\n");
		return;
	}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				matrix1_rows, matrix2_columns, matrix1_columns, 
				1.0, matrix1, matrix1_columns, 
				matrix2, matrix2_columns, 
				0.0, result, matrix2_columns);
	
	return;
}


void test_matrix_matrix_multiply()
{

	double matrix1[4*1];
	double matrix2[1*3];
	double result[4*3];

	//===Create First Matrix===//
	matrix1[0] = 1;
	matrix1[1] = 2;
	matrix1[2] = 3;
	matrix1[3] = 4;

	//===Create Second Matrix===//
	matrix2[0] = 5; matrix2[1] = 6; matrix2[2] = 7;

	//===Run Multiply===//
	matrix_matrix_multiply(matrix1, 4, 1, matrix2, 1, 3, result);
	
	//===Test Result===//
	if ( (int)result[0] != 5 || (int)result[1] != 6 || (int)result[2] != 7 ||
		 (int)result[3] != 10 || (int)result[4] != 12 || (int)result[5] != 14 ||
		 (int)result[6] != 15 || (int)result[7] != 18 || (int)result[8] != 21 ||
		 (int)result[9] != 20 || (int)result[10] != 24 || (int)result[11] != 28){
		fprintf(stderr, "Error: Function matrix_matrix_multiply Has Failed!\n");
	} 


	return;
}


void matrix_update( double* matrix,
					int matrix_rows,
					int matrix_columns,
					double* update,
					double update_weight )
{

	int i, j;
	for (i=0; i<matrix_rows; i++){
		for (j=0; j<matrix_columns; j++){
			matrix[j + i*matrix_columns] += update_weight * update[j + i*matrix_columns];
		}
	}

	return;
}
