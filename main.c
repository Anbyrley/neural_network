#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <float.h>
#include "neural_network.h"
#include "helper.h"


int main(void)
{

	#if UNIT_TESTS	
		test_neural_network();
	#else

		unsigned int num_nodes[MAX_LAYERS];
		neural_network_parameters_t* vad_parameters;
		neural_network_t* vad;

		//===Create Parameters===//
		num_nodes[0] = 3;
		num_nodes[1] = 5;
		num_nodes[2] = 3;
		num_nodes[3] = 1;
		vad_parameters = create_neural_network_parameters(2, num_nodes, 0.5);

		//===Create Network===//
		vad = create_neural_network(vad_parameters);

		//print_weight_matrices(vad);
		//exit(1);

		//===Feed Data To Network===//
		unsigned int i;
		char line[80];
		char string[80];
		double data[3];
		double decision;
		FILE *fp;

		//===Read Data===//
		fp = NULL;
		fp = fopen("2d_data.dat","r");	
		i = 0;
		while (fgets(line, 80, fp) != NULL && i < 40000){
			
			//===Get First Data Point===//
			strncpy(string, line, 7);
			data[0] = atof(string);

			//===Get Second Data Point===//
			strncpy(string, line+7, 7);
			data[1] = atof(string);

			//===Get Third Data Point===//
			strncpy(string, line+7+7, 7);
			data[2] = atof(string);

			//===Get Class Label===//
			strncpy(string, line+7+7+7, 5);
			strcat(string, "\n");
			decision = atof(string);

			//fprintf(stdout, "%+lf %+lf %+lf %+lf \n", data[0], data[1], data[2], decision);
#if DEBUG
			fprintf(stdout,"\nIteration: %d\n", i);
#endif
			iterate_network(vad, data, &(decision));	

			i++;	
		}

		fprintf(stdout, "\n\n");
		while (fgets(line, 80, fp) != NULL){

			//===Get First Data Point===//
			strncpy(string, line, 7);
			data[0] = atof(string);

			//===Get Second Data Point===//
			strncpy(string, line+7, 7);
			data[1] = atof(string);

			//===Get Third Data Point===//
			strncpy(string, line+7+7, 7);
			data[2] = atof(string);

			//===Get Class Label===//
			strncpy(string, line+7+7+7, 5);
			strcat(string, "\n");
			decision = atof(string);

			fprintf(stdout, "\nTrue Decision: %+lf \n", decision);
			feed_forward(vad, data);
			fprintf(stdout, "Network Decision: %+lf \n", vad->output[0]);

		}
		
			
	#endif


	return 0;
}

