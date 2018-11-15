#include "neural_network.h"
#include "helper.c"

//================================================================================================//
//===================================Activation Functions=========================================//
//================================================================================================//

double pass_through(double x)
{
	return x;
}

double sigmoid(double x)
{
	return 1/(1+exp(-x));
}

double sigmoid_derivative(double x)
{
	return sigmoid(x)*(1-sigmoid(x));
}

//================================================================================================//
//===================================Neural Layer Functions=======================================//
//================================================================================================//

void initialize_neural_layer( neural_layer_t* self,
							  unsigned int num_nodes, 
							  neural_layer_t* previous_layer,
							  neural_layer_t* next_layer )
{

	//===Check Parameters===//
	if (num_nodes > MAX_LAYER_NODES){
		fprintf(stderr, "Error:: Input Parameter 'num_nodes' Is Invalid! In Function -- create_neural_layer\n");
		return;
	}
	
	//===Set Local Data===//
	self->num_nodes = num_nodes;
	self->previous_layer = previous_layer;
	self->next_layer = next_layer;

	//===Set Functions===//
	if (previous_layer == NULL){
		self->activate = &(pass_through);
		self->derivate = &(pass_through);
	}
	else{
		self->activate = &(sigmoid);
		self->derivate = &(sigmoid_derivative);
	}

	return;
}

void initialize_weight_matrix( neural_layer_t* self )
{

	unsigned int i,j;
	double temp;

	//===Initialize Input Weights===//
	for (i=0; i<self->num_nodes+1; i++){
		if (self->next_layer != NULL){
			for (j=0; j<self->next_layer->num_nodes; j++){
				temp = ((double)rand()/(double)(RAND_MAX)) * 1.0;
				self->weight_matrix[j + i*self->next_layer->num_nodes] = temp;			
			}
		}
		else{
			temp = ((double)rand()/(double)(RAND_MAX)) * 1.0;
			self->weight_matrix[i] = temp;			
		}
	}

	return;
}

void set_weight_matrix( neural_layer_t* self,
					    double* matrix )
{
	unsigned int i, j;

	//===Initialize Input Weights===//
	for (i=0; i<self->num_nodes+1; i++){
		if (self->next_layer != NULL){
			for (j=0; j<self->next_layer->num_nodes; j++){
				self->weight_matrix[j + i*self->next_layer->num_nodes] = matrix[j + i*self->next_layer->num_nodes];			
			}
		}
		else{
			self->weight_matrix[i] = matrix[i];			
		}
	}

	return;
}

void print_weight_matrix( neural_layer_t* self )
{
	if (self->next_layer != NULL){
		print_matrix(self->weight_matrix, self->num_nodes+1, self->next_layer->num_nodes);	
	}
	else{
		print_vector(self->weight_matrix, self->num_nodes+1);	
	}
	return;
}

void set_input( neural_layer_t* self,
				double* input )
{
	unsigned int i;
	
	for (i=0; i<self->num_nodes; i++){
		self->input[i] = input[i];
	}		
	
	return;
}

void print_input( neural_layer_t* self )
{
	print_vector(self->input, self->num_nodes);
	return;
}	

void set_delta( neural_layer_t* self,
				double* delta )
{
	unsigned int i;

	for (i=0; i<self->num_nodes; i++){
		self->delta[i] = delta[i];
	}

}

void print_delta( neural_layer_t* self )
{
	print_vector(self->delta, self->num_nodes);
	return;
}

void set_derivative( neural_layer_t* self,
					 double* derivative )
{
	unsigned int i;
	
	for (i=0; i<self->num_nodes; i++){
		self->derivative[i] = derivative[i];
	}
	return;
}

void print_derivative( neural_layer_t* self )
{
	print_vector(self->derivative, self->num_nodes);
	return;
}

void feed_layer_forward(neural_layer_t* self)
{
	unsigned int i;
	
	//===Set Input Activation===//
	for (i=0; i<self->num_nodes; i++){
		self->activation[i] = self->activate(self->input[i]);
		self->derivative[i] = self->derivate(self->input[i]);
	}
	self->activation[self->num_nodes] = 1;

	//===Pass To Next Layer===//
	if (self->next_layer != NULL){ 
		vector_matrix_multiply(self->activation, self->num_nodes+1,
							   self->weight_matrix, self->num_nodes+1, self->next_layer->num_nodes,
							   self->next_layer->input); 
	}

	return;
}

void feed_layer_backwards(neural_layer_t* self)
{

	unsigned int i;

	if (self->previous_layer != NULL){
		matrix_vector_multiply( self->delta,
								self->num_nodes,
								self->previous_layer->weight_matrix,
								self->previous_layer->num_nodes,
								self->num_nodes,
								self->previous_layer->delta );	
	
		//===Make Deltas===//
		for (i=0; i<self->previous_layer->num_nodes; i++){
			self->previous_layer->delta[i] *= self->previous_layer->derivative[i];
		}		
	}

	return;
}

void update_weight_matrix( neural_layer_t* self )
{

	if (self->next_layer != NULL){

		matrix_matrix_multiply(self->activation, self->num_nodes+1, 1,
							   self->next_layer->delta, 1, self->next_layer->num_nodes,
							   self->weight_update);

		matrix_update(self->weight_matrix, self->num_nodes+1, self->next_layer->num_nodes,
			  		  self->weight_update, -(*self->learning_rate));

	}


}

//================================================================================================//
//===================================Neural Network Functions=====================================//
//================================================================================================//

neural_network_parameters_t* create_neural_network_parameters( unsigned int num_hidden_layers,
															   unsigned int* num_nodes,
															   double learning_rate )
{
	unsigned int i;
	neural_network_parameters_t* self;

	if (num_hidden_layers > MAX_HIDDEN_LAYERS || num_hidden_layers < MIN_HIDDEN_LAYERS){
		fprintf(stderr, "Error:: Input Parameter 'num_hidden_layers' Is Invalid! In Function -- create_neural_network_parameters\n");
		return NULL;
	}
	if (num_nodes == NULL){
		fprintf(stderr, "Error:: Input Parameter 'num_nodes' Is NULL! In Function -- create_neural_network_parameters\n");
		return NULL;
	}
	else{
		for(i=0; i<num_hidden_layers+2; i++){
			if (num_nodes[i] == 0 || num_nodes[i] > MAX_LAYER_NODES){
				fprintf(stderr, "Error:: Input Parameter 'num_nodes[%d]' Is Invalid! In Function -- create_neural_network_parameters\n", i);
				return NULL;
			}
		}
	}

	self = NULL;
	self = malloc(sizeof(neural_network_parameters_t));
	if (self == NULL){
		fprintf(stderr, "Error:: Neural Network Parameters Was Not Allocated! In Function -- create_neural_network_parameters\n");
		return self;
	}
	
	//===Set Local Data===//
	self->num_hidden_layers = num_hidden_layers;
	for(i=0; i<num_hidden_layers+2; i++){
		self->num_nodes[i] = num_nodes[i];
	}	
	self->learning_rate = learning_rate;

	return self;
}


neural_network_t* create_neural_network( neural_network_parameters_t* parameters )
{

	unsigned int i;
	neural_network_t *self;
	neural_layer_t *previous_layer, *next_layer;
	self = NULL;
	self = malloc(sizeof(neural_network_t));
	if (self == NULL){
		fprintf(stderr, "Error:: Neural Network Was Not Allocated! In Function -- create_neural_network\n");
		return self;
	}	

	//===Initialize Random Number Generator===//
 	srand((unsigned int)time(NULL));

	//===Set Local Data===//
	self->num_hidden_layers = parameters->num_hidden_layers;
	self->input = self->layer[0].input;
	self->output = self->layer[self->num_hidden_layers+1].activation;
	self->error = self->layer[self->num_hidden_layers+1].delta;
	self->learning_rate = parameters->learning_rate;

	//===Create Layers===//
	for (i=0; i<self->num_hidden_layers+2; i++){

		if (i == 0){
			previous_layer = NULL;
			next_layer = &(self->layer[1]);
		}
		else if (i == self->num_hidden_layers+1){
			previous_layer = &(self->layer[i-1]);
			next_layer = NULL;
		}
		else{
			previous_layer = &(self->layer[i-1]);
			next_layer = &(self->layer[i+1]);
		}

		initialize_neural_layer(&(self->layer[i]), parameters->num_nodes[i], previous_layer, next_layer);
		self->layer[i].learning_rate = &(self->learning_rate);
	}

	for (i=0; i<self->num_hidden_layers+2; i++){
		//===Initialize Weight Matrix===//
		initialize_weight_matrix(&(self->layer[i]));
	}
	
	return self;
}

void print_weight_matrices( neural_network_t* self )
{
	unsigned int i;
	for (i=0; i<self->num_hidden_layers+2; i++){
		print_weight_matrix(&(self->layer[i]));
	}
}

void feed_forward( neural_network_t* self, 
				   double* input )
{
	unsigned int i;

	//===Set Input===//
	for (i=0; i<self->layer[0].num_nodes; i++){
		self->input[i] = input[i];
	}

	//===Feed Through Layers===//
	for (i=0; i<self->num_hidden_layers+2; i++){
		feed_layer_forward(&(self->layer[i]));
	}

	return;
}

void back_propagate( neural_network_t* self,
					 double* true_decision )
{
	unsigned int i;
	double temp;

	//===Create Error===//
	for (i=0; i<self->layer[self->num_hidden_layers+1].num_nodes; i++){
		temp = self->layer[self->num_hidden_layers+1].activation[i] - true_decision[i];
		self->error[i] = temp;
	}

	//===Feed Backwards===//
	for (i=self->num_hidden_layers+1; i>0; i--){
		feed_layer_backwards(&(self->layer[i]));
	}

	return;
}

void update_weights( neural_network_t* self )
{
	unsigned int i;
	for (i=0; i<self->num_hidden_layers+2; i++){
		update_weight_matrix(&(self->layer[i]));
	}

}

void iterate_network( neural_network_t* self,
					  double* input,
					  double* true_decision )
{

	//===Feed Forward===//
	feed_forward(self, input);

#if DEBUG
	unsigned int i;
	fprintf(stdout, "\n");
	for (i=0; i<self->num_hidden_layers+2; i++){
		fprintf(stdout, "Input %d:", i);
		print_vector(self->layer[i].input, self->layer[i].num_nodes);
	}

	fprintf(stdout, "\n");
	for (i=0; i<self->num_hidden_layers+2; i++){
		fprintf(stdout, "Activation %d:", i);
		print_vector(self->layer[i].activation, self->layer[i].num_nodes+1);
	}
#endif

	//===Back Propagation===//
	back_propagate(self, true_decision);

#if DEBUG
	fprintf(stdout, "\n");
	fprintf(stdout, "Error:");
	print_vector(self->error, self->layer[self->num_hidden_layers+1].num_nodes);


	fprintf(stdout, "\n");
	for (i=0; i<self->num_hidden_layers+2; i++){
		fprintf(stdout, "Delta %d:", i);
		print_vector(self->layer[i].delta, self->layer[i].num_nodes);
	}
#endif

	//===Update Weights===//
	update_weights(self);	

#if DEBUG
	fprintf(stdout, "\n");
	for (i=0; i<self->num_hidden_layers+2; i++){
		fprintf(stdout, "Weight Update %d:", i);
		if (i<self->num_hidden_layers+1){
			print_matrix(self->layer[i].weight_update, self->layer[i].num_nodes, self->layer[i].next_layer->num_nodes);
		}
		else{
			print_vector(self->layer[i].weight_update, self->layer[i].num_nodes);
		}
	}


	fprintf(stdout, "\n");
	for (i=0; i<self->num_hidden_layers+2; i++){
		fprintf(stdout, "Weight Matrix %d:", i);
		if (i<self->num_hidden_layers+1){
			print_matrix(self->layer[i].weight_matrix, self->layer[i].num_nodes, self->layer[i].next_layer->num_nodes);
		}
		else{
			print_vector(self->layer[i].weight_matrix, self->layer[i].num_nodes);
		}
	}
#endif

	return;
}


//================================================================================================//
//======================================Testing Functions=========================================//
//================================================================================================//

neural_network_t* create_test_neural_network()
{
	neural_network_t* network;
	neural_network_parameters_t* parameters;
	unsigned int num_nodes[4];
	double input_matrix[4*5];
	double hidden1_matrix[6*3];
	double hidden2_matrix[4*1];

	//===Set Data===//
	num_nodes[0] = 3; num_nodes[1] = 5; num_nodes[2] = 3; num_nodes[3] = 1;
	parameters = create_neural_network_parameters(2, num_nodes, 0.05);

	//===Create Network===//
	network = create_neural_network(parameters);	

	//===Create Input Matrix===//
	input_matrix[0] = 2; input_matrix[1] = 3; input_matrix[2] = 4; input_matrix[3] = 5; input_matrix[4] = 6;		
	input_matrix[5] = 4; input_matrix[6] = 5; input_matrix[7] = 6; input_matrix[8] = 7; input_matrix[9] = 8;
	input_matrix[10] = 5; input_matrix[11] = 6; input_matrix[12] = 7; input_matrix[13] = 8; input_matrix[14] = 9;
	input_matrix[15] = 1; input_matrix[16] = 1; input_matrix[17] = 1; input_matrix[18] = 1; input_matrix[19] = 1;
	set_weight_matrix(&(network->layer[0]),input_matrix);

	//===Create Hidden Layer Matrices===//	
	hidden1_matrix[0] = 3; hidden1_matrix[1] = 4; hidden1_matrix[2] = 5;	
	hidden1_matrix[3] = 5; hidden1_matrix[4] = 6; hidden1_matrix[5] = 7;
	hidden1_matrix[6] = 7; hidden1_matrix[7] = 8; hidden1_matrix[8] = 9;
	hidden1_matrix[9] = 9; hidden1_matrix[10] = 10; hidden1_matrix[11] = 11;
	hidden1_matrix[12] = 10; hidden1_matrix[13] = 12; hidden1_matrix[14] = 13;
	hidden1_matrix[15] = 1; hidden1_matrix[16] = 1; hidden1_matrix[17] = 1;
	set_weight_matrix(&(network->layer[1]),hidden1_matrix);

	hidden2_matrix[0] = 5; hidden2_matrix[1] = 6; hidden2_matrix[2] = 7; hidden2_matrix[3] = 1;
	set_weight_matrix(&(network->layer[2]),hidden2_matrix);

	return network;
}

void test_feed_forward(neural_network_t* self)
{
	double input[3];

	//===Create Input===//
	input[0] = 1; input[1] = 2; input[2] = 3;
	
	//===Feed Forward===//
	feed_forward(self, input);

	//===Print Inputs===//
	fprintf(stdout, "Inputs: \n");
	print_vector(self->layer[0].input, self->layer[0].num_nodes);
	print_vector(self->layer[1].input, self->layer[1].num_nodes);
	print_vector(self->layer[2].input, self->layer[2].num_nodes);
	print_vector(self->layer[3].input, self->layer[3].num_nodes);

	return;
}

void test_back_propagation(neural_network_t* self)
{
	unsigned int i, j;
	double temp;
	double delta2[3];
	double true_decision[1];

	//===Create Decision===//
	true_decision[0] = 0;

	//===Back Propagate===//
	for (i=0; i<self->num_hidden_layers+2; i++){
		for (j=0; j<self->layer[i].num_nodes; j++){
			self->layer[i].derivative[j] = 1;
		}
	}
	
	//===Create Error===//
	for (i=0; i<self->layer[self->num_hidden_layers+1].num_nodes; i++){
		temp = self->layer[self->num_hidden_layers+1].activation[i] - true_decision[i];
		self->error[i] = temp;
	}

	//===Feed Backwards===//
	delta2[0] = 1; delta2[1] = 2; delta2[2] = 3;
	for (i=self->num_hidden_layers+1; i>0; i--){
		if (i == self->num_hidden_layers){
			set_delta(&(self->layer[i]), delta2);
		}
		feed_layer_backwards(&(self->layer[i]));
	}


	//===Print Deltas===//
	fprintf(stdout, "\nDeltas: \n");
	print_vector(self->layer[1].delta, self->layer[1].num_nodes);
	print_vector(self->layer[2].delta, self->layer[2].num_nodes);
	print_vector(self->layer[3].delta, self->layer[3].num_nodes);


	return;
}

void test_weight_update(neural_network_t* self)
{
	unsigned int i, j;

	//===Set Activation To Inputs===//
	for (i=0; i<self->num_hidden_layers+2; i++){
		for (j=0; j<self->layer[i].num_nodes; j++){
			self->layer[i].activation[j] = self->layer[i].input[j];
		}
	}

	//===Update Weights===//
	update_weights(self);	

	//===Print Weight Updates===//
	fprintf(stdout, "\nWeight Updates: \n");
	print_matrix(self->layer[0].weight_update, 4, 5);
	print_matrix(self->layer[1].weight_update, 6, 3);
	print_matrix(self->layer[2].weight_update, 4, 1);

	return;
}	

void test_neural_network()
{

	neural_network_t* self;
	self = create_test_neural_network();

	//===Test Feed Forward===//
	test_feed_forward(self);

	//===Test Back Propagation===//
	test_back_propagation(self);

	//===Test Weight Update===//
	test_weight_update(self);

	return;
}
