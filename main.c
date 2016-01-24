//
//  main.c
//  Feed forward neural network for classification
//
//  Created by Robert Lösch on 11.11.15.
//  Copyright © 2015 Robert Lösch. All rights reserved.
//

#define IPATH  "testInput11A.txt"
#define OPATH "testOutput11A.txt"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_INPUT_NODES 2
#define NUM_HIDDEN_LAYER 1
#define NUM_OUTPUT_NODES 1

int num_nodes_per_hidden_layer[NUM_HIDDEN_LAYER] = {7};
double mx = 0, my = 0;

typedef int Tlabel;

typedef struct {
    double data[NUM_INPUT_NODES];
    Tlabel label;
} Training;

typedef struct {
    double data[NUM_INPUT_NODES];
} Test;

typedef double (*act_fct_ptr)(double);
typedef double (*act_fct_pr_ptr)(double);

typedef struct {
    act_fct_ptr act_fct;
    act_fct_pr_ptr act_fct_pr;
    int num_of_weights;
    double* weight;
    double weight_bias;
    double net;
    double out;
    double error;
} Neuron;

typedef struct {
    Neuron neuron[NUM_INPUT_NODES];
} Input_layer;

typedef struct {
    int num_of_nodes;
    Neuron* neuron;
    double* input;
} Hidden_layer;

typedef struct {
    Neuron neuron[NUM_OUTPUT_NODES];
    double* input;
} Output_layer;

typedef struct {
    Input_layer input;
    Hidden_layer hidden[NUM_HIDDEN_LAYER];
    Output_layer output;
} Neural_net;

void readInput(Training* trainset, int* length_training, Test* testset, int* length_test);
void readOutput(Training* outputset, int* length_output, Test* testset);
double calc_net_input(Neuron* n, double* data, int index);
double calc_net(Neuron* n, double* data, int length);
double activation_function(double net);
double activation_function_prime(double net);
void train(Neural_net* nn, Training* training_set, const int length);
double* feed_forward(Neural_net* nn, Test* test_set, int length, Training* output_set, int print);
void backpropagating(Neural_net* nn, double out, Training current_train);
void weight_update(Neural_net* nn, double eta, double* pchange, Training current_train);
void init_neuron(Neuron* n, int num_of_weights);
void init_neural_net(Neural_net* nn);
void destroy_neural_network(Neural_net* nn);

int main(int argc, const char * argv[]) {
    
    Training training_set[1000];
    Test test_set[1000];
    Training output_set[1000];
    int length_training = 0;
    int length_test = 0;
    int length_output = 0;
    
    readInput(training_set, &length_training, test_set, &length_test);
    readOutput(output_set, &length_output, test_set);

    Neural_net nn;
    int correct = 0, wrong = 0, count = 0;
    do {
        init_neural_net(&nn);
        double* outp;
        correct = 0;
        wrong = 0;
        for (int i = 0; i < length_training; i++) {
            Test tmp;
            tmp.data[0] = training_set[i].data[0];
            tmp.data[1] = training_set[i].data[1];
            outp = feed_forward(&nn, &tmp, 1, NULL, 0);
            
            if ( (*outp < 0 && training_set[i].label == -1) || (*outp > 0 && training_set[i].label == +1) )
                correct++;
            else
                wrong++;
        }
        count++;
        if (count >= 15000) break;
    } while ((double)correct/(correct+wrong) <= 0.7);
    printf("%i/%i = %lf (count:%i)\n", correct, correct+wrong, (double)correct/(correct+wrong), count);
  
    train(&nn, training_set, length_training);
    
    feed_forward(&nn, test_set, length_test, output_set, 1);
    
    destroy_neural_network(&nn);
    return 0;
}

void readInput(Training* trainset, int* length_training, Test* testset, int* length_test) {
    double x, y;
    Tlabel label;
    
    FILE *file;
    file = fopen(IPATH, "r");
    if (file) {
        while (fscanf(file, "%lf,%lf,%i\n",&x, &y, &label) == 3) {
            if (label != 0) {
                trainset[*length_training].data[0] = x;
                trainset[*length_training].data[1] = y;
                trainset[*length_training].label = label;
                (*length_training)++;
                if (fabs(x) > mx) {
                    mx = fabs(x);
                }
                if (y > fabs(my)) {
                    my = fabs(y);
                }
            } else {
                break;
            }
        }
            
        while(fscanf(file, "%lf,%lf\n",&x, &y) == 2) {
            testset[*length_test].data[0] = x;
            testset[*length_test].data[1] = y;
            (*length_test)++;
            if (fabs(x) > mx) {
                mx = fabs(x);
            }
            if (fabs(y) > my) {
                my = fabs(y);
            }
        }
        
        for (int i = 0; i < *length_training; i++) {
            trainset[i].data[0] /= mx;
            trainset[i].data[1] /= my;
        }
        for (int i = 0; i < *length_test; i++) {
            testset[i].data[0] /= mx;
            testset[i].data[1] /= my;
        }
    }
    fclose(file);
}

void readOutput(Training* outputset, int* length_output, Test* testset) {
    Tlabel label;
    
    FILE *file;
    file = fopen(OPATH, "r");
    if (file) {
        while (fscanf(file, "%i\n", &label) == 1) {
            outputset[*length_output].data[0] = testset[*length_output].data[0];
            outputset[*length_output].data[1] = testset[*length_output].data[1];
            outputset[*length_output].label = label;
            (*length_output)++;
        }
    }
    fclose(file);
}

double calc_net_input(Neuron* n, double* data, int index) {
    double result = n->weight[0]*data[index];
    result += n->weight_bias;
    return result;
}
        
double calc_net(Neuron* n, double* data, int length) {
    double result = 0.0;
    for (int i = 0; i < length; i++) {
        result += n->weight[i]*data[i];
    }
    result += n->weight_bias;
    return result;
}

double activation_function(double net) {
    return tanh(net);
}

double activation_function_prime(double net) {
    return (1 - tanh(net)*tanh(net));
}

void train(Neural_net* nn, Training* training_set, const int length) {
    double eta = 0.0005;
    double * outp = NULL;
    double change = 1.0;
    double thresh = 0.000015;
    int i = -1;
    int correct = 0, wrong = 0;
    
    while (fabs(change) > thresh) {
        i++;
        if (i%10000== 0) {
            printf("%3.0ik change: %.20lf eta: %lf  %i/%i\n", i/1000, change, eta, correct, correct+wrong);
        }
        correct = 0;
        wrong = 0;

        for (int i = 0; i < length; i++) {
            Test tmp;
            tmp.data[0] = training_set[i].data[0];
            tmp.data[1] = training_set[i].data[1];
            outp = feed_forward(nn, &tmp, 1, NULL, 0);

            if ( (outp[0] < 0 && training_set[i].label == -1) || (outp[0] > 0 && training_set[i].label == +1) )
                correct++;
            else
                wrong++;

            backpropagating(nn, outp[0], training_set[i]);
            weight_update(nn, eta, &change, training_set[i]);
        }
        eta *= 0.99999;
    }
    printf("%i change: %.20lf eta: %lf  %i/%i\n", i, change, eta, correct, correct+wrong);
}

double* feed_forward(Neural_net* nn, Test* test_set, int length, Training* output_set, int print) {
    int correct = 0, wrong = 0;
    
    for (int current_test_dataset = 0; current_test_dataset < length; current_test_dataset++) {

        // intput layer
        for (int in = 0; in < NUM_INPUT_NODES; in++) {
            Neuron* n = &nn->input.neuron[in];
            n->net = calc_net_input(n, test_set[current_test_dataset].data, in);
            n->out = (*n->act_fct)(n->net);
            nn->hidden[0].input[in] = n->out;
        }

        // first hidden layer
        for (int neur = 0; neur < num_nodes_per_hidden_layer[0]; neur++) {
            Neuron* n = &nn->hidden[0].neuron[neur];
            n->net = calc_net(n, nn->hidden[0].input, NUM_INPUT_NODES);
            n->out = (*n->act_fct)(n->net);
            if (NUM_HIDDEN_LAYER > 1) {
                nn->hidden[1].input[neur] = n->out;
            } else {
                nn->output.input[neur] = n->out;
            }
        }
        
        // other hidden layers
        for (int hid = 1; hid < NUM_HIDDEN_LAYER; hid++) {
            for (int neur = 0; neur < num_nodes_per_hidden_layer[hid]; neur++) {
                Neuron * n = &nn->hidden[hid].neuron[neur];
                n->net = calc_net(n, nn->hidden[hid].input, num_nodes_per_hidden_layer[hid-1]);
                n->out = (*n->act_fct)(n->net);
                if (hid < NUM_HIDDEN_LAYER -1) {
                    nn->hidden[hid+1].input[neur] = n->out;
                } else {
                    nn->output.input[neur] = n->out;
                }
            }
        }

        double* out = (double*)malloc(NUM_OUTPUT_NODES * sizeof(double));
        // output layer
        for (int k = 0; k < NUM_OUTPUT_NODES;k++) {
            Neuron* n = &nn->output.neuron[k];
            n->net = calc_net(n, nn->output.input, num_nodes_per_hidden_layer[NUM_HIDDEN_LAYER - 1]);
            n->out = (*n->act_fct)(n->net);
            out[k] = n->out;
        }
        
        if (print) {
            for (int k = 0; k < NUM_OUTPUT_NODES; k++) {
//                printf("out %2i: %lf", current_test_dataset, out[k]); printf(" (%i)\n", output_set[current_test_dataset].label);
                if ( (out[k] < 0 && output_set[current_test_dataset].label == -1) || (out[k] > 0 && output_set[current_test_dataset].label == +1) )
                    correct++;
                else
                    wrong++;
                if (out[k] < 0) printf("-1\n");
                if (out[k] > 0) printf("+1\n");
            }
        }
        if (current_test_dataset == length -1) {
            if (print)
                printf("%i/%i = %lf\n", correct, (correct + wrong), (double)correct/(correct+wrong));
            return out;
        }
        free(out);
    }
    return NULL;
}

void backpropagating(Neural_net* nn, double out, Training current_train) {
    
    // error for output neuron(s)
    for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
        Neuron* n = & nn->output.neuron[i];
        n->error = 2*((double)current_train.label - out); // 2 * (target_out - out)
    }
    
    // last hidden layer
    Hidden_layer* hl = &nn->hidden[NUM_HIDDEN_LAYER-1];
    for (int hidneur = 0; hidneur < num_nodes_per_hidden_layer[NUM_HIDDEN_LAYER-1]; hidneur++) {
        Neuron* n = &hl->neuron[hidneur];
        double sum = 0.0;
        for (int outneur = 0; outneur < NUM_OUTPUT_NODES; outneur++) {
            sum += nn->output.neuron[outneur].weight[hidneur]*nn->output.neuron[outneur].error;
        }
        n->error = (*n->act_fct_pr)(n->net)*sum;
    }
    
    // other hidden layers
    for (int i = NUM_HIDDEN_LAYER-2; i >= 0; i--) {
        Hidden_layer* hl = &nn->hidden[i];
        for (int j = 0; j < num_nodes_per_hidden_layer[i]; j++) {
            Neuron* n = &hl->neuron[j];
            double sum = 0.0;
            for (int k = 0; k < num_nodes_per_hidden_layer[i+1]; k++) {
                sum += nn->hidden[i+1].neuron[k].weight[j]*nn->hidden[i+1].neuron[k].error;
            }
            n->error = (*n->act_fct_pr)(n->net)*sum;
        }
    }
    
    // input layer
    for (int j = 0; j < NUM_INPUT_NODES; j++) {
        Neuron* n = &nn->input.neuron[j];
        double sum = 0.0;
        for (int k = 0; k < num_nodes_per_hidden_layer[0]; k++) {
            sum += nn->hidden[0].neuron[k].weight[j]*nn->hidden[0].neuron[k].error;
        }
        n->error = (*n->act_fct_pr)(n->net)*sum;
    }
}

void weight_update(Neural_net* nn, double eta, double* pchange, Training current_train) {
    
    double max_change = 0.0;
    double change = 0.0;
    
    // error for output neuron(s)
    for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
        Neuron* n = & nn->output.neuron[i];
        // weight update
        for (int w = 0; w < n->num_of_weights; w++) {
            change = eta * n->error * nn->output.input[w];
            n->weight[w] += change;
            if (fabs(change) > max_change) max_change = fabs(change);
        }
        change = eta * nn->output.neuron[i].error;
        nn->output.neuron[i].weight_bias += change;
        if (fabs(change) > max_change) max_change = fabs(change);
    }
    
    // last hidden layer
    Hidden_layer* hl = &nn->hidden[NUM_HIDDEN_LAYER-1];
    for (int hidneur = 0; hidneur < num_nodes_per_hidden_layer[NUM_HIDDEN_LAYER-1]; hidneur++) {
        Neuron* n = &hl->neuron[hidneur];
        // weight update
        for (int w = 0; w < n->num_of_weights; w++) {
            change = eta * n->error * hl->input[w];
            n->weight[w] += change;
            if (fabs(change) > max_change) max_change = fabs(change);
        }
        change = eta * n->error;
        n->weight_bias += change;
        if (fabs(change) > max_change) max_change = fabs(change);
    }
    
    // other hidden layers
    for (int i = NUM_HIDDEN_LAYER-2; i >= 0; i--) {
        Hidden_layer* hl = &nn->hidden[i];
        for (int j = 0; j < num_nodes_per_hidden_layer[i]; j++) {
            Neuron* n = &hl->neuron[j];
            //weight update
            for (int w = 0; w < n->num_of_weights; w++) {
                change = eta * n->error * hl->input[w];
                n->weight[w] += change;
                if (fabs(change) > max_change) max_change = fabs(change);
            }
            change = eta * n->error;
            n->weight_bias += change;
            if (fabs(change) > max_change) max_change = fabs(change);
        }
    }
    
    // input layer
    for (int j = 0; j < NUM_INPUT_NODES; j++) {
        Neuron* n = &nn->input.neuron[j];
        // weight update
        change = eta * n->error*current_train.data[j];
        n->weight[0] += change;
        if (fabs(change) > max_change) max_change = fabs(change);
        change = eta * n->error;
        n->weight_bias += change;
        if (fabs(change) > max_change) max_change = fabs(change);
    }
    *pchange = max_change;
}
        
void init_neuron(Neuron* n, int num_of_weights) {
    srand((unsigned)time(NULL)+(unsigned)rand());
    n->weight = (double*)malloc(num_of_weights * sizeof(double));
    n->num_of_weights = num_of_weights;
    
    for (int i = 0; i < n->num_of_weights; i++) {
        n->weight[i] = rand()/(double)RAND_MAX - 0.5; // random number in [-0.5,0.5]
    }
    n->weight_bias = rand()/(double)RAND_MAX - 0.5;
    
    n->act_fct = &activation_function;
    n->act_fct_pr = &activation_function_prime;
    n->net = 0.0;
    n->out = 0.0;
    n->error = 0.0;
    return;
}

void init_neural_net(Neural_net* nn) {
    // intput layer
    for (int i = 0; i < NUM_INPUT_NODES; i++) {
        init_neuron(&nn->input.neuron[i], 1);
    }

    // first hidden layer
    nn->hidden[0].num_of_nodes = num_nodes_per_hidden_layer[0];
    nn->hidden[0].neuron = (Neuron*)malloc(nn->hidden[0].num_of_nodes * sizeof(Neuron));
    for (int j = 0; j < nn->hidden[0].num_of_nodes; j++) {
        init_neuron(&nn->hidden[0].neuron[j], NUM_INPUT_NODES);
    }
    nn->hidden[0].input = (double*)malloc(NUM_INPUT_NODES*sizeof(double));
    
    // hidden layer
    for (int i = 1; i < NUM_HIDDEN_LAYER; i++) {
        nn->hidden[i].num_of_nodes = num_nodes_per_hidden_layer[i];
        nn->hidden[i].neuron = (Neuron*)malloc(nn->hidden[i].num_of_nodes * sizeof(Neuron));
        for (int j = 0; j < nn->hidden[i].num_of_nodes; j++) {
            init_neuron(&nn->hidden[i].neuron[j], num_nodes_per_hidden_layer[i-1]);
        }
        nn->hidden[i].input = (double*)malloc(num_nodes_per_hidden_layer[i-1]*sizeof(double));
    }
    // output layer
    for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
        init_neuron(&nn->output.neuron[i], num_nodes_per_hidden_layer[NUM_HIDDEN_LAYER-1]);
    }
    nn->output.input = (double*)malloc(num_nodes_per_hidden_layer[NUM_HIDDEN_LAYER-1]*sizeof(double));
}

void destroy_neural_network(Neural_net* nn) {
    // hidden layer
    for (int i = 0; i < NUM_HIDDEN_LAYER; i++) {
        free(nn->hidden[i].neuron);
        free(nn->hidden[i].input);
    }
}