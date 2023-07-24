#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MNIST.h"
#include <math.h>

//splits line up for csv parsing
const char* getfield(char* line, int num) {
    const char* tok;
    for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
        if (!--num)
            return tok;
    }
    return NULL;
}

//the softmax function takes an array Z as input and computes the softmax activation, storing the result in the A array. 
void softmax(double *Z, double *A, int size) {
    double max_val = Z[0];
    for (int i = 1; i < size; i++) {
        if (Z[i] > max_val) {
            max_val = Z[i];
        }
    }

    double exp_sum = 0.0;
    for (int i = 0; i < size; i++) {
        A[i] = exp(Z[i] - max_val);
        exp_sum += A[i];
    }

    for (int i = 0; i < size; i++) {
        A[i] /= exp_sum;
    }
}

// returns random double lol
double random_double() {
    return (double)rand() / (double)RAND_MAX - 0.5;
}

//ReLU (Rectified Linear Unit)
double ReLU(double z) {
    return fmax(0.0, z);
}

//initiate parameters for forward propagation
//for our dataset rows = 10, colums = 784
void init_params(double **W1, double **b1, double **W2, double **b2, int rows, int cols) {
    *W1 = (double *)malloc(sizeof(double) * rows * cols);
    *b1 = (double *)malloc(sizeof(double) * rows);
    *W2 = (double *)malloc(sizeof(double) * rows * rows);
    *b2 = (double *)malloc(sizeof(double) * rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*W1)[i * cols + j] = random_double();
        }
        (*b1)[i] = random_double();
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            (*W2)[i * rows + j] = random_double();
        }
        (*b2)[i] = random_double();
    }
}

// ReLU derivative function
// MAY NEED TO change
double ReLU_deriv(double Z) {
    return Z > 0 ? 1 : 0;
}

// Function to create one-hot encoded labels from an array Y
double* one_hot(int label) {
    double* one_hot_Y = (double*)calloc(10, sizeof(double));
    one_hot_Y[label] = 1.0;
    return one_hot_Y;
}

//function to perform dot product, this only works with non-adjustable ANN
double dot(double vect_A[], double vect_B[], int n) {
    double product = 0;
    for (int i = 0; i < n; i++) {
        product = product + vect_A[i] * vect_B[i];
    }
    return product;
}

// 
void forward_prop(double* W1, double* b1, double* W2, double* b2, double* X,
                  double* Z1, double* A1, double* Z2, double* A2) {
    int input_size = 784; // Number of input features
    int num_classes = 10; // Number of output classes
    // Forward propagation for the first layer (only one hidden unit)
    Z1[0] = 0.0;
    for (int k = 0; k < input_size; k++) {
        Z1[0] += W1[k] * X[k];
    }
    Z1[0] += b1[0];
    A1[0] = Z1[0] > 0 ? Z1[0] : 0;

    // Forward propagation for the second layer
    for (int j = 0; j < num_classes; j++) {
        Z2[j] = 0.0;
        Z2[j] += W2[j] * A1[0];
        Z2[j] += b2[j];
    }

    // Apply softmax activation function to get the final output probabilities
    double max_val = Z2[0];
    for (int j = 1; j < num_classes; j++) {
        if (Z2[j] > max_val) {
            max_val = Z2[j];
        }
    }
    double sum_exp = 0.0;
    for (int j = 0; j < num_classes; j++) {
        A2[j] = exp(Z2[j] - max_val);
        sum_exp += A2[j];
    }
    for (int j = 0; j < num_classes; j++) {
        A2[j] /= sum_exp;
    }
}
// Function for backward propagation
void backward_prop(double* Z1, double* A1, double* Z2, double* A2, double* W1, double* W2, double* X,
                   int Y, double* dW1, double* db1, double* dW2, double* db2) {

    int input_size = 784; 
    int num_classes = 10;

    // Step 1: Compute the one-hot encoded labels for Y (similar to the one_hot function provided)
    double* one_hot_Y = (double*)calloc(num_classes, sizeof(double));
    int class_index = Y;
    one_hot_Y[class_index] = 1;

    // Step 2: Compute dZ2 (derivative of the softmax with respect to Z2)
    double* dZ2 = (double*)malloc(sizeof(double) * num_classes);
    for (int j = 0; j < num_classes; j++) {
        dZ2[j] = A2[j] - one_hot_Y[j];
    }

    // Step 3: Compute dW2 and db2 (gradients for weights and biases of the second layer)
    for (int i = 0; i < num_classes; i++) {
        db2[i] = dZ2[i];
        dW2[i] = dZ2[i] * A1[0];
    }

    // Step 4: Compute dZ1 (derivative of the ReLU with respect to Z1)
    double dZ1 = ReLU_deriv(Z1[0]);

    // Step 5: Compute dW1 and db1 (gradients for weights and biases of the first layer)
    db1[0] = dZ1;
    for (int j = 0; j < input_size; j++) {
        dW1[j] = dZ1 * X[j];
    }

    // Step 6: Free memory
    free(one_hot_Y);
    free(dZ2);
}
void update_params(double* W1, double* b1, double* W2, double* b2, double* dW1, double* db1, double* dW2, double* db2, double alpha) {
    // Update W1, b1, W2, and b2 with the computed gradients using the learning rate (alpha)
    int input_size = 784; // Number of input features
    int num_classes = 10; // Number of output classes

    // Update W1 and b1
    for (int j = 0; j < input_size; j++) {
        W1[j] -= alpha * dW1[j];
    }

    *b1 -= alpha * (*db1);

    // Update W2 and b2
    for (int i = 0; i < num_classes; i++) {
        W2[i] -= alpha * dW2[i];
    }

    for (int i = 0; i < num_classes; i++) {
        b2[i] -= alpha * db2[i];
    }
}
int get_prediction(double* A2, int num_classes) {
    int predicted_class = 0;
    double max_prob = A2[0];
    for (int i = 1; i < num_classes; i++) {
        if (A2[i] > max_prob) {
            max_prob = A2[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

// Function to calculate accuracy
double get_accuracy(int* predictions, int* Y, int m) {
    int correct_predictions = 0;
    for (int i = 0; i < m; i++) {
        if (predictions[i] == Y[i]) {
            correct_predictions++;
        }
    }
    return (double)correct_predictions / m;
}


struct input* testCSV(){
    FILE* fp;
    int rowCount = 0; // Initialize rowCount to 0

    // Open the CSV file
    fp = fopen("TT.csv", "r");
    if (fp == NULL) {
        printf("Failed to open the CSV file.\n");
    }

    // Count the number of newline characters (rows) in the CSV file
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '\n') {
            rowCount++;
        }
    }
    //printf("%d\n", rowCount);
    
    // Rewind the file pointer to the beginning
    rewind(fp);

    // Create an array of input structs to store the data
    struct input* data = (struct input*)malloc(rowCount * sizeof(struct input));
    

    // Check if memory allocation was successful
    if (data == NULL) {
        printf("Memory allocation failed.\n");
        fclose(fp);
    }

    // This loads data from CSV fill into an array of structs
    char line[7000];
    int k = 1;
    while (fgets(line, sizeof(line), fp)) {
        char* tmp = strdup(line);
        data[k].label = atoi(getfield(tmp, 1));
        data[k].pixel = (double*)malloc(sizeof(double) * 784);
        int x = 0;

        for (int i = 0; i < 784; i++) { // Fix the loop range to read 784 pixel values
            char* tmp2 = strdup(line);
            x = i + 2;
            data[k].pixel[i] = atoi(getfield(tmp2, x))/255.0; // normalizes and assigns data
            free(tmp2);
            if(k%1000==0){
                //printf("Line: %d Data from Pixel: %f\n",k,data[k].pixel[i]);
                //printf("Line: %d, Pixel: %d\n",k,i);
            }
            
        }
        free(tmp);   
        k++;
    }
    // Close the file after reading and storing data
    fclose(fp);
    return data;

    }

struct input* inpCSV(){
    FILE* fp;
    int rowCount = 0; // Initialize rowCount to 0

    // Open the CSV file
    fp = fopen("train.csv", "r");
    if (fp == NULL) {
        printf("Failed to open the CSV file.\n");
    }

    // Count the number of newline characters (rows) in the CSV file
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '\n') {
            rowCount++;
        }
    }
    //printf("%d\n", rowCount);
    
    // Rewind the file pointer to the beginning
    rewind(fp);

    // Create an array of input structs to store the data
    struct input* data = (struct input*)malloc(rowCount * sizeof(struct input));
    

    // Check if memory allocation was successful
    if (data == NULL) {
        printf("Memory allocation failed.\n");
        fclose(fp);
    }

    // This loads data from CSV fill into an array of structs
    char line[7000];
    int k = 0;
    while (fgets(line, sizeof(line), fp)) {
        char* tmp = strdup(line);
        data[k].label = atoi(getfield(tmp, 1));
        data[k].pixel = (double*)malloc(sizeof(double) * 784);
        int x = 0;

        for (int i = 0; i < 784; i++) { // Fix the loop range to read 784 pixel values
            char* tmp2 = strdup(line);
            x = i + 2;
            data[k].pixel[i] = atoi(getfield(tmp2, x))/255.0; // normalizes and assigns data
            free(tmp2);
            if(k%1000==0){
                //printf("Line: %d Data from Pixel: %f\n",k,data[k].pixel[i]);
                printf("Line: %d, Pixel: %d\n",k,i);
            }
            
        }
        free(tmp);   
        k++;
    }
    // Close the file after reading and storing data
    fclose(fp);
    return data;

    }
           
void freedata(struct input* data){
            FILE* fp;
    int rowCount = 0; // Initialize rowCount to 0

    // Open the CSV file
    fp = fopen("train.csv", "r");
    if (fp == NULL) {
        printf("Failed to open the CSV file.\n");
    }

    // Count the number of newline characters (rows) in the CSV file
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '\n') {
            rowCount++;
        }
    }
        for (int i = 0; i < rowCount-1; i++) {
        free(data[i].pixel);
    }

    // Free the memory allocated for the array of structs
    free(data);
    fclose(fp);
}
// Function to get the index of the maximum element in an array A2

// Function to perform gradient descent
void gradient_descent(double alpha, int iterations) {
    int m = iterations; // Number of samples
    int input_size = 784; // Number of input features
    int hidden_units = 1; // Number of hidden units
    int num_classes = 10; // Number of output classes

    struct input* data = inpCSV();

    // Allocate memory for Z1, A1, Z2, and A2
    double* Z1 = (double*)malloc(sizeof(double) * hidden_units);
    double* A1 = (double*)malloc(sizeof(double) * hidden_units);
    double* Z2 = (double*)malloc(sizeof(double) * num_classes);
    double* A2 = (double*)malloc(sizeof(double) * num_classes);
    double* dW1 = (double*)malloc(sizeof(double) * hidden_units * input_size);
    double* db1 = (double*)malloc(sizeof(double) * hidden_units);
    double* dW2 = (double*)malloc(sizeof(double) * num_classes * hidden_units);
    double* db2 = (double*)malloc(sizeof(double) * num_classes);
    int* Y = (int*)malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) {
        Y[i] = data[i].label;
    }
    int* P = (int*)malloc(sizeof(int) * m);
    int* k = (int*)malloc(sizeof(int) * m);

    // Initialize parameters using init_params function
    double *W1, *b1, *W2, *b2;
    init_params(&W1, &b1, &W2, &b2, hidden_units, input_size);

    for (int i = 0; i < iterations; i++) {
        forward_prop(W1, b1, W2, b2, data[i].pixel, Z1, A1, Z2, A2);
        backward_prop(Z1, A1, Z2, A2, W1, W2, data[i].pixel, Y[i], dW1, db1, dW2, db2);
        update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
        P[i%10] = get_prediction(A2, num_classes);
        k[i%10] = Y[i];
        if (i % 10 == 0) {
            printf("Iteration: %d\n", i);
            double accuracy = get_accuracy(P, k, i + 1);
            printf("Accuracy: %.2f%%\n", accuracy * 100);
        }
    }

    // Free memory
    freedata(data);
    free(W1);
    free(b1);
    free(W2);
    free(b2);
    free(Z1);
    free(A1);
    free(Z2);
    free(A2);
    free(dW1);
    free(db1);
    free(dW2);
    free(db2);
    free(Y);
    free(k);
    free(P);
}
int main() {

    //print data from arays befor freeing

    // Print the initialized parameters for testing
/*
printf("W1:\n");
for (int i = 0; i < hidden_units; i++) {
    for (int j = 0; j < input_size; j++) {
        printf("%f ", W1[i * input_size + j]);
    }
    printf("\n");
}

printf("\nb1:\n");
for (int i = 0; i < hidden_units; i++) {
    printf("%f\n", b1[i]);
}

printf("\nW2:\n");
for (int i = 0; i < num_classes; i++) {
    for (int j = 0; j < hidden_units; j++) {
        printf("%f ", W2[i * hidden_units + j]);
    }
    printf("\n");
}

printf("\nb2:\n");
for (int i = 0; i < num_classes; i++) {
    printf("%f\n", b2[i]);
}

// Print Z1, A1, Z2, and A2 before freeing their memory
printf("\nZ1:\n");
for (int i = 0; i < m * hidden_units; i++) {
    printf("%f ", Z1[i]);
}
printf("\n");

printf("\nA1:\n");
for (int i = 0; i < m * hidden_units; i++) {
    printf("%f ", A1[i]);
}
printf("\n");

printf("\nZ2:\n");
for (int i = 0; i < m * num_classes; i++) {
    printf("%f ", Z2[i]);
}
printf("\n");

printf("\nA2:\n");
for (int i = 0; i < m * num_classes; i++) {
    printf("%f ", A2[i]);
}
printf("\n");

*/

    int iterations = 5000;
    double alpha = 0.1;
    gradient_descent(alpha, iterations);
    return 0;
    return 0;
}