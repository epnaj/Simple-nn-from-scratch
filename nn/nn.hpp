#ifndef nn_hpp
#define nn_hpp

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <cmath>

using namespace std;

double sigmoid(double x);
double ReLU(double x);
double sigmoidDerivativeUsingSourceVal(double x);
double sigmoidDerivative(double x);

class neuron{
    public:
    vector <double> weights;
    double val;
    double bias;
    double d;
    neuron(int nonb);
    ~neuron();
};

class layer{
    public:
    vector <neuron *> lay;
    layer(int nOfNeurons, int nonb);
    ~layer();
};

class nnet{
    private:
    double learningRate = 0.01;
    vector <layer *> net;
    public:
    nnet(vector <int> &n_struct);
    ~nnet();
    void printNet();
    void saveNet(string name);
    void forward(vector <double> &data);
    void back_d(vector <double> &error);
    void for_d();
    vector <double> softmax();
    int max_of_last_lay_inx();
    int max_of_softmax();
    vector <double> error_last(vector <double> &target);
    double cost_last_lay(vector <double> &target);
    void out_last_lay();
    void train(vector<double> &data, vector<double> &target);
    friend nnet loadNet(string name);
};

struct vectors{
    vector <double> error;
    vector <double> soft;
    vector <double> target; 
    vector <int> n_struct;
    vector <double> data;
};

#endif