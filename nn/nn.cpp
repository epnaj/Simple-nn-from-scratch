#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <fstream>
#include <cmath>
#include "nn.hpp"

using namespace std;

double sigmoid(double x){
    return (1/(1+exp(-x)));
}

double ReLU(double x){
    return max((double)0,x);
}

double ReLUDerivative(double x){
    return x>0;
}

double sigmoidDerivativeUsingSourceVal(double x){
    return sigmoid(x)*(1-sigmoid(x));
}

double sigmoidDerivative(double x){
    return x*(1-x);
}

neuron::neuron(int nonb){
    bias = (((double) rand() / (RAND_MAX))*2-1);
    for(int i=0;i<nonb;i++)
        weights.push_back(((double) rand() / (RAND_MAX))*2-1);
}

neuron::~neuron(){
    weights.~vector();
}

layer::layer(int nOfNeruons,int nonb){
    for(int i=0;i<nOfNeruons;i++){
        lay.push_back(new neuron(nonb));
    }
}

layer::~layer(){
    lay.~vector();
}

nnet::nnet(vector <int> &n_struct){
    layer *l;
    l = new layer(n_struct[0], 0);
    net.push_back(l);
    for(int i=1;i<n_struct.size();i++){
        net.push_back(new layer(n_struct[i], n_struct[i-1]));
    }
    l = NULL;
    delete l;
}

nnet::~nnet(){
    net.~vector();
}

void nnet::saveNet(string name){
    fstream f;
    f.open(name.c_str(),ios::out|ios::trunc);
    for(int i=0;i<net.size();i++){
        f << net[i]->lay.size() << " ";
    }
    f << endl;
    for(int i=0;i<net.size();i++){
        for(int j=0;j<net[i]->lay.size();j++){
            for(int k=0;k<net[i]->lay[j]->weights.size();k++){
                f << net[i]->lay[j]->weights[k] << " ";
            }
        }
        f<< endl;
        for(int j=0;j<net[i]->lay.size();j++){
            f << net[i]->lay[j]->bias << " ";
        }        
        f << endl;
    }
    f.close();
}

void nnet::printNet(){
    for(int i=0;i<net.size();i++){
        for(int j=0;j<net[i]->lay.size();j++){
            for(int k=0;k<net[i]->lay[j]->weights.size();k++){
                cout << net[i]->lay[j]->weights[k] << " ";
            }
        }
        cout << endl;
        for(int j=0;j<net[i]->lay.size();j++){
            cout << net[i]->lay[j]->val << " ";
        }        
        cout << endl;
    }
}

nnet loadNet(string name){
    vector <int> toConv;
    string a,num;
    fstream f;
    f.open(name.c_str(),ios::in);
        getline(f,a);
    for(int i=0;i<a.size();i++){
        if(a[i]!=' '){
            num+=a[i];
        }
        else{
            toConv.push_back(atoi(num.c_str()));
            num.clear();
        }
    }
    nnet n(toConv);
    for(int i=0;i<n.net.size();i++){
        for(int j=0;j<n.net[i]->lay.size();j++){
            for(int k=0;k<n.net[i]->lay[j]->weights.size();k++){
                f >> n.net[i]->lay[j]->weights[k];
            }
        }
        for(int j=0;j<n.net[i]->lay.size();j++){
            f >> n.net[i]->lay[j]->bias;
        }        
    }
    f.close();
    return n;
}

void nnet::forward(vector <double> &data){
    double s=0;
    for(int j=0;j<net[0]->lay.size();j++){
        net[0]->lay[j]->val = data[j]; 
    }
    for(int i=1;i<net.size();i++){
        for(int j=0;j<net[i]->lay.size();j++){
            s=0;
            for(int k=0;k<net[i]->lay[j]->weights.size();k++){
                s += net[i]->lay[j]->weights[k] * net[i-1]->lay[k]->val;
            }
            net[i]->lay[j]->val = sigmoid( s + net[i]->lay[j]->bias);
        }
    }
}

double nnet::cost_last_lay(vector <double> &target){
    double sum=0;
    for(int j=0;j<net[net.size()-1]->lay.size();j++){
        sum+=(net[net.size()-1]->lay[j]->val - target[j])*(net[net.size()-1]->lay[j]->val - target[j]);
    }
    return sum/net[net.size()-1]->lay.size();
} 

vector <double> nnet::softmax(){
    double maks = 0;
    for(int j=0;j<net[net.size()-1]->lay.size();j++){
        if(net[net.size()-1]->lay[j]->val>maks)
            maks=net[net.size()-1]->lay[j]->val;
    }
    vector <double> exp_values; //  = e ^ x
    for(int j=0;j<net[net.size()-1]->lay.size();j++){
        exp_values.push_back(exp(net[net.size()-1]->lay[j]->val-maks));
    }
    double norm_base = 0 , sum =0 ;
    vector <double> norm_values;
    for(int i=0;i<exp_values.size();i++){
        norm_base += exp_values[i];
    }
    for(int i=0;i<exp_values.size();i++){
        norm_values.push_back(exp_values[i]/norm_base);
    }
    return norm_values;
}

int nnet::max_of_last_lay_inx(){
    double maks = 0;
    int inx;
    for(int j=0;j<net[net.size()-1]->lay.size();j++){
        if(net[net.size()-1]->lay[j]->val>maks){
            maks=net[net.size()-1]->lay[j]->val;
            inx = j;
        }
    }
    return inx;
}

int nnet::max_of_softmax(){
    vector <double> s = this->softmax();
    double maks = 0;
    int inx;
    for(int j=0;j<s.size();j++){
        if(s[j]>maks){
            maks=s[j];
            inx = j;
        }
    }
    return inx;
}

vector <double> nnet::error_last(vector <double> &target){
    vector <double> err;
    double cost = this->cost_last_lay(target);
    for(int j=0;j<net[net.size()-1]->lay.size();j++){
       err.push_back(2*(target[j] - net[net.size()-1]->lay[j]->val));
    }
    return err;
}

void nnet::back_d(vector <double> &error){
    for(int j=0;j<net[net.size()-1]->lay.size();j++){
            net[net.size()-1]->lay[j]->d = error[j];
    }

    for(int i=net.size()-2;i>0;i--){ 
        for(int j=0;j<net[i]->lay.size();j++){
            net[i]->lay[j]->d = 0;
            for(int k=0;k<net[i+1]->lay.size();k++){ 
                net[i]->lay[j]->d += net[i+1]->lay[k]->weights[j] * net[i+1]->lay[k]->d;
            }
        }
    }
}

void nnet::for_d(){
    for(int i=1;i<net.size();i++){
        for(int j=0;j<net[i]->lay.size();j++){
            for(int k=0;k<net[i]->lay[j]->weights.size();k++){
                net[i]->lay[j]->weights[k] = net[i]->lay[j]->weights[k] + learningRate * net[i]->lay[j]->d * net[i-1]->lay[k]->val * sigmoidDerivative(net[i]->lay[j]->val);
            }
            net[i]->lay[j]->bias = net[i]->lay[j]->bias + learningRate * net[i]->lay[j]->d * sigmoidDerivative(net[i]->lay[j]->val);
        }
    }
}

void nnet::out_last_lay(){
    for(int j=0;j<net[net.size()-1]->lay.size();j++){
        cout << "j : " << j << " " << net[net.size()-1]->lay[j]->val << " ";
    }
    cout << endl;
}

void nnet::train(vector<double> &data, vector<double> &target){
    this->forward(data);
    vector<double> error = this->error_last(target);
    this->back_d(error);
    this->for_d();
}