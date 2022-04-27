#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "mnist_csv.hpp"

using namespace std;

void mnist_img::load(string name,int number){
    double label;
    vector <double> digit;
    string alpha,n;
    int size,x;
    fstream file;
    file.open(name.c_str(),ios::in);
    for(int i=0;i<number;i++){
        getline(file,alpha);
        size = alpha.size();
        x=2;
        digit.clear();
        label = alpha[0]-'0';
        while(x<size){
            if(alpha[x]!=','){
                n = n + alpha[x];
            }
            else{
                digit.push_back(atof(n.c_str())/255.0);
                //digit.push_back(atof(n.c_str()));
                n.clear();
            }
            x++;
        }
        digit.push_back(0);
        labels.push_back(label);
        images.push_back(digit);
    }
}

void mnist_img::read_digit(int i){
        //cout << "Label : " << labels[i] << endl;
        for(int k=0;k<images[i].size();k+=28){
            for(int p=0;p<28;p++){
                //cout << m.images[i][k+p] << " ";
                if(images[i][k+p]>0)
                    cout << "*";
                else
                    cout << " ";
            }
            cout << endl;
        }
}