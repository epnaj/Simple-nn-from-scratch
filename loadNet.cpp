#include <iostream>
#include "nn/nn.hpp"
#include "mnist_reader/mnist_csv.hpp"

int main(){
    int testNum = 10000, x;
    vectors v;
    v.n_struct = {0};
    nnet nn(v.n_struct);
    mnist_img pictures;
    pictures.load("mnistcsv/mnist_test.csv", testNum);

    nn = loadNet("trainedModel/net_2_hidd_lay_200_200_0_9573_acc.txt");

    for(int i=0; i<5; i++){
        std::cout << "wprowadz liczbe miedzy 0 a " << testNum-1 << std::endl;
        std::cin >> x;
        nn.forward(pictures.images[x]);
        pictures.read_digit(x);
        std::cout << "odp: " << nn.max_of_softmax() << std::endl;
        std::cout << "etykieta: " << pictures.labels[x] << std:: endl;
    }
    
    return 0;
}