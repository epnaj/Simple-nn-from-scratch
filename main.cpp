#include <iostream>
#include <ctime>
#include <cstdlib>
#include "nn/nn.hpp"
#include "mnist_reader/mnist_csv.hpp"

int main(){
    srand(time(NULL));
    int trainNum = 60000, testNum = 10000, x; // ilość obrazów przenaczonych do treningu (plik zawiera 60000), oraz do testów (plik zawiera 10000)
    vectors v;
    v.n_struct = {784,50,50,10}; // 784 = 28 x 28 , obrazy są zapisane w formacie 28 x 28 pixeli , 50 , 50  - warstwy tzw. ukryte, liczba neuronów jest dowolna, 10 - mamy 10 cyfr
    // inicjalizacja sieci o strukturze podanej w argumencie
    nnet nn(v.n_struct);
    // pobierzmy dane
    mnist_img pictures;
    pictures.load("mnistcsv/mnist_train.csv", 60000);
    // pętla treningu 
    for(int i=0; i<trainNum; i++){
        v.target = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        v.target[pictures.labels[i]] = 1;
        nn.train(pictures.images[i], v.target);

        if((i+1)%(trainNum/10) == 0)  // wskaźnik postępu, przydaje się przy dużej ilości danych  
            std::cout << (i+1)/(trainNum/100) << "%" << std::endl;
    }
    // test sprawności sieci
    std::cout << "Sprawdzanie dokladnosci sieci\n";
    pictures.load("mnistcsv/mnist_test.csv", testNum);
    double good=0; // ilość dobrze rozpoznanych obrazów
    for(int i=0; i<testNum; i++){
        nn.forward(pictures.images[i]);
        if(nn.max_of_softmax() == pictures.labels[i])
            good++;
    }
    std::cout << "Dokladnosc = " << good << " / " << testNum << " = " << double(good/testNum) << std::endl;
    // test sieci przez użykownikia
    for(int i=0; i<5; i++){
        std::cout << "wprowadz liczbe miedzy 0 a " << testNum-1 << std::endl;
        cin >> x;
        nn.forward(pictures.images[x]);
        pictures.read_digit(x);
        std::cout << "odp: " << nn.max_of_softmax() << std::endl;
        std::cout << "etykieta: " << pictures.labels[x] << std:: endl;
    }
    
    return 0;
}