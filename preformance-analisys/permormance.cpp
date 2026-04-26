#include <iostream>
#include <fstream>

int main()
{
    std::ifstream b64("64.txt");
    std::ifstream b128("128.txt");
    std::ifstream b256("256.txt");
    std::ifstream b512("512.txt");
    std::ifstream b1024("1024.txt");
    std::ifstream b2048("2048.txt");
    std::ifstream b4096("4096.txt");
    std::ifstream b8192("8192.txt");
    std::ifstream b16384("16384.txt");
    std::ifstream b32768("32768.txt");
    std::ifstream b65536("65536.txt");

    std::ofstream avg("avg.txt");

    float temp;
   
    //64
    float szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b64 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 64 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 64 bodies: " << szum/2500  << std::endl;

    //128
    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b128 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 128 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 128 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b256 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 256 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 256 bodies: " << szum/2500  << std::endl;

    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b512 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 512 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 512 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b1024 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 1024 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 1024 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b2048 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 2048 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 2048 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b4096 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 4096 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 4096 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b8192 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 8192 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 8192 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b16384 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 16384 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 16384 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b32768 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 32768 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 32768 bodies: " << szum/2500  << std::endl;


    szum = 0;
    for(int i = 0; i < 2500; i++)
    {
        b65536 >> temp;
        szum = szum + temp;
    }
    std::cout << "AVG fps for 65536 bodies: " << szum/2500 << std::endl;
    avg << "AVG fps for 65536 bodies: " << szum/2500  << std::endl;

    avg.close();
}