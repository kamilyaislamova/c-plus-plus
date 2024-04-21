#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include <string>
using namespace std;

const long long llmax = 9223372036854775807;
struct point {
    float x, y, z;
};

point start[3];

float distance(point a, point b) {
    return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

long long random_seed(long long seed1) {
    string t;
    while (seed1 > 0) {
        t += (seed1 % 2) + '0';
        seed1 /= 2;
    }
    while (t.size() < 64) {
        t += '0';
    }
    long long answ = 0;
    long long num = 211;
    long long mod = 1e9 + 7;
    for (int i = t.size() - 1; i >= 0; i--) {
        answ += (num * (t[i] - '0' + 1)) % mod;
        answ %= mod;
        num *= 211;
        num %= mod;
    }
    return answ;
}

long long state = 1;

float xorshift64(long long state1, float cube_edge)
{
    long long x = state1;
    x = x ^ (x << 13);
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    float result = (float)x / llmax * cube_edge;
    return result;
}

long long xorshift64_parallels(long long state1)
{
    long long x = state1;
    x = x ^ (x << 13);
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        cerr << "Incorrect number of arguments";
        exit(1);
    }
    int threads = stoi((argv[1]));
    if (threads < -1) {
        cerr << "Incorrect number of threads";
        exit(1);
    }
    FILE* file = fopen(argv[2], "r");
    FILE* file_end = fopen(argv[3], "w");
    if (file == nullptr) {
        cerr << "This file doesn't exist or can't be open";
        exit(1);
    }
    long long n;
    int checker1 = fscanf(file, "%lld\n", &n);
    if (n < 0 || checker1 == 0) {
        cerr << "Incorrect input";
        exit(1);
    }
    for (int i = 0; i < 3; i++) {
        int checker2 = fscanf(file, "(%f%f%f)\n", &start[i].x, &start[i].y, &start[i].z);
        if (checker2 < 3) {
            cerr << "Incorrect input";
            exit(1);
        }
    }

    double begin = omp_get_wtime();

    float edge = distance(start[0], start[2]);
    for (int i = 0; i < 2; i++) {
        edge = min(edge, distance(start[i], start[i + 1]));
    }
    float radius = edge * sqrtf(2) / 2;
    float edge_cube = radius * 2;
    long long cube = 0, oct = 0;
    float result = edge * edge * edge * sqrtf(2) / 3;
    float V_cube = edge_cube * edge_cube * edge_cube;
    int final_threads = 0;

    if (threads == -1) {
        state = random_seed(1);
        long long local_state = random_seed(1);
        for (long long i = 0; i < n; i++) {
            point a;
            a.x = xorshift64(state, radius);
            a.y = xorshift64(state, radius);
            a.z = xorshift64(state, radius);
            float sum = abs(a.x) + abs(a.y) + abs(a.z);
            if (sum <= radius)
                oct++;
            cube++;
        }
        final_threads = 0;
    }
    if (threads == 0) {
        #pragma omp parallel 
        {
            long long thread = omp_get_thread_num();
            long long local_state = random_seed(thread);
            long long local_oct = 0;
            point a = {0, 0, 0};
            #pragma omp for schedule (static)
                for (long long i = 0; i < n; i++) {
                    local_state = xorshift64_parallels(local_state);
                    a.x = (float)local_state / llmax * radius;
                    local_state = xorshift64_parallels(local_state);
                    a.y = (float)local_state / llmax * radius;
                    local_state = xorshift64_parallels(local_state);
                    a.z = (float)local_state / llmax * radius;
                    float sum = abs(a.x) + abs(a.y) + abs(a.z);
                    if (sum <= radius)
                        local_oct++;
                }
            #pragma omp atomic 
                oct += local_oct;
        }
        final_threads = omp_get_num_threads();
    }
    if (threads > 0) {
        #pragma omp parallel num_threads(threads)
        {
            long long thread = (long long)omp_get_thread_num();
            long long local_state = random_seed(thread);
            long long local_oct = 0;
            point a = { 0, 0, 0 };
            #pragma omp for schedule (static)
            for (long long i = 0; i < n; i++) {
                local_state = xorshift64_parallels(local_state);
                a.x = (float)local_state / llmax * radius;
                local_state = xorshift64_parallels(local_state);
                a.y = (float)local_state / llmax * radius;
                local_state = xorshift64_parallels(local_state);
                a.z = (float)local_state / llmax * radius;
                float sum = abs(a.x) + abs(a.y) + abs(a.z);
                if (sum <= radius)
                    local_oct++;
            }
            #pragma omp atomic 
                oct += local_oct;
        }
        final_threads = omp_get_num_threads();
    }
    float result_random = V_cube * (float)oct / n;
    double finish = omp_get_wtime();
    printf("Time (%i thread(s)): %g ms\n", final_threads, finish - begin);
    fprintf(file_end, "%g %g\n", result, result_random);
}


//5 76 34567 7567 34 -543 
//7 3333333 4345 6 -45678 3234
//5 