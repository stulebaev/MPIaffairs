/* calcGini.cpp
  Calculating Gini coefficient for genome sequences.
*/
#include <iostream>
#include <numeric>
#include <cmath>
#include "mpi.h"
#include "bioio.hpp"

#define PARTNUMS_TAG 100
#define SEQUENCE_TAG 101
#define ARGUMENT_TAG 102

void calcHistogram(std::string sequence, std::vector<int>& histogram)
{
  const uint8_t powersOfTwo[] = { 1, 2, 4, 8, 16, 32 };
  std::fill(histogram.begin(), histogram.end(), 0);
  for (int i = 0; i < sequence.size()-5; i += 3) {
    int j = 0;
    for (int k = i; k <= i+5; k++)
      j += (int)(sequence[k]-'A')*powersOfTwo[k-i];
    histogram[j] += 1;
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int numprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (numprocs < 2) {
    std::cerr << "Not enough work processes" << std::endl;
    MPI_Finalize();
    return -1;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int sequences_num, part_nums;
  if (rank == 0) { // процесс N0 -- "диспетчер"
    if (argc < 2) {
      std::cerr << "The name of genetic sequence file is not specified" << std::endl;
      MPI_Finalize();
      return -1;
    }
    std::string sequence_file(argv[1]);
    // читаем файл с генетическими последовательностями
    auto records = bioio::read_fasta(sequence_file);
    sequences_num = records.size();
    if (sequences_num == 0) {
      std::cerr << "Failed to open sequence file " << sequence_file << " for reading" << std::endl;
      MPI_Finalize();
      return -1;
    }
    // вычисляем количество последовательностей, обрабатываемых каждым процессом
    part_nums = sequences_num / (numprocs-1);
    if (sequences_num % (numprocs-1)) {
      // количество последовательностей не делится нацело на число процессов
      part_nums++;
      int res = sequences_num % part_nums;
      // последний процесс обрабатывает остаток
      MPI_Send(&res, 1, MPI_INT, numprocs-1, PARTNUMS_TAG, MPI_COMM_WORLD);
    }
    else
      MPI_Send(&part_nums, 1, MPI_INT, numprocs-1, PARTNUMS_TAG, MPI_COMM_WORLD);
    // рассылаем количество обрабатываемых последовательностей
    for (int k = 1; k < numprocs-1; k++)
      MPI_Send(&part_nums, 1, MPI_INT, k, PARTNUMS_TAG, MPI_COMM_WORLD);
    int sendto = 0;
    // процесс N0 рассылает генетические последовательности
    for (int k = 0; k < sequences_num; k++) {
      if ((k % part_nums) == 0) sendto++;
      std::string& sequence = records[k].sequence;
      // посылаем последовательность процессу N"sendto"
      MPI_Send(sequence.c_str(), sequence.size(), MPI_CHAR, sendto, SEQUENCE_TAG, MPI_COMM_WORLD);
    }
  }

  if (rank != 0) {
    // рабочий процесс
    MPI_Status status;
    // получаем количество обрабатываемых последовательностей
    MPI_Recv(&part_nums, 1, MPI_INT, 0, PARTNUMS_TAG, MPI_COMM_WORLD, &status);
    std::vector<int> U(4096);
    int received = 0;
    while (received < part_nums) { // пока не обработали все последовательности
      int flag;
      // проверка -- есть ли посылка для нас
      MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
      if (flag) { // есть данные 
        int count; //длина сообщения
        MPI_Get_count(&status, MPI_CHAR, &count);
        char* buffer = new char[count]; //выделяем временный буфер
        // получаем генетическую последовательность
        MPI_Recv(buffer, count, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        std::string sequence(buffer, count);
        delete[] buffer;
        received++;
        calcHistogram(sequence, U);
        double sum = std::accumulate(U.begin(), U.end(), 0);
        std::vector<double> Q;
        Q.reserve(U.size());
        // производим нормировку массива U
        std::transform(U.begin(), U.end(), std::back_inserter(Q), [sum](int u){ return 1.0e6*u/sum; });
        // перемешиваем генетическую последовательность 100 раз
        for (int k = 0; k < 100; k++)
          std::random_shuffle(sequence.begin(), sequence.end());
        calcHistogram(sequence, U);
        sum = std::accumulate(U.begin(), U.end(), 0);
        std::vector<double> R;
        R.reserve(U.size());
        std::transform(U.begin(), U.end(), std::back_inserter(R), [sum](int u){ return 1.0e4*u/sum; });
        double sumQ = std::accumulate(Q.begin(), Q.end(), 0.0);
        double sumR = std::accumulate(R.begin(), R.end(), 0.0);
        double L = sumQ + sumR;
        double I = 0.0;
        int j;
        sum = 0.0;
        for (j = 0; j < 4096; j++)
          sum += Q[j];
        for (j = 0; j < 4096; j++)
          sum += R[j];
        I += sum;
        I -= sumQ; I -= sumR; //-x(i)
        sum = 0.0;
        for (j = 0; j < 4096; j++) {
          sum += (Q[j] + R[j]); //y(j)
        }
        I -= sum;
        I += L*log(L);
        double W = sqrt(4*I) - sqrt(2*4095*15-1);
        // отсылаем результат процессу "диспетчеру"
        MPI_Send(&W, 1, MPI_DOUBLE, 0, ARGUMENT_TAG, MPI_COMM_WORLD);
      }
    }
  }

  if (rank == 0) {
    std::ofstream results("results.bin", std::ios::out | std::ios::binary); 
    double W;
    MPI_Status status;
    // получаем результаты от всех рабочих процессов
    for (int k = 0; k < sequences_num; k++) {
      MPI_Recv(&W, 1, MPI_DOUBLE, MPI_ANY_SOURCE, ARGUMENT_TAG, MPI_COMM_WORLD, &status);
      if (results.is_open()) results << W;
    }
  }

  MPI_Finalize();
  return 0;
}
