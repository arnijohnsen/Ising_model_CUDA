#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Lattice dimensions and thread_num
#define thread_num 512 // Should be 2^k
#define grid_dim_x 1024
#define grid_size 1048576
#define iter 500
#define iterbal 100

void calc_cpu(float B, float kT, float Q, float *E_avg, float *M_avg, float *E_var, float *M_var);
__global__ void set_lattice(bool *lattice);
__global__ void iterate_grid(float B, float kT, float Q, bool round, bool *dev_lattice, float *d_E_vec, float *d_M_vec, int seed);
__global__ void reset_vec(float *vec);
__global__ void vec_sum(float *vec, float *result);
__global__ void set_val(float *variable, float value);
__global__ void add_val(float *variable, float *addition);
__device__ int posMod(int number, int modulus);
__device__ int indexMap(int xi, int yi);

// Inn í Accumulator eru lesin gildi í úrtaki, haldið er utan um meðalgildi og 
// dreifni þess úrtaks. 
class Accumulator
{
private:
    int N;
    float m;
    float s ;
    // Fastayrðing gagna:
    //  N er fjöldi talna í því úrtaki sem hefur verið lesið inn í eintak af Accumulator, N >= 0
    //  m er meðaltal talna í því úrtaki sem hefur verið lesið inn í eintak af Accumulator
    //  s er summa ferningsfrávika (frávik sérhvers gildis frá meðaltali, í öðru veldi), í því 
    //    úrtaki sem hefur verið lesið inn í eintak af Accumulator, s >= 0
public:
    // N: Accumulator a;
    // F: Ekkert
    // E: a er nýtt eintak af Accumulator, sem engar tölur hafa lesnar inn í. 
    //      Öll gögn í a hafa verið núllstillt, það er a.N = 0, a.m = 0.0 og a.s = 0.0
    Accumulator() {
        N = 0;
        m = 0.0;
        s = 0.0;
    }

    // N: a.addDataValue(x)
    // F: Ekkert
    // E: Búið er að bæta x í úrtakið a
    void addDataValue(float x)
    {
        N++;
        s = s + 1.0*(N-1)/N*(x-m)*(x-m);
        m = m + (x-m)/N;
    }

    // N: x = a.mean()
    // F: Ekkert
    // E: x inniheldur meðaltal talna í úrtakinu a
    float mean()
    {
        return m;
    }

    // N: x = a.var()
    // F: N > 1
    // E: x inniheldur dreifni talna í úrtakinu a
    float var()
    {
        return s/(N-1);
    }

    // N: x = a.stddev()
    // F: N > 1
    // E: x inniheldur staðalfrávik talna í úrtakinu a
    float stddev ( )
    {
        return sqrt(s/(N-1));
    }
};

int main(){
    // Minimum and maximum values of B, and number of steps. 
    // If Bsteps = 1, then only Bmin is used. 
    float B;
    float Bmin = 0.0;
    float Bmax = 1.0;
    int Bsteps = 1;
    // Minimum and maximum values of kT, and number of steps. 
    // If kTsteps = 1, then only kTmin is used. 
    float kT;
    float kTmin = 0.5;
    float kTmax = 5.0;
    int kTsteps = 1;
    // Minimum and maximum values of Q, and number of steps. 
    // If Qsteps = 1, then only Qmin is used. 
    float Q;
    float Qmin = -1.0;
    float Qmax = 1.0;
    int Qsteps = 1;

    srand(time(NULL)); // Seed GPU RNG
    float Emean;
    float Mmean;
    float Evar;
    float Mvar;

    
    char filename[20];
    sprintf(filename, "results.dat");
    FILE *fp;
    fp = fopen(filename, "w");
    for (int i=0;i<Bsteps;i++){ // B loop
        if (Bsteps>1){
            B = Bmin + i*(Bmax-Bmin)/(Bsteps-1);
        }
        else{
            B = Bmin;
        }
        for(int k=0; k<Qsteps; k++){ // Q loop
            if (Qsteps>1){
                Q = Qmin + k*(Qmax-Qmin)/(Qsteps-1);
            }
            else{
                Q = Qmin;
            }
            for(int j=0; j<kTsteps; j++){ // kTsteps
                if (kTsteps>1){
                    kT = kTmin + j*(kTmax-kTmin)/(kTsteps-1);
                }
                else{
                    kT = kTmin;
                }
                // printf("Performing calculation at B=%g, kT=%g, Q=%g\n", B, kT, Q);
                calc_cpu(B, kT, Q, &Emean, &Mmean, &Evar, &Mvar);
                fprintf(fp, "%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n", B, kT, Q, Emean, Mmean, Evar, Mvar);
            }
        }
    }
    fclose(fp);
}

// U: calc_cpu(...)
// B: kT > 0, n => 0 
// A: The results of an ising simulation at magnetic field B and
//      temperature kT have been stored in Earr[n] (mean energy) 
//      and Marr[n] (mean magnetization)
void calc_cpu(float B, float kT, float Q, float *E_avg_out, float *M_avg_out, float *E_var_out, float *M_var_out){
    // Degbug things
    // Template:
    // cudaMemcpy( &buggy, dev_value, sizeof(float), cudaMemcpyDeviceToHost);
    // printf("%g\n",buggy);
    /*float buggy;*/
    /*float buggyvec[thread_num];*/

    // Create, allocate memory for and set lattice
    bool *dev_lattice;
    cudaMalloc( (void**)&dev_lattice, grid_size*sizeof(bool) );
    set_lattice<<<1, thread_num>>>(dev_lattice);

    float *dev_dEvec;
    float *dev_dMvec;
    cudaMalloc( (void**)&dev_dEvec, thread_num*sizeof(float) );
    cudaMalloc( (void**)&dev_dMvec, thread_num*sizeof(float) );

    float *dev_Etot;
    float *dev_Mtot;
    /*float *dev_Eavg;*/
    /*float *dev_Mavg;*/
    cudaMalloc( (void**)&dev_Etot, sizeof(float) );
    cudaMalloc( (void**)&dev_Mtot, sizeof(float) );
    /*cudaMalloc( (void**)&dev_Eavg, sizeof(float) );*/
    /*cudaMalloc( (void**)&dev_Mavg, sizeof(float) );*/

    set_val<<<1,1>>>(dev_Etot, grid_size*(-2.0-2.0*Q-B));
    set_val<<<1,1>>>(dev_Mtot, grid_size);
    /*set_val<<<1,1>>>(dev_Eavg, 0.0);*/
    /*set_val<<<1,1>>>(dev_Mavg, 0.0);*/

    Accumulator energy;
    Accumulator magnet;

    float Etot;
    float Mtot;

    for (int j=0; j<iter; j++){
        reset_vec<<<1, thread_num>>>(dev_dEvec);
        reset_vec<<<1, thread_num>>>(dev_dMvec);

        iterate_grid<<<1, thread_num>>>(B, kT, Q, 0, dev_lattice, dev_dEvec, dev_dMvec, rand() );
        iterate_grid<<<1, thread_num>>>(B, kT, Q, 1, dev_lattice, dev_dEvec, dev_dMvec, rand() );

        vec_sum<<<1, thread_num>>>(dev_dEvec, dev_Etot);
        vec_sum<<<1, thread_num>>>(dev_dMvec, dev_Mtot);

        if (j>iterbal){
            cudaMemcpy( &Etot, dev_Etot, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy( &Mtot, dev_Mtot, sizeof(float), cudaMemcpyDeviceToHost);
            Etot = Etot/grid_size;
            Mtot = Mtot/grid_size;
            energy.addDataValue(Etot);
            magnet.addDataValue(Mtot);
        }
    }
    *E_avg_out = energy.mean();
    *M_avg_out = magnet.mean();
    *E_var_out = energy.var();
    *M_var_out = magnet.var();
    cudaFree(dev_lattice);
    cudaFree(dev_dEvec);
    cudaFree(dev_dMvec);
    cudaFree(dev_Etot);
    cudaFree(dev_Mtot);

}

// U: set_lattice<<<1, thread_num>>>(dev_lattice);
// B: dev_lattice points to allocated device memory for grid_size bool numbers
// A: all elements of dev_lattice are set to 1
__global__ void set_lattice(bool *lattice){
    int tid = threadIdx.x;
    for (int i=tid;i<grid_size;i+=thread_num){
        lattice[i] = 1;
    }
}

// U: iterate_grid<<<1, thread_num>>>(...)
// B: 
// A: One ising iteration has been performed over a checkerboard. If round=0 it's over the white squares, if round=1 it's over 
//     the black squares. The change done by each thread has been added to d_E_vec[tid] and d_M_vec[tid]
__global__ void iterate_grid(float B, float kT, float Q, bool round, bool *dev_lattice, float *d_E_vec, float *d_M_vec, int seed){
    int tid=threadIdx.x;

    curandState_t state;
    curand_init(seed+tid, 0, 0, &state);

    int si;
    float ssum;
    float delta_E;
    float delta_M;
    float p;
    float r;
    int xi;
    int yi;
    for (int i=round+2*tid;i<grid_size;i+=2*thread_num){
        yi = i/grid_dim_x;
        if ((yi%2)==0){
            xi = i%grid_dim_x;
        }
        else{
            xi = grid_dim_x-i%grid_dim_x-1;
        }
        si = 2*dev_lattice[i]-1;
        ssum = 2*dev_lattice[indexMap(xi-1,yi)]
              +2*dev_lattice[indexMap(xi+1,yi)]
              -2
            +Q*2*dev_lattice[indexMap(xi,yi-1)]
            +Q*2*dev_lattice[indexMap(xi,yi+1)]
            -Q*2;
        delta_E = 2*si*(ssum+B);
        delta_M = -2*si;
        if (delta_E < 0){
            p = 1;
        }
        else{
            p = exp(-delta_E/kT);
        }
        r = curand_uniform(&state);
        if (r<p){ // Spin flip!
            d_E_vec[tid] += delta_E;
            d_M_vec[tid] += delta_M;
            dev_lattice[i] = !( dev_lattice[i] );
        }
    }
}

// U: reset_vec<<<1, thread_num>>>(dev_vec)
// B: dev_vec has been allocated device memory for thread_num float numbers
// A: All elements of dev_vec have been set as 0.0
__global__ void reset_vec(float *vec){
    vec[threadIdx.x] = 0.0;
}

// U: vec_sum<<<1, thread_num>>>(dev_vec, dev_result)
// B: dev_vec has length thread_num
// A: The sum of elements in dev_vec has been added to result
__global__ void vec_sum(float *vec, float *result){
    // Right multithread version (has to use threads)
    int tid = threadIdx.x;
    int offset = thread_num>>1;
    while (offset>0){
        if (tid < offset){
            vec[tid] += vec[tid+offset];
        }
        __syncthreads();
        offset=offset>>1;
    }
    if (tid==0){
        *result += vec[0];
    }
    
    // Right single thread version
    /*int tid = threadIdx.x;*/
    /*if (tid == 0){*/
        /*for (int i=1;i<thread_num;i++){*/
            /*vec[0] += vec[i];*/
        /*}*/
        /**result += vec[0];*/
    /*}*/
}

// U: set_val<<<1, 1>>>(variable, value)
// B: 
// A: *variable = value
__global__ void set_val(float *variable, float value){
    *variable = value;
}

// U: add_val<<<1, 1>>>(variable, addition)
// B:
// A: *variabe += *addition
__global__ void add_val(float *variable, float *addition){
    *variable += *addition;
}

// U: z = posMod(n,m)
// B: m > 0
// A: z = n%m if n>=0, z = n%m + m if n < 0
__device__ int posMod(int number, int modulus){
    int result = number%modulus;
    if (result<0){
        result +=modulus;
    }
    return result;
}

__device__ int indexMap(int xi, int yi){
    xi = posMod(xi,grid_dim_x);
    yi = posMod(yi,grid_dim_x);
    int i = yi*grid_dim_x;
    if (yi%2==0){
        i += xi;
    }
    else{
        i += grid_dim_x-xi-1;
    }
    return i;
}

