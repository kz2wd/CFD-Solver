#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>

#define UNUSED(X) (void)(X)

#define at2d(M, i, j, c, X) ((M)[(i) * ((X) * 2) + (j) * (2) + (c)])
#define at1d(M, i, j, X) ((M)[(i) * (X) + (j)])


typedef struct {

    const double dx;
    const double dt;
    const double nu;
    const double K;
    double* u;
    double* p;

    // begin compute var
    double* ubuf1;
    double* ubuf2;
    // end compute var
    //
    const size_t X;
    const long seed;

} simulation;


simulation init_simulation(const double re, const double dt, const size_t X, const size_t K, const long seed) {
    simulation simu = {
        .dx= 1.0 / X,
        .dt=dt,
        .nu= 1.0 / re,
        .K = K,
        .u = (double*) calloc(X * X * 2, sizeof(double)),
        .p = (double*) calloc(X * X, sizeof(double)),
        .ubuf1 = (double*) calloc(X * X * 2, sizeof(double)),
        .ubuf2 = (double*) calloc(X * X * 2, sizeof(double)),
        .X = X,
        .seed = seed,
    };
    return simu;
}


typedef struct {
    atomic_size_t write_idx;
    double data[];
} shm_t;

static shm_t *shm;
void connect_mmap(const size_t X) {
    int fd = shm_open("/sim_shm", O_RDWR, 0600);
    const size_t N = X*X*2;
    shm = mmap(NULL, sizeof(atomic_size_t) + sizeof(double) * N,
                      PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
}

void write_u(const double * const u, const size_t X, const unsigned int step) {
    const size_t N = X*X*2;
    memcpy(shm->data, u, N * sizeof(double));
    atomic_store_explicit(&shm->write_idx, step, memory_order_release);
}


double rand_range(double min_n, double max_n)
{
    return (double)rand()/RAND_MAX * (max_n - min_n) + min_n;
}


void init_u(double* const u, const size_t X, const double scale) {
    const double up = 0.5 * scale;
    const double low = -0.5 * scale;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(u, i, j, 0, X) = rand_range(low, up);
            at2d(u, i, j, 1, X) = rand_range(low, up);
        }
    }
}

void init_p(double* const p, const size_t X, const double scale) {
    const double up = 0.5 * scale;
    const double low = -0.5 * scale;
    for (size_t j = 1; j < X - 1; ++j) {
        for (size_t i = 1; i < X - 1; ++i) {
            at1d(p, i, j, X) = rand_range(low, up);
        }
    }
}


// Ldc bc
void bcu(double* const u, const double K, const size_t X) {
    for (size_t i = 1; i < X - 1; ++i) {
        // lid top
        at2d(u, i, 0, 0, X) = 2 * K - at2d(u, i, 1, 0, X);
        at2d(u, i, 0, 1, X) = - at2d(u, i, 1, 1, X);

        // no slip everywhere else
        at2d(u, i, X - 1, 0, X)   = - at2d(u, i, X - 2, 0, X);
        at2d(u, i, X - 1, 1, X)   = - at2d(u, i, X - 2, 1, X);

        at2d(u, 0, i, 0, X)       = - at2d(u, 1, i, 0, X);
        at2d(u, 0, i, 1, X)       = - at2d(u, 1, i, 1, X);

        at2d(u, X - 1, i, 0, X)   = - at2d(u, X - 2, i, 0, X);
        at2d(u, X - 1, i, 1, X)   = - at2d(u, X - 2, i, 1, X);
    }

    // Corners are never used

}

// Ldc bc
void bcp(double* p, const size_t X) {
    for (size_t i = 1; i < X - 1; ++i) {
        at1d(p, i, 0, X) = at1d(p, i, 1, X);
        at1d(p, i, X - 1, X) = at1d(p, i, X - 2, X);
        at1d(p, 0, i, X) = at1d(p, 1, i, X);
        at1d(p, X - 1, i, X) = at1d(p, X - 2, i, X);
    }
}

// (u⋅∇)uc = ux ∂x uc + uy ∂y uc | (d = ∂)
void convection(double* const u, double* const conv, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    const double two_dx = 2 * dx;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            const double dxu0 = (at2d(u, i + 1, j, 0, X) - at2d(u, i - 1, j, 0, X)) / (two_dx);
            const double dyu0 = (at2d(u, i, j + 1, 0, X) - at2d(u, i, j - 1, 0, X)) / (two_dx);
            at2d(conv, i, j, 0, X) = at2d(u, i, j, 0, X) * dxu0 + at2d(u, i, j, 1, X) * dyu0;
            const double dxu1 = (at2d(u, i + 1, j, 1, X) - at2d(u, i - 1, j, 1, X)) / (two_dx);
            const double dyu1 = (at2d(u, i, j + 1, 1, X) - at2d(u, i, j - 1, 1, X)) / (two_dx);
            at2d(conv, i, j, 1, X) = at2d(u, i, j, 0, X) * dxu1 + at2d(u, i, j, 1, X) * dyu1;
        }
    }
}

void viscous_drag(double* const u, double* const visc, const double nu, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    const double d2 = dx * dx;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(visc, i, j, 0, X) = ((at2d(u, i + 1, j, 0, X) + at2d(u, i - 1, j, 0, X) + at2d(u, i, j + 1, 0, X) + at2d(u, i, j - 1, 0, X) - 4 * at2d(u, i, j, 0, X)) / (d2)) * nu;
            at2d(visc, i, j, 1, X) = ((at2d(u, i + 1, j, 1, X) + at2d(u, i - 1, j, 1, X) + at2d(u, i, j + 1, 1, X) + at2d(u, i, j - 1, 1, X) - 4 * at2d(u, i, j, 1, X)) / (d2)) * nu;
        }
    }
}

void pressure(double* const u, double* const p, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    bcp(p, X);

    const double two_dx = 2 * dx;
    const double d2 = dx * dx;
    const double omega = 2.0 / (1.0 + sin(M_PI / (double)X));
    const double one_m_omega = 1.0 - omega;

    const unsigned int iters = X;
    for (unsigned int i = 0; i < iters; ++i) {
        for (size_t i = 1; i < X - 1; ++i) {
            for (size_t j = 1; j < X - 1; ++j) {
                const double rhs = ((at2d(u, i + 1, j, 0, X) - at2d(u, i - 1, j, 0, X) + at2d(u, i, j + 1, 1, X) - at2d(u, i, j - 1, 1, X)) / two_dx ) * d2;
                at1d(p, i, j, X) = one_m_omega * at1d(p, i, j, X)
                    + omega * (at1d(p, i + 1, j, X) + at1d(p, i - 1, j, X) + at1d(p, i, j + 1, X) + at1d(p, i, j - 1, X) - rhs) / 4.0;
            }
        }
    }
}

void project_velocity(double* const u, double* const u_out, double* const p, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    bcp(p, X);
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(u_out, i, j, 0, X) = at2d(u, i, j, 0, X) - (at1d(p, i + 1, j, X) - at1d(p, i - 1, j, X)) / (2 * dx);
            at2d(u_out, i, j, 1, X) = at2d(u, i, j, 0, X) - (at1d(p, i, j + 1, X) - at1d(p, i, j - 1, X)) / (2 * dx);
        }
    }
}

void assemble_u(double* const u, double* const u_out, const double* const conv, const double* const visc, const double dt, const size_t X) {
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(u_out, i, j, 0, X) =  at2d(u, i, j, 0, X) + dt * (-at2d(conv, i, j, 0, X) + at2d(visc, i, j, 0, X));
        }
    }
}

void debug(const double* const u, const size_t X){

    double total = 0.0;
    double count = 0.0;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            total += at2d(u, i, j, 0, X);
            total += at2d(u, i, j, 1, X);
            count += 2.0;
        }
    }
    double mean = total / count;
    printf("mean: %f\n", mean);
}

void step_EE(simulation* sim){

    convection(sim->u, sim->ubuf1, sim->X, sim->dx, sim->K);
    viscous_drag(sim->u, sim->ubuf2, sim->nu, sim->X, sim->dx, sim->K);

    assemble_u(sim->u, sim->u, sim->ubuf1, sim->ubuf2, sim->dt, sim->X);

    pressure(sim->u, sim->p, sim->X, sim->dx, sim->K);

    project_velocity(sim->u, sim->u, sim->p, sim->X, sim->dx, sim->K);

}

void loop(simulation* sim, const size_t steps, const unsigned int write_interval) {
    connect_mmap(sim->X);
    for (unsigned int i = 0; i < steps; ++i) {
        printf("i: %d\n", i);
        debug(sim->u, sim->X);
        step_EE(sim);
        if (i % write_interval == 0) {
            write_u(sim->u, sim->X, i);
        }
    }
}


int main(int argc, char** argv) {

    UNUSED(argc);
    UNUSED(argv);

    printf("Starting simulation\n");

    simulation sim = init_simulation(100.0, 0.00001, 64, 1.0, 0);
    init_u(sim.u, sim.X, 0.0001);
    init_p(sim.p, sim.X, 0.0001);

    loop(&sim, 10000, 50);

    return EXIT_SUCCESS;
}
