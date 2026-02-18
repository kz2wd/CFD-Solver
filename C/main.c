#include <stddef.h>
#include <stdint.h>
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
#include <ini.h>

#define UNUSED(X) (void)(X)

#define at2d(M, i, j, c, X) ((M)[(i) * ((X) * 2) + (j) * (2) + (c)])
#define at1d(M, i, j, X) ((M)[(i) * (X) + (j)])

typedef const double* const f64ro;
typedef double* const f64rw;
typedef const size_t dim;
typedef const double cd;

typedef struct {

    const double dx;
    const double dt;
    const double nu;
    const double K;
    f64rw u;
    f64rw p;

    // begin compute var
    f64rw ubuf1;
    f64rw ubuf2;
    f64rw ubuf3;
    f64rw ubuf4;
    f64rw ubuf5;
    f64rw ubuf6;
    f64rw ubuf7;
    f64rw pbuf1;
    f64rw pbuf2;
    f64rw xbuf1;
    f64rw xbuf2;

    f64rw udebug1;
    // end compute var
    //
    const size_t X;
    const long seed;

} simulation;


simulation init_simulation(const double re, const double dt, const size_t X, const double K, const long seed) {
    simulation simu = {
        .dx= 1.0 / X,
        .dt=dt,
        .nu= (1.0 * K) / re,
        .K = K,
        .u = (f64rw) calloc(X * X * 2, sizeof(double)),
        .p = (f64rw) calloc(X * X, sizeof(double)),
        .ubuf1 = (f64rw) calloc(X * X * 2, sizeof(double)),
        .ubuf2 = (f64rw) calloc(X * X * 2, sizeof(double)),
        .ubuf3 = (f64rw) calloc(X * X * 2, sizeof(double)),
        .ubuf4 = (f64rw) calloc(X * X * 2, sizeof(double)),
        .ubuf5 = (f64rw) calloc(X * X * 2, sizeof(double)),
        .ubuf6 = (f64rw) calloc(X * X * 2, sizeof(double)),
        .ubuf7 = (f64rw) calloc(X * X * 2, sizeof(double)),
        .pbuf1 = (f64rw) calloc(X * X, sizeof(double)),
        .pbuf2 = (f64rw) calloc(X * X, sizeof(double)),
        .xbuf1 = (f64rw) calloc(X, sizeof(double)),
        .xbuf2 = (f64rw) calloc(X, sizeof(double)),
        .udebug1 = (f64rw) calloc(X * X * 2, sizeof(double)),
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

void write_u(f64ro u, const size_t X, const unsigned int step) {
    const size_t N = X*X*2;
    memcpy(shm->data, u, N * sizeof(double));
    atomic_store_explicit(&shm->write_idx, step, memory_order_release);
}


double rand_range(double min_n, double max_n)
{
    return (double)rand()/RAND_MAX * (max_n - min_n) + min_n;
}


void init_u(f64rw u, const size_t X, const double scale) {
    const double up = 0.5 * scale;
    const double low = -0.5 * scale;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(u, i, j, 0, X) = rand_range(low, up);
            at2d(u, i, j, 1, X) = rand_range(low, up);
        }
    }
}

void init_p(f64rw p, const size_t X, const double scale) {
    const double up = 0.5 * scale;
    const double low = -0.5 * scale;
    for (size_t j = 1; j < X - 1; ++j) {
        for (size_t i = 1; i < X - 1; ++i) {
            at1d(p, i, j, X) = rand_range(low, up);
        }
    }
}


// Ldc bc
void bcu(f64rw u, const double K, const size_t X) {
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
void bcp(f64rw p, const size_t X) {
    for (size_t i = 1; i < X - 1; ++i) {
        at1d(p, i, 0, X) = at1d(p, i, 1, X);
        at1d(p, i, X - 1, X) = at1d(p, i, X - 2, X);
        at1d(p, 0, i, X) = at1d(p, 1, i, X);
        at1d(p, X - 1, i, X) = at1d(p, X - 2, i, X);
    }
}

// (u⋅∇)uc = ux ∂x uc + uy ∂y uc | (d = ∂)
void convection(f64rw u, f64rw conv, const size_t X, const double dx, const double K) {
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

void viscous_drag(f64rw u, f64rw visc, const double nu, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    const double d2 = dx * dx;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(visc, i, j, 0, X) = ((at2d(u, i + 1, j, 0, X) + at2d(u, i - 1, j, 0, X) + at2d(u, i, j + 1, 0, X) + at2d(u, i, j - 1, 0, X) - 4 * at2d(u, i, j, 0, X)) / (d2)) * nu;
            at2d(visc, i, j, 1, X) = ((at2d(u, i + 1, j, 1, X) + at2d(u, i - 1, j, 1, X) + at2d(u, i, j + 1, 1, X) + at2d(u, i, j - 1, 1, X) - 4 * at2d(u, i, j, 1, X)) / (d2)) * nu;
        }
    }
}

void compute_pressure_rhs(f64ro u, f64rw rhs, const size_t X, const double dx) {
    const double two_dx = 2 * dx;
    const double d2 = dx * dx;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at1d(rhs, i, j, X) = ((at2d(u, i + 1, j, 0, X) - at2d(u, i - 1, j, 0, X) + at2d(u, i, j + 1, 1, X) - at2d(u, i, j - 1, 1, X)) / two_dx ) * d2;
        }
    }
}

// Not very jacobi... I dont use it anymore so I wont fix it for now
void pressure_jacobi(f64rw u, f64ro rhs, f64rw p, f64rw pbuf, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    bcp(p, X);

    const double omega = 2.0 / (1.0 + sin(M_PI / (double)X));
    const double one_m_omega = 1.0 - omega;

    const unsigned int iters = 25;

    for (unsigned int iter = 0; iter < iters; ++iter) {
        for (size_t i = 1; i < X - 1; ++i) {
            for (size_t j = 1; j < X - 1; ++j) {
                at1d(p, i, j, X) = one_m_omega * at1d(p, i, j, X)
                    + omega * (at1d(p, i + 1, j, X) + at1d(p, i - 1, j, X) + at1d(p, i, j + 1, X) + at1d(p, i, j - 1, X) - at1d(rhs, i, j, X)) / 4.0;
            }
        }
    }
}

void pressure_GS(f64rw u, f64rw p, f64ro rhs, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    bcp(p, X);

    const double two_dx = 2 * dx;
    const double d2 = dx * dx;
    const double omega = 2.0 / (1.0 + sin(M_PI / (double)X));
    const double one_m_omega = 1.0 - omega;

    const unsigned int iters = X;
    for (unsigned int iter = 0; iter < iters; ++iter) {
        for (size_t i = 1; i < X - 1; ++i) {
            for (size_t j = 1; j < X - 1; ++j) {
                at1d(p, i, j, X) = one_m_omega * at1d(p, i, j, X)
                    + omega * (at1d(p, i + 1, j, X) + at1d(p, i - 1, j, X) + at1d(p, i, j + 1, X) + at1d(p, i, j - 1, X) - at1d(rhs, i, j, X)) * 0.25;
            }
        }
    }
}

void project_velocity(f64rw u, f64rw u_out, double* const p, const size_t X, const double dx, const double K) {
    bcu(u, K, X);
    bcp(p, X);
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(u_out, i, j, 0, X) = at2d(u, i, j, 0, X) - (at1d(p, i + 1, j, X) - at1d(p, i - 1, j, X)) / (2 * dx);
            at2d(u_out, i, j, 1, X) = at2d(u, i, j, 1, X) - (at1d(p, i, j + 1, X) - at1d(p, i, j - 1, X)) / (2 * dx);
        }
    }
}

void assemble_u(f64ro u, f64rw u_out, f64ro conv, f64ro visc, cd factor, const size_t X) {
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(u_out, i, j, 0, X) =  at2d(u, i, j, 0, X) + factor * (-at2d(conv, i, j, 0, X) + at2d(visc, i, j, 0, X));
            at2d(u_out, i, j, 1, X) =  at2d(u, i, j, 1, X) + factor * (-at2d(conv, i, j, 1, X) + at2d(visc, i, j, 1, X));
        }
    }
}

void compute_divergence(f64rw out, f64ro u, dim X, cd dx) {
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(out, i, j, 0, X) = (at2d(u, i + 1, j, 0, X) - at2d(u, i - 1, j, 0, X)) / 2 * dx;
            at2d(out, i, j, 1, X) = (at2d(u, i, j + 1, 1, X) - at2d(u, i, j - 1, 1, X)) / 2 * dx;
        }
    }
}

void debug_apply_bcu(f64rw target, char* label, simulation* sim) {
    dim X = sim->X;
    cd dx = sim->dx;

    f64rw divergence = sim->udebug1;
    compute_divergence(divergence, target, X, dx);
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            cd u = at2d(divergence, i, j, 0, X);
            cd v = at2d(divergence, i, j, 1, X);
            cd norm = sqrt(u * u + v * v);
            if (norm < min) {
                min = norm;
            }
            if (norm > max) {
                max = norm;
            }
            mean += norm;
        }
    }

    mean /= (X * X);
    printf("Divergence %s : min|avg|max %.3f %.3f %.3f\n", label, min, mean, max);

}

void debug(f64ro u, const size_t X){

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

void field_add(f64rw out, f64ro a, cd b_factor, f64ro b, dim X) {
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(out, i, j, 0, X) = at2d(a, i, j, 0, X) + b_factor * at2d(b, i, j, 0, X);
            at2d(out, i, j, 1, X) = at2d(a, i, j, 1, X) + b_factor * at2d(b, i, j, 1, X);
        }
    }
}

void step_EE(simulation* sim){

    convection(sim->u, sim->ubuf1, sim->X, sim->dx, sim->K);
    viscous_drag(sim->u, sim->ubuf2, sim->nu, sim->X, sim->dx, sim->K);

    assemble_u(sim->u, sim->u, sim->ubuf1, sim->ubuf2, sim->dt, sim->X);
    compute_pressure_rhs(sim->u, sim->pbuf1, sim->X, sim->dx);
    // pressure_jacobi(sim->u, sim->pbuf1, sim->p, sim->pbuf2, sim->X, sim->dx, sim->K);
    pressure_GS(sim->u, sim->p, sim->pbuf1, sim->X, sim->dx, sim->K);

    project_velocity(sim->u, sim->u, sim->p, sim->X, sim->dx, sim->K);

}


void rk4_f(f64rw out, f64rw u, f64rw conv, f64rw visc, dim X, cd dx, cd K, cd nu) {
    convection(u, conv, X, dx, K);
    viscous_drag(u, visc, nu, X, dx, K);
    field_add(out, visc, -1.0, conv, X);
}

void rk4_combine_k(f64rw out, f64rw u, f64ro k1, f64ro k2, f64ro k3, f64ro k4, cd factor, dim X) {
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(out, i, j, 0, X) = at2d(u, i, j, 0, X) + factor * (at2d(k1, i, j, 0, X) + 2.0 * at2d(k2, i, j, 0, X) + 2.0 * at2d(k3, i, j, 0, X) + at2d(k4, i, j, 0, X));
            at2d(out, i, j, 1, X) = at2d(u, i, j, 1, X) + factor * (at2d(k1, i, j, 1, X) + 2.0 * at2d(k2, i, j, 1, X) + 2.0 * at2d(k3, i, j, 1, X) + at2d(k4, i, j, 1, X));
        }
    }
}

void apply_pressure(f64rw out, f64rw u, f64rw p, f64rw rhs, dim X, cd dx, cd K) {
    compute_pressure_rhs(u, rhs, X, dx);
    pressure_GS(u, p, rhs, X, dx, K);
    project_velocity(u, out, p, X, dx, K);
}

void step_RK4(simulation* sim) {

    dim X = sim->X;
    cd dx = sim->dx;
    cd K = sim->K;
    cd nu = sim->nu;
    cd dt = sim->dt;

    f64rw un = sim->u;

    f64rw conv = sim->ubuf1;
    f64rw visc = sim->ubuf2;

    f64rw k1 = sim->ubuf3;
    f64rw k2 = sim->ubuf4;
    f64rw k3 = sim->ubuf5;
    f64rw k4 = sim->ubuf6;

    f64rw p = sim->p;
    f64rw rhs = sim->pbuf1;

    rk4_f(k1, un, conv, visc, X, dx, K, nu);

    field_add(k2, un, 0.5 * dt, k1, X);
    rk4_f(k2, k2, conv, visc, X, dx, K, nu);

    field_add(k3, un, 0.5 * dt, k2, X);
    rk4_f(k3, k3, conv, visc, X, dx, K, nu);

    field_add(k4, un, dt, k3, X);
    rk4_f(k4, k4, conv, visc, X, dx, K, nu);

    rk4_combine_k(un, un, k1, k2, k3, k4, 0.166666666667 * dt, X);

    apply_pressure(un, un, p, rhs, X, dx, K);

}

void mult_field(f64rw out, f64rw in, cd factor, dim X) {
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(out, i, j, 0, X) = factor * at2d(in, i, j, 0, X);
            at2d(out, i, j, 1, X) = factor * at2d(in, i, j, 1, X);
        }
    }

}

void imex_f(f64rw out, f64rw u, f64rw conv_buf, dim X, cd dx, cd K) {
    convection(u, conv_buf, X, dx, K);
    // field_add(out, u, -1.0 * dt, out, X);
    mult_field(out, conv_buf, -1.0, X);
}

// https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
// a lower diag, b mid diag, c upper diag
// a_i x_i-1 + b_i x_i + c_i x_i+1 = d_i
//
// here a = c; a, b, c constant
void thomas_algorithm(f64rw out, cd a, cd b, f64ro d, f64rw cprime, f64rw dprime, dim X, dim ld) {
    cprime[0] = a / b;
    dprime[0] = d[0 * ld] / b;

    for (int i = 1; i < X; ++i) {
        cprime[i] = a / (b - a * cprime[i - 1]);
        dprime[i] = (d[i * ld] - a * dprime[i - 1]) / (b - a * cprime[i - 1]);
    }

    out[(X - 1) * ld] = dprime[X - 1];
    for (int i = X - 2 ; i >= 0; --i) {
        out[i * ld] = dprime[i] - cprime[i] * out[(i + 1) * ld];
    }
}

// Helmholtz solve with alpha = 1, beta = nu dt
// Implicit Euler
// (I - νΔt ∇2) un+1 = un
// A X = B
// Implicit Euler ADI
// (I - νΔt ∂xx)(I - νΔt ∂yy) un+1 = un
void ie_adi(f64rw out, f64rw u, f64rw inter_u, f64rw xbuf1, f64rw xbuf2, dim X, cd K, cd dx, cd nu, cd dt) {
    cd a = 1 * nu * dt / (dx * dx);
    cd b = -2 * nu * dt / (dx * dx);

    bcu(u, K, X);
    for (int j = 1; j < X - 1; ++j) {
        // when changing column, leading dimension is (X) * 2
        thomas_algorithm(&at2d(inter_u, 0, j, 0, X), -a, 1 - b, &at2d(u, 0, j, 0, X), xbuf1, xbuf2, X, (X) * 2);
        thomas_algorithm(&at2d(inter_u, 0, j, 1, X), -a, 1 - b, &at2d(u, 0, j, 1, X), xbuf1, xbuf2, X, (X) * 2);
    }

    bcu(inter_u, K, X);
    for (int i = 1; i < X - 1; ++i) {
        thomas_algorithm(&at2d(out, i, 0, 0, X), -a, 1 - b, &at2d(inter_u, i, 0, 0, X), xbuf1, xbuf2, X, 2);
        thomas_algorithm(&at2d(out, i, 0, 1, X), -a, 1 - b, &at2d(inter_u, i, 0, 1, X), xbuf1, xbuf2, X, 2);
    }
}

// expects bcu already applied
// computes: out = u + factor * (u,yy)
void u_plus_dyy(f64rw out, f64ro u, cd factor, dim X, cd dx) {
    cd inner_factor = factor * 1.0 / (dx * dx);
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(out, i, j, 0, X) = at2d(u, i, j, 0, X) + inner_factor * (at2d(u, i, j + 1, 0, X) + at2d(u, i, j - 1, 0, X) - 2 * at2d(u, i, j, 0, X));
            at2d(out, i, j, 1, X) = at2d(u, i, j, 1, X) + inner_factor * (at2d(u, i, j + 1, 1, X) + at2d(u, i, j - 1, 1, X) - 2 * at2d(u, i, j, 1, X));
        }
    }
}

// expects bcu already applied
// computes: out = u + factor * (u,xx)
void u_plus_dxx(f64rw out, f64ro u, cd factor, dim X, cd dx) {
    cd inner_factor = factor * 1.0 / (dx * dx);
    for (size_t i = 1; i < X - 1; ++i) {
        for (size_t j = 1; j < X - 1; ++j) {
            at2d(out, i, j, 0, X) = at2d(u, i, j, 0, X) + inner_factor * (at2d(u, i + 1, j, 0, X) + at2d(u, i - 1, j, 0, X) - 2 * at2d(u, i, j, 0, X));
            at2d(out, i, j, 1, X) = at2d(u, i, j, 1, X) + inner_factor * (at2d(u, i + 1, j, 1, X) + at2d(u, i - 1, j, 1, X) - 2 * at2d(u, i, j, 1, X));
        }
    }
}


void cn_adi(f64rw out, f64rw u, f64rw inter_u, f64rw deri_buf, f64rw xbuf1, f64rw xbuf2, dim X, cd K, cd dx, cd nu, cd dt) {
    cd theta = 0.5;
    cd a = 1 * theta * nu * dt / (dx * dx);
    cd b = -2 * theta * nu * dt / (dx * dx);

    bcu(u, K, X);
    // u[:, j, 0] => u[:, j] + 1/2 * self.dt * self.nu * ((u[:, j+1] - 2 * u[:, j] + u[:, j-1]) / self.dy**2)
    // &at2d(u, 0, j, 0, X) => dyy func
    u_plus_dyy(deri_buf, u, theta * nu * dt, X, dx);
    bcu(deri_buf, K, X);
    for (int j = 1; j < X - 1; ++j) {
        // when changing column, leading dimension is (X) * 2
        thomas_algorithm(&at2d(inter_u, 0, j, 0, X), -a, 1 - b, &at2d(deri_buf, 0, j, 0, X), xbuf1, xbuf2, X, (X) * 2);
        thomas_algorithm(&at2d(inter_u, 0, j, 1, X), -a, 1 - b, &at2d(deri_buf, 0, j, 1, X), xbuf1, xbuf2, X, (X) * 2);
    }

    bcu(inter_u, K, X);
    u_plus_dxx(deri_buf, inter_u, theta * nu * dt, X, dx);
    bcu(deri_buf, K, X);
    for (int i = 1; i < X - 1; ++i) {
        thomas_algorithm(&at2d(out, i, 0, 0, X), -a, 1 - b, &at2d(deri_buf, i, 0, 0, X), xbuf1, xbuf2, X, 2);
        thomas_algorithm(&at2d(out, i, 0, 1, X), -a, 1 - b, &at2d(deri_buf, i, 0, 1, X), xbuf1, xbuf2, X, 2);
    }
}


void step_IMEX(simulation* sim) {

    dim X = sim->X;
    cd dx = sim->dx;
    cd K = sim->K;
    cd nu = sim->nu;
    cd dt = sim->dt;

    f64rw un = sim->u;

    f64rw ubuf1 = sim->ubuf1;
    f64rw visc = sim->ubuf2;

    f64rw k1 = sim->ubuf3;
    f64rw k2 = sim->ubuf4;
    f64rw k3 = sim->ubuf5;
    f64rw k4 = sim->ubuf6;
    f64rw deriv_buf = sim->ubuf7;

    f64rw p = sim->p;
    f64rw rhs = sim->pbuf1;

    f64rw x1 = sim->xbuf1;
    f64rw x2 = sim->xbuf2;

    imex_f(k1, un, ubuf1, X, dx, K);
    field_add(k2, un, 0.5 * dt, k1, X);

    imex_f(k2, k2, ubuf1, X, dx, K);

    field_add(k3, un, 0.5 * dt, k2, X);
    imex_f(k3, k3, ubuf1, X, dx, K);

    field_add(k4, un, dt, k3, X);
    imex_f(k4, k4, ubuf1, X, dx, K);

    rk4_combine_k(un, un, k1, k2, k3, k4, 0.166666666667 * dt, X);

    apply_pressure(un, un, p, rhs, X, dx, K);

    // now solve EI ADI

    // ie_adi(visc, un, ubuf1, x1, x2, X, K, dx, nu, dt);
    cn_adi(visc, un, ubuf1, deriv_buf, x1, x2, X, K, dx, nu, dt);

    // Reapply pressure
    apply_pressure(un, visc, p, rhs, X, dx, K);

}

void loop(simulation* sim, const size_t steps, const unsigned int write_interval) {
    connect_mmap(sim->X);
    for (unsigned int i = 0; i < steps; ++i) {
        // step_EE(sim);
        // step_RK4(sim);
        step_IMEX(sim);
        if (i % write_interval == 0) {
            write_u(sim->u, sim->X, i);
            printf("i: %d\n", i);
            debug(sim->u, sim->X);
        }
    }
}

typedef struct {
    double re;
    int N;
    int steps;
    int sampling;
    double K;
    double dt;
    char* scheme;
} simulation_parameters;

void clean_simulation_parameters(simulation_parameters* params) {
    free(params->scheme);
}

static int handler(void* user, const char* section, const char* name, const char* value){
    simulation_parameters* params = (simulation_parameters*)user;

    #define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0
    if (MATCH("Simulation", "re")) {
        params->re = atof(value);
    } else if (MATCH("Simulation", "n")) {
        params->N = atoi(value);
    } else if (MATCH("Simulation", "steps")) {
        params->steps = atoi(value);
    } else if (MATCH("Simulation", "sampling")) {
        params->sampling = atoi(value);
    } else if (MATCH("Simulation", "k")) {
        params->K = atof(value);
    } else if (MATCH("Simulation", "dt")) {
        params->dt = atof(value);
    } else if (MATCH("Simulation", "scheme")) {
        params->scheme = strdup(value);
    } else {
        return 0;
    }
    return 1;
}



simulation_parameters load_from_file(char* filepath) {
    simulation_parameters params;

    if (ini_parse(filepath, handler, &params) < 0) {
        fprintf(stderr, "Could not read simulation parameters.");
        exit(EXIT_FAILURE);
    }

    return params;
}

int main(int argc, char** argv) {

    UNUSED(argc);
    UNUSED(argv);

    printf("Starting simulation\n");

    simulation_parameters params = load_from_file("simulation.ini");
    simulation sim = init_simulation(params.re, params.dt, params.N, params.K, 0);
    init_u(sim.u, sim.X, 0.0001);
    init_p(sim.p, sim.X, 0.0001);

    loop(&sim, params.steps, params.sampling);

    clean_simulation_parameters(&params);
    return EXIT_SUCCESS;
}
