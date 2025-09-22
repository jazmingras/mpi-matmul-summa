#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ===== Utilidades ===== */
static double* alloc_mat(int n){
    double *m = (double*)malloc((size_t)n*(size_t)n*sizeof(double));
    if(!m){ fprintf(stderr,"Error: sin memoria\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    return m;
}
static void fill_random(double *m,int n,unsigned seed){
    srand(seed);
    for(int i=0;i<n*n;i++) m[i] = (double)rand()/(double)RAND_MAX;
}
static void zero(double *m,int n){ for(int i=0;i<n*n;i++) m[i]=0.0; }

static void write_matrix_txt(const char* path, const double* M, int n){
    FILE *f=fopen(path,"w");
    if(!f){ fprintf(stderr,"No se pudo abrir %s\n",path); return; }
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++) fprintf(f,"%.6f%s", M[i*n+j], (j+1==n)?"":" ");
        fputc('\n',f);
    }
    fclose(f);
}

/* Tipo MPI para submatriz nloc x nloc dentro de matriz NxN (row-major) */
static void create_block_type(int N, int block, MPI_Datatype *block_t){
    MPI_Datatype tmp;
    MPI_Type_vector(block, block, N, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(double), block_t);
    MPI_Type_free(&tmp);
}

/* Offsets de bloques 2D (en “doubles”) para Scatterv/Gatherv */
static void build_2d_block_displs_counts(int q,int N,int block,int *counts,int *displs){
    int idx=0;
    for(int bi=0; bi<q; ++bi){
        for(int bj=0; bj<q; ++bj){
            counts[idx]=1;
            displs[idx]= bi*(N*block) + bj*block; /* índice del (0,0) del bloque */
            idx++;
        }
    }
}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int r,p; MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&p);

    if(argc<2){ if(r==0) fprintf(stderr,"Uso: mpirun -np P %s N [out]\n",argv[0]); MPI_Abort(MPI_COMM_WORLD,1); }
    const int N = atoi(argv[1]);
    const char* outpath = (argc>=3)? argv[2] : "C.txt";

    int q=(int)round(sqrt((double)p));
    if(q*q!=p){ if(r==0) fprintf(stderr,"P debe ser cuadrado perfecto. P=%d\n",p); MPI_Abort(MPI_COMM_WORLD,1); }
    if(N%q!=0){ if(r==0) fprintf(stderr,"N (%d) debe ser múltiplo de sqrt(P)=%d\n",N,q); MPI_Abort(MPI_COMM_WORLD,1); }

    int dims[2]={q,q}, periods[2]={0,0}, coords[2];
    MPI_Comm grid,row_comm,col_comm;
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,1,&grid);
    MPI_Cart_coords(grid,r,2,coords);
    MPI_Comm_split(grid, coords[0], coords[1], &row_comm);
    MPI_Comm_split(grid, coords[1], coords[0], &col_comm);

    const int block = N/q, nloc = block;

    double *A=NULL,*B=NULL,*C=NULL;
    if(r==0){
        A=alloc_mat(N); B=alloc_mat(N); C=alloc_mat(N);
        fill_random(A,N,(unsigned)time(NULL));
        fill_random(B,N,(unsigned)time(NULL)+7);
    }

    double *A_loc=(double*)malloc((size_t)nloc*nloc*sizeof(double));
    double *B_loc=(double*)malloc((size_t)nloc*nloc*sizeof(double));
    double *C_loc=(double*)malloc((size_t)nloc*nloc*sizeof(double));
    if(!A_loc||!B_loc||!C_loc){ fprintf(stderr,"Sin memoria local\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    zero(C_loc,nloc);

    MPI_Datatype block_t; create_block_type(N,block,&block_t); MPI_Type_commit(&block_t);
    int *counts=NULL,*displs=NULL;
    if(r==0){ counts=(int*)malloc(p*sizeof(int)); displs=(int*)malloc(p*sizeof(int)); build_2d_block_displs_counts(q,N,block,counts,displs); }

    MPI_Scatterv(A,counts,displs,block_t, A_loc,nloc*nloc,MPI_DOUBLE, 0,MPI_COMM_WORLD);
    MPI_Scatterv(B,counts,displs,block_t, B_loc,nloc*nloc,MPI_DOUBLE, 0,MPI_COMM_WORLD);

    double *A_panel=(double*)malloc((size_t)nloc*nloc*sizeof(double));
    double *B_panel=(double*)malloc((size_t)nloc*nloc*sizeof(double));
    if(!A_panel||!B_panel){ fprintf(stderr,"Sin memoria panel\n"); MPI_Abort(MPI_COMM_WORLD,1); }

    int row_rank,col_rank; MPI_Comm_rank(row_comm,&row_rank); MPI_Comm_rank(col_comm,&col_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0=MPI_Wtime();

    for(int k=0;k<q;++k){
        if(coords[1]==k) for(int i=0;i<nloc*nloc;i++) A_panel[i]=A_loc[i];
        MPI_Bcast(A_panel,nloc*nloc,MPI_DOUBLE, k, row_comm);

        if(coords[0]==k) for(int i=0;i<nloc*nloc;i++) B_panel[i]=B_loc[i];
        MPI_Bcast(B_panel,nloc*nloc,MPI_DOUBLE, k, col_comm);

        for(int i=0;i<nloc;i++){
            for(int kk=0; kk<nloc; kk++){
                double a=A_panel[i*nloc+kk];
                const double* bp=&B_panel[kk*nloc];
                double* cp=&C_loc[i*nloc];
                for(int j=0;j<nloc;j++) cp[j]+= a*bp[j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime()-t0;

    MPI_Gatherv(C_loc,nloc*nloc,MPI_DOUBLE, C,counts,displs,block_t, 0,MPI_COMM_WORLD);

    double elapsed_max; MPI_Reduce(&elapsed,&elapsed_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(r==0){
        printf("N=%d, P=%d (q=%d), Tiempo_total=%.6f s\n", N, p, q, elapsed_max);
        write_matrix_txt(outpath, C, N);
        free(A); free(B); free(C); free(counts); free(displs);
    }

    free(A_loc); free(B_loc); free(C_loc); free(A_panel); free(B_panel);
    MPI_Type_free(&block_t);
    MPI_Comm_free(&row_comm); MPI_Comm_free(&col_comm); MPI_Comm_free(&grid);
    MPI_Finalize(); return 0;
}
