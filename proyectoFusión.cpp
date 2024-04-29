//Dudas
//1. Hay 2 formas de definir ser_ref_levels, cual escoger? (YA)
//2. Para definir los elementos del espacio finito se pues hacer 
// como lo puse aca o como esta entre las lineal 160-189 del ex1p 
//3. Definición de Kappa? Como definirla de acuerdo al volumen (Código de Boyan)


//  Anotaciones
// Paso 7.b importante para cuando definamos la matriz. Lo dejamos?



#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// Definición de funciones y procesos para el desarrollo



// Declaración de funciones







// double myFun1(); // Declaración de myFun1 con un parámetro para el volumen

// double funCoef();        // kappa coefficient.

//double corn_nbc(double Vector &x, double Vector &k); // Declaración de corn_nbc con dos parámetros: E y kappa_l

//double escl_nbc();       //Function 0 inhomogeneous Neumann BC.

//double f_analytic();     //-Div(kappa Delta T) = f = 0. ----> = 0


//Mesh * GenerateSerialMesh(int ref);

int main(int argc, char *argv[])
{
    //1. Initialize MPI and HYPRE.
    Mpi::Init();
    if (!Mpi::Root()){mfem::out.Disable(); mfem::err.Disable();}
    Hypre::Init();

    //2. Parse command-line options.
    //const char *mesh_file = "../data/ojo5.msh";
    const char *mesh_file = "../data/ojo-Fusion.msh";
    int ser_ref_levels = 2;
    int par_ref_levels = 2;
    int order = 1;
    bool visualization = true;

    //Array<int> ess_tdof_list(0);
    
    //const double &E = 40.0; 
    //const double &kappa_l = 1.0;


    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(mfem::out);
        return 1;
    }
   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ser_ref_levels =  (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);  // ya declarado arriba pero asi ?
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   
   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 4;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }
   // 5. Define a parallel finite element space on the parallel mesh. 
   //    Here we use continuous Lagrange finite elements of the
   //    specified order.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   mfem::out << "Number of finite element unknowns" << size << endl;

   // 6. Create "marker arrays" to define the portions of boundary associated
   //    with each type of boundary condition. These arrays have an entry
   //    corresponding to each boundary attribute. Placing a '1' in entry i
   //    marks attribute i+1 as being active, '0' is inactive.

   //Array<int> nbc_corn(pmesh.bdr_attributes.Max());
   //Array<int> dbc_scle(pmesh.bdr_attributes.Max());
   
  // nbc_corn = 0; nbc_corn[0] = 1;
   //dbc_scle = 1; dbc_scle[1] = 1;

   
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   ConstantCoefficient Kappa(1.0);

   //Array<int> ess_bdr;
   //ess_bdr.SetSize(pmesh.bdr_attributes.Max());
   //fespace.GetEssentialTrueDofs(dbc_scle, ess_tdof_list);



   //7. Setup the various coefficients needed for the Laplace operator and the
   //    various boundary conditions.
   //FunctionCoefficient p(funCoef);
   //ParGridFunction kappa1(&fespace); //Let's add the coefficient kappa
   //kappa1.ProjectCoefficient(p);
   //GridFunctionCoefficient kappa(&kappa1);

   ConstantCoefficient corn_nbc(40.0);
   ConstantCoefficient escl_nbc(0.0);
   ConstantCoefficient f_analytic(1.0);

   //7.a Now we convert the boundary conditions and the f in the Laplace's
   // equation in to a FunctionCoeffcient. 
   //FunctionCoefficient g1(corn_nbc);
   //FunctionCoefficient g2(escl_nbc);
   //FunctionCoefficient f_an(f_analytic);

   // 7.b Since the n.Grad(u) terms arise by integrating -Div(rho Grad(u)) by parts we
   // must introduce the coefficient 'rho' (rho) into the boundary conditions.
   // Therefore, in the case of the Neumann BC, we actually enforce rho n.Grad(u)
   // = rho g rather than simply n.Grad(u) = g.
   ProductCoefficient m_nbc1Coef(corn_nbc, Kappa);
   ProductCoefficient m_nbc2Coef(escl_nbc, Kappa);

   // 8. Define the solution vector u as a parallel finite element grid function
   //    corresponding to fespace. Initialize u with initial guess of zero.
   ParGridFunction T(&fespace);
   T = 0.0;

   // 9. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm a(&fespace);
   //BilinearFormIntegrator *integ = new MixedGradGradIntegrator(Kappa);
   BilinearFormIntegrator *integ = new DiffusionIntegrator(Kappa);
   a.AddDomainIntegrator(integ);
   BilinearFormIntegrator *integ2 = new MassIntegrator(Kappa);
   a.AddDomainIntegrator(integ2);
   a.Assemble();

   // 10. Assemble the parallel linear form for the right hand side vector.

   // Set the Dirichlet values in the solution vector |
   //ParLinearForm b(&fespace);                       |--> Necesario?   
   //T.ProjectBdrCoefficient(g3,dbc_bdr3);            |  

   ParLinearForm b(&fespace); 
   //Add the f parameter to the vector b
   b.AddDomainIntegrator(new DomainLFIntegrator(f_analytic));
   //Add the desired value for n.Grad(T) on the Neumann Boundary 1
   //b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbc1Coef),nbc_corn);
   //Add the desired value for n.Grad(T) on the Neumann Boundary 2
   // b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbc2Coef),dbc_scle);
   // Assemble
   b.Assemble();


    // 11. Construct the linear system.
    OperatorPtr A;
    Vector B, X;
    //a.FormLinearSystem(ess_tdof_list, T, b, A, X, B);  
   a.FormLinearSystem(ess_tdof_list, T, b, A, X, B);  
    // 12. Solver the linear system AX=B.
    HypreSolver *amg = new HypreBoomerAMG;
    HyprePCG pcg(MPI_COMM_WORLD);
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(200);
    pcg.SetPrintLevel(2);
    pcg.SetPreconditioner(*amg);
    pcg.SetOperator(*A);
    pcg.Mult(B, X);
    delete amg;

    // 13. Recover the parallel grid function corresponding to T. This is the
    //     local finite element solution on each processor.
    a.RecoverFEMSolution(X, b, T);
    // 14.1 Compute the H^1 norms of the error.
    //double h1_err_prev = 0.0;
    //double h_prev = 0.0;
    //double h1_err = T.ComputeH1Error(&T1,&T_grad,&p,1.0,1.0);
    //mfem::out <<"Calculate of the error: "  << h1_err << endl;

    // 15. Save the refined mesh and the solution in parallel. This output can be
    //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
    {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << Mpi::WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      T.Save(sol_ofs);
   }
   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string title_str = "H1";
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize()
               << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << T
               << "window_title '" << title_str << " Solution'"
               << " keys 'mmc'" << flush;
      socketstream exact_sol(vishost, visport);
 
      
   }

 ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("proyecto1bbb", &pmesh);
   pd->SetPrefixPath("ParaView");
   pd->RegisterField("solution", &T);
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();



   // 17. Free the used memory.
   delete fec;

   return 0; 
}



//double myFun1(double kk)
//{
//    return kk; // Esta función devuelve 1.0, puedes cambiarla para que dependa del volumen si es necesario
//}                  


//double funCoef(double kk)
//{
//    return myFun1(double kk); 
//}

//We are going to define the functions in order to get the boundary condition
// Definir constantes


//double corn_nbc(double E, double kappa_l)
//{
//    return E / kappa_l; // Retorna el resultado de E dividido por kappa_l
//}

//double escl_nbc( )
//{
//    return(0.0);
//}

//double f_analytic()
//{
//   return(0.0);       
//}





