
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



class MyMatrixCoefficient:public mfem::MatrixCoefficient
{
    public:
    //constructor
    MyMatrixCoefficient(int dim,double a_,double b_):mfem::MatrixCoefficient(dim)
    {
        a=a_;
        b=b_;
    }
virtual
void Eval(mfem::DenseMatrix&K,mfem::ElementTransformation &pmesh,const mfem::IntegrationPoint &ip)
{//initializate K to 0.0 everywhere
K=0.0;
//if in domain 1
if(pmesh.Attribute==3){
    for(int i=0; i<mfem::MatrixCoefficient::GetHeight();i++){
        K(i,i)=a; 
    }
}if(pmesh.Attribute==4){
  //else for other clases
  for(int i=0;i<mfem::MatrixCoefficient::GetHeight();i++){
    K(i,i)=b;
  }
}
}
private:
//add any local data here
double a,b;
};



int main(int argc, char *argv[])
{
    //1. Initialize MPI and HYPRE.
    Mpi::Init();
    if (!Mpi::Root()){mfem::out.Disable(); mfem::err.Disable();}
    Hypre::Init();

    //2. Parse command-line options.

    const char *mesh_file = "../data/HumanEyeMesh.msh"; 

    int ser_ref_levels = 2;
    int par_ref_levels = 2;
    int order = 1;
    bool visualization = true;

    Array<int> ess_tdof_list(0);
    
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
      int par_ref_levels = 2;
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


   Array<int> rbc_corn(pmesh.bdr_attributes.Max());
   Array<int> rbc_scle(pmesh.bdr_attributes.Max());
   
   rbc_corn = 0; rbc_corn[0] = 1;
   rbc_scle = 0; rbc_scle[1] = 1;
   
   
   MyMatrixCoefficient Kappa(pmesh.Dimension(), 0.58,0.8);  // Duda Boyan    

   //7. Setup the various coefficients needed for the Laplace operator and the
   //    various boundary conditions.
   
   //Constantes fisicas:  
   ConstantCoefficient E_corn(-40.0);
   ConstantCoefficient alpha_a(10.0);
   ConstantCoefficient alpha_nega(-10.0);
   ConstantCoefficient T_a(25.0);
   ConstantCoefficient T_bl(37.0); 
   ConstantCoefficient alpha_bl(65.0);
   ConstantCoefficient alpha_negbl(-65.0);
   ConstantCoefficient f_analytic(0.0);

   //CoeficientesparaelB
   ProductCoefficient rbc1Coef(alpha_a,T_a);
   ProductCoefficient rbc2Coef(alpha_bl, T_bl);

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
   a.AddBoundaryIntegrator(new MassIntegrator(alpha_a), rbc_corn);
   a.AddBoundaryIntegrator(new MassIntegrator(alpha_bl), rbc_scle);
   a.Assemble();


   // 10. Assemble the parallel linear form for the right hand side vector.
        
   ParLinearForm b(&fespace); 
   //Add the f parameter to the vector b
   b.AddDomainIntegrator(new DomainLFIntegrator(f_analytic));
   //Add the desired value for n.Grad(T) on the Neumann Boundary 1
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(rbc1Coef),rbc_corn);  
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(E_corn),rbc_corn);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(rbc2Coef),rbc_scle);
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
   pd = new ParaViewDataCollection("HeatDistributionRobin", &pmesh);
   //pd = new ParaViewDataCollection("proyectoFusion1", &pmesh);
   //  pd = new ParaViewDataCollection("proyectoFusion", &pmesh);
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






