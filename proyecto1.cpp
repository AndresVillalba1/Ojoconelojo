//Dudas
//1. Hay 2 formas de definir ser_ref_levels, cual escoger?
//2. Para definir los elementos del espacio finito se pues hacer 
// como lo puse aca o como esta entre las lineal 160-189 del ex1p 


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// Definición de funciones y procesos para el desarrollo

double funCoef(const Vector &x); //Rho coefficient.




//
Mesh * GenerateSerialMesh(int ref);

int main(int argc, char *argv[])
{
    //1. Initialize MPI and HYPRE.
    Mpi::Init();
    if (!Mpi::Root()){mfem::out.Disable(); mfem::err.Disable();}
    Hypre::Init();

    //2. Parse command-line options.
    const char *mesh_file = "../data/OJO.msh";
    int ser_ref_levels = 2;
    int par_ref_levels = 4;
    int order = 1;
    bool visualization = true;

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

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      // int par_ref_levels = 2; ya declarado arriba
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
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

   Array<int> nbc_corn(pmesh.bdr_attributes.Max());
   Array<int> nbc_scle(pmesh.bdr_attributes.Max());
   
   nbc_corn = 0; nbc_corn[0] = 1;
   nbc_scle = 0; nbc_scle[1] = 1;

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> nbc_corn(pmesh.bdr_attributes.Max());
      ess_bdr = 15;  // 15 es un número ejemplo 
      fespace1.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   }


   



    












}


