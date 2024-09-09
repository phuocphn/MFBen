// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0045 * 100;
scale=1;

Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

R =0.8*0.5*Pi*scale; 
Point(5) = {2*Pi*scale - R, 2*Pi*scale, 0, mesh_size0};
Point(6) = {2*Pi*scale, 2*Pi*scale-R,0,  mesh_size0};
Point(7) = {2*Pi*scale + R, 2*Pi*scale, 0, mesh_size0};
Point(8) = {2*Pi*scale, 2*Pi*scale+R,0,  mesh_size0};
Point(9) = {2*Pi*scale, 2*Pi*scale,0,  mesh_size0};

Circle(5) = {8, 9, 7};
Circle(6) = {7, 9, 6};
Circle(7) = {6, 9, 5};
Characteristic Length {5} = mesh_size0;
Characteristic Length {6} = mesh_size0;
Characteristic Length {7} = mesh_size0;

Line(8) = {8, 9};
Line(9) = {9, 5};

Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, 0} {
  Curve{9}; Curve{8}; Curve{5}; Curve{6}; Curve{7}; 
}
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {7, -9, -8, 5, 6};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}

//+
Physical Surface("inlet", 28) = {2};
Physical Surface("outlet", 29) = {4};
Physical Surface("obstacle", 30) = {6, 7, 8, 9, 10};
Physical Surface("wall", 31) = {5, 3};
Physical Surface("frontAndBack", 32) = {11, 1};
Physical Volume("volume", 33) = {1};
