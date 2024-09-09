// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0042 * 100;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;

a = 0.8 * Pi * scale;
b = (14/15) * Pi * scale;
c = 0.4 * Pi * scale;

Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Point(5) = {cx-b/2, cy - a/2, 0, mesh_size0};
Point(6) = {cx+b/2, cy - a/2, 0, mesh_size0};
Point(7) = {cx+c/2, cy + a/2, 0, mesh_size0};
Point(8) = {cx-c/2, cy + a/2, 0, mesh_size0};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, Pi} {
  Curve{8}; Curve{5}; Curve{6}; Curve{7}; 
}

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {6, 7, 8, 5};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}//+//+

Physical Surface("inlet", 25) = {2};
Physical Surface("outlet", 26) = {4};
Physical Surface("obstacle", 27) = {8, 7, 6, 9};
Physical Surface("wall", 30) = {3, 5};
Physical Surface("frontAndBack", 28) = {1, 10};
Physical Volume("volume", 29) = {1};
