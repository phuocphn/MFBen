// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0044 * 100;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;

Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

R = 0.8*0.5*Pi*scale;
Circle(5) = {cx, cy, 0, R, 0, 2*Pi};
Characteristic Length {5} = mesh_size0;

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}

Physical Surface("inlet", 16) = {2};
Physical Surface("outlet", 17) = {4};
Physical Surface("obstacle", 18) = {6};
Physical Surface("wall", 19) = {5, 3};
Physical Surface("frontAndBack", 20) = {7, 1};
Physical Volume("volume", 21) = {1};
