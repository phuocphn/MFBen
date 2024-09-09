// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.0047 * 100;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;
a = 1.04 * Pi * scale;
b = 0.4 * Pi * scale;

Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Point(5) = {cx-a/2, cy, 0, mesh_size0};
Point(6) = {cx+a/2, cy, 0, mesh_size0};
Point(7) = {cx, cy-b/2, 0, mesh_size0};
Point(8) = {cx, cy, 0, mesh_size0};
Point(9) = {cx, cy+b/2, 0, mesh_size0};

Ellipse(5) = {5, 8, 6, 7};
Ellipse(6) = {6, 8, 5, 7};
Ellipse(7) = {5, 8, 6, 9};
Ellipse(8) = {6, 8, 5, 9};
//+
Rotate {{0, 0, 1}, {2*Pi*scale, 2*Pi*scale, 0}, Pi*5/6} {
  Curve{7}; Curve{8}; Curve{6}; Curve{5}; 
}

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {7, -8, 6, -5};
Plane Surface(1) = {1, 2};

Extrude {0, 0, 0.005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}//+
Physical Surface("inlet", 25) = {2};
Physical Surface("outlet", 26) = {4};
Physical Surface("obstacle", 27) = {6, 9, 8, 7};
Physical Surface("wall", 28) = {3};
Physical Surface("wall", 28) += {5};
Physical Surface("frontAndBack", 29) = {10, 1};
Physical Volume("volume", 30) = {1};
