// Gmsh project created on Fri Apr 05 21:01:20 2024
SetFactory("OpenCASCADE");
mesh_size0=0.005 * 10;
scale=1;
cx = 2*Pi*scale;
cy = 2*Pi*scale;
side_length=0.8*Pi * (Sqrt(2) -1)* scale;

h=2.414*side_length;
Ri = h/2;
side_length=Ri* scale;


Point(1) = {Pi * scale, Pi * scale, 0, mesh_size0};
Point(2) = {Pi * scale, 3*Pi*scale, 0, mesh_size0};
Point(3) = {3*Pi*scale, 3*Pi*scale, 0, mesh_size0};
Point(4) = {3*Pi*scale, Pi * scale, 0, mesh_size0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



Point(5) = {cx, cy-Ri, 0, mesh_size0};
Point(7) = {cx+Ri/2, cy-side_length/2, 0, mesh_size0};
Point(8) = {cx+Ri/2, cy+side_length/2, 0, mesh_size0};
Point(9) = {cx+Ri, cy, 0, mesh_size0};
Point(10) = {cx, cy+Ri, 0, mesh_size0};


Point(11) = {cx-Ri/2, cy-side_length/2, 0, mesh_size0};
Point(12) = {cx-Ri/2, cy+side_length/2, 0, mesh_size0};
Point(13) = {cx-Ri, cy, 0, mesh_size0};


//+
Spline(6) = {12, 10, 8};
Spline(7) = {8, 9, 7};
Spline(8) = {7, 5, 11};
Spline(9) = {11, 13, 12};
//+

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {9, 6, 7, 8};
Plane Surface(1) = {1, 2};
Extrude {0, 0, 0.0005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}//+//+
Physical Surface("inlet", 26) = {2};
Physical Surface("outlet", 27) = {4};
Physical Surface("wall", 28) = {5, 3};
Physical Surface("obstacle", 29) = {7, 8, 9, 6};
Physical Surface("frontAndBack", 30) = {10, 1};
//+
Physical Volume("volume", 31) = {1};
