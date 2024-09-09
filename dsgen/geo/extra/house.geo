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



Point(7) = {cx+Ri/2, cy-side_length/2, 0, mesh_size0};
Point(8) = {cx+Ri/2, cy+side_length/4, 0, mesh_size0};
Point(9) = {cx+Ri, cy+side_length/4, 0, mesh_size0};
// Point(9) = {cx+Ri, cy+side_length/2, 0, mesh_size0};
Point(10) = {cx, cy+Ri, 0, mesh_size0};


Point(11) = {cx-Ri/2, cy-side_length/2, 0, mesh_size0};

Point(12) = {cx-Ri/2, cy+side_length/4, 0, mesh_size0};
Point(13) = {cx-Ri, cy+side_length/4, 0, mesh_size0};

Line(5) = {10, 9};
Line(6) = {9, 8};
Line(7) = {8, 7};
Line(8) = {7, 11};
Line(9) = {11, 12};
Line(10) = {12, 13};
Line(11) = {13, 10};
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {11, 5, 6, 7, 8, 9, 10};
//+
Plane Surface(1) = {1, 2};
Extrude {0, 0, 0.0005} {
  Surface{1}; 
  Layers{1};
  Recombine;
}//+//+
Physical Surface("inlet", 34) = {2};
Physical Surface("outlet", 35) = {4};
Physical Surface("wall", 36) = {3, 5};
Physical Surface("obstacle", 37) = {6, 7, 8, 9, 10, 11, 12};
Physical Surface("frontAndBack", 38) = {1, 13};
Physical Volume("volume", 39) = {1};
